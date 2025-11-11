import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import warnings
from typing import List, Tuple, Optional

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="U.S. Contrarian Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 기본/디폴트 티커 (M7)
M7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
DEFAULT_PARAMS = {
    'lookback_months': 3,
    'lookback_days': 63,
    'rebalance_freq': 'M',
    'threshold': -0.3,
    'weight_split': 0.60,
    'min_weight_change': 0.0
}
BENCHMARK_TICKER = 'QQQ'

# -------------------------
# 캐시 / 유틸리티 함수
# -------------------------
@st.cache_data(ttl=3600)
def download_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """주가 데이터 다운로드 (종가). tickers: list of tickers (str)."""
    try:
        if isinstance(tickers, str):
            tickers = [tickers]
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        # Ensure columns are the tickers in same order; if missing, fill with NaN
        data = data.reindex(columns=tickers)
        return data.ffill().bfill()
    except Exception as e:
        st.error(f"데이터 다운로드 실패: {str(e)}")
        return None

@st.cache_data(ttl=86400)
def fetch_ticker_name(ticker: str) -> str:
    """티커의 회사명(가능하면 shortName) 반환, 실패 시 ticker 반환."""
    try:
        info = yf.Ticker(ticker).info
        name = info.get('shortName') or info.get('longName') or ticker
        return name
    except Exception:
        return ticker

@st.cache_data(ttl=86400)
def get_first_available_date(ticker: str) -> Optional[date]:
    """
    티커의 전체 이용 가능한 데이터에서 첫 거래 가능일(종가가 존재하는 첫 날짜)을 반환합니다.
    - yfinance Ticker.history(period='max')를 사용하여 전체 히스토리를 가져옴.
    - 반환 타입을 datetime.date 로 하여 tz 관련 비교 문제를 회피함.
    """
    try:
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        if 'Close' in hist.columns:
            series = hist['Close']
        else:
            series = hist.iloc[:, 0]
        first = series.first_valid_index()
        if first is None:
            return None
        # 반환값을 date 객체로 변환 (tz-naive 비교를 위해)
        return pd.Timestamp(first).date()
    except Exception:
        return None

def calculate_drawdown_from_peak(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    rolling_max = prices.rolling(window=lookback_days, min_periods=1).max()
    return (prices - rolling_max) / rolling_max

def calculate_weights_by_drawdown(drawdowns: pd.Series, threshold: float, weight_split: float) -> pd.Series:
    """하락률 기반 가중치 계산 (Series: index=ticker)."""
    if drawdowns is None or len(drawdowns.dropna()) == 0:
        idx = drawdowns.index if drawdowns is not None else []
        return pd.Series(1.0 / max(1, len(idx)), index=idx)

    drawdowns = drawdowns.dropna()
    idx = drawdowns.index
    weights = pd.Series(0.0, index=idx)

    deep_mask = drawdowns <= threshold
    if deep_mask.any():
        deep = drawdowns[deep_mask].abs()
        others = drawdowns[~deep_mask].abs()
        if deep.sum() > 0:
            weights[deep.index] = (deep / deep.sum()) * weight_split
        remaining = 1 - weight_split
        if len(others) > 0:
            if others.sum() > 0:
                weights[others.index] = (others / others.sum()) * remaining
            else:
                weights[others.index] = remaining / len(others)
    else:
        abs_dd = drawdowns.abs()
        if abs_dd.sum() > 0:
            weights = abs_dd / abs_dd.sum()
        else:
            weights = pd.Series(1.0 / len(idx), index=idx)

    if weights.sum() <= 0:
        weights = pd.Series(1.0 / len(idx), index=idx)
    else:
        weights = weights / weights.sum()
    return weights

def backtest_strategy(prices: pd.DataFrame, lookback_days: int, rebalance_freq: str, threshold: float,
                      weight_split: float, min_weight_change: float = 0.0) -> Tuple[pd.Series, pd.DataFrame]:
    """
    백테스트 수행.
    - prices: DataFrame (index: date, columns: tickers)
    - 반환: portfolio_values (Series indexed by date), weight_history (DataFrame with 'date' column then tickers)
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    if rebalance_freq == 'W':
        reb_dates = prices.resample('W-MON').last().index
    else:
        reb_dates = prices.resample('M').last().index

    reb_actual = []
    for dt in reb_dates:
        if dt in prices.index:
            reb_actual.append(dt)
        else:
            later = prices.index[prices.index >= dt]
            if len(later) > 0:
                reb_actual.append(later[0])
            else:
                earlier = prices.index[prices.index <= dt]
                if len(earlier) > 0:
                    reb_actual.append(earlier[-1])
    reb_actual = sorted(list(dict.fromkeys(reb_actual)))

    portfolio_value = 100.0
    pv_list = []
    pv_dates = []
    weight_history = []
    current_holdings = pd.Series(0.0, index=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if i > 0 and (current_holdings > 0).any():
            portfolio_value = (current_holdings * prices.loc[date]).sum()

        pv_list.append(portfolio_value)
        pv_dates.append(date)

        if date in reb_actual:
            prices_up_to = prices.loc[:date]
            drawdowns = calculate_drawdown_from_peak(prices_up_to, lookback_days)
            if isinstance(drawdowns, pd.DataFrame):
                cur_dd = drawdowns.loc[date]
            else:
                cur_dd = drawdowns
            target_weights = calculate_weights_by_drawdown(cur_dd, threshold, weight_split)

            aligned_target = target_weights.reindex(prices.columns).fillna(0)
            weight_change_sum = (aligned_target - last_weights).abs().sum()

            if last_weights.sum() == 0 or weight_change_sum >= min_weight_change:
                current_prices = prices.loc[date]
                current_holdings = (portfolio_value * aligned_target) / current_prices.replace(0, np.nan)
                current_holdings = current_holdings.fillna(0)
                last_weights = aligned_target
                weight_history.append({'date': date, **{t: last_weights.get(t, 0.0) for t in prices.columns}})
            else:
                if (current_holdings > 0).any():
                    current_value_per_stock = current_holdings * prices.loc[date]
                    if current_value_per_stock.sum() > 0:
                        cur_weights = current_value_per_stock / current_value_per_stock.sum()
                    else:
                        cur_weights = pd.Series(0.0, index=prices.columns)
                else:
                    cur_weights = pd.Series(0.0, index=prices.columns)
                weight_history.append({'date': date, **{t: cur_weights.get(t, 0.0) for t in prices.columns}})

    portfolio_series = pd.Series(pv_list, index=pv_dates).sort_index()
    weight_df = pd.DataFrame(weight_history)
    return portfolio_series, weight_df

def calculate_performance_metrics(value_series: pd.Series, benchmark_series: Optional[pd.Series] = None) -> dict:
    out = {}
    if value_series is None or len(value_series.dropna()) < 2:
        return None

    values = value_series.dropna()
    returns = values.pct_change().dropna()
    if len(returns) == 0:
        return None

    final_value = float(values.iloc[-1])
    initial_value = float(values.iloc[0])
    total_return = (final_value / initial_value - 1) * 100
    n_days = len(returns)
    n_years = n_days / 252.0 if n_days > 0 else 0
    cagr = ((final_value / initial_value) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    volatility = float(returns.std() * np.sqrt(252) * 100)
    returns_std = float(returns.std())
    sharpe = float((returns.mean() * 252) / (returns_std * np.sqrt(252))) if returns_std > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    mdd = float(drawdown.min() * 100) if len(drawdown) > 0 else 0.0
    calmar = float(cagr / abs(mdd)) if abs(mdd) > 0.001 else 0.0

    tracking_error = None
    if benchmark_series is not None and len(benchmark_series.dropna()) > 2:
        bvals = benchmark_series.reindex(values.index).dropna()
        if len(bvals) > 1:
            bret = bvals.pct_change().dropna()
            common_idx = returns.index.intersection(bret.index)
            if len(common_idx) > 1:
                diff = returns.reindex(common_idx) - bret.reindex(common_idx)
                tracking_error = float(diff.std() * np.sqrt(252) * 100)

    out['Total Return (%)'] = total_return
    out['CAGR (%)'] = cagr
    out['Volatility (%)'] = volatility
    out['Sharpe Ratio'] = sharpe
    out['Max Drawdown (%)'] = mdd
    out['Tracking Error (%)'] = tracking_error if tracking_error is not None else np.nan
    out['Calmar Ratio'] = calmar
    return out

def calculate_turnover(weight_history: pd.DataFrame, rebalance_freq: str) -> Tuple[float, float]:
    if weight_history is None or len(weight_history) < 2:
        return 0.0, 0.0
    wh = weight_history.copy()
    if 'date' in wh.columns:
        wh = wh.set_index('date')
    wh = wh.sort_index()
    total_turnover = 0.0
    reb_count = 0
    for i in range(1, len(wh)):
        w_t = wh.iloc[i].fillna(0)
        w_tm1 = wh.iloc[i-1].fillna(0)
        turnover_i = (w_t - w_tm1).abs().sum() / 2.0
        if w_tm1.sum() > 0 or turnover_i > 0:
            total_turnover += turnover_i
            reb_count += 1
    if reb_count == 0:
        return 0.0, 0.0
    avg_rebal_turn = total_turnover / reb_count
    annual_rebalances = 52 if rebalance_freq == 'W' else 12
    annual_turnover = avg_rebal_turn * annual_rebalances
    monthly_turnover = annual_turnover / 12
    return monthly_turnover * 100, annual_turnover * 100

# -------------------------
# 스트림릿 UI
# -------------------------
def main():
    st.title("📈 U.S. Contrarian Strategy")
    st.markdown("동적 리밸런싱을 기반으로 한 컨트래리언 포트폴리오 분석 및 시각화")

    # Sidebar: Ticker 입력, 기간, 옵션
    with st.sidebar:
        st.header("⚙️ 설정")

        st.subheader("종목 티커 (콤마로 구분, 예: AAPL,MSFT,TSLA)")
        # 요청: M7 종목들을 디폴트로 설정
        tickers_default = ", ".join(M7_TICKERS)
        tickers_input = st.text_area("티커 목록", value=tickers_default, placeholder="예: AAPL, MSFT, TSLA", height=120)
        tickers = [t.strip().upper() for t in tickers_input.replace(';', ',').split(',') if t.strip() != ""]

        st.subheader("📅 기간 설정")
        default_start = datetime(2017, 1, 1)
        default_end = datetime.now()
        start_date = st.date_input("시작일", value=default_start.date(), min_value=datetime(1990,1,1).date(), max_value=default_end.date())
        end_date = st.date_input("종료일", value=default_end.date(), min_value=start_date, max_value=default_end.date())

        st.subheader("📈 벤치마크")
        benchmark_option = st.selectbox("벤치마크 선택", options=["Equal Weight (tickers)", f"{BENCHMARK_TICKER} (Nasdaq 100)"], index=0)

        st.subheader("🔧 전략 파라미터 (선택)")
        lookback_months = st.number_input("Lookback (months)", min_value=1, max_value=24, value=DEFAULT_PARAMS['lookback_months'])
        rebalance_freq = st.selectbox("Rebalance Frequency", options=['M','W'], format_func=lambda x: "Monthly" if x=='M' else "Weekly", index=0)
        threshold = st.slider("Threshold (negative drawdown, %)", min_value=-100.0, max_value=0.0, value=float(DEFAULT_PARAMS['threshold']*100), step=1.0) / 100.0
        weight_split = st.slider(
            "Weight split to deep drawdown group (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(DEFAULT_PARAMS['weight_split']*100),
            step=1.0
        ) / 100.0
        min_weight_change = st.number_input("Min weight change to trigger rebalance (absolute sum)", min_value=0.0, max_value=1.0, value=DEFAULT_PARAMS['min_weight_change'])

        st.markdown("---")
        st.info("자동 실행이 비활성화되어 있습니다. 반드시 '포트폴리오 분석 실행' 버튼을 눌러주세요.")
        run_button = st.button("🚀 포트폴리오 분석 실행", type="primary", use_container_width=True)

    if not run_button:
        st.info("사이드바에서 티커 및 기간을 설정한 뒤 '포트폴리오 분석 실행'을 눌러 결과를 보세요.")
        return

    # 입력 검증
    if len(tickers) == 0:
        st.error("티커 목록이 비어 있습니다. 하나 이상의 티커를 입력하세요.")
        return

    # 1) 먼저 각 티커의 '전체 사용 가능한 첫 거래일'을 확인 (get_first_available_date 사용)
    with st.spinner("티커별 전체 사용가능한 첫 거래일을 조회 중 (yfinance)..."):
        first_dates = {}
        for t in tickers:
            fd = get_first_available_date(t)  # returns datetime.date or None
            first_dates[t] = fd

    # 2) 시작일 기준 상장 여부 검사: get_first_available_date가 반환한 date와 비교 (tz 문제 회피)
    not_listed = []
    listed_ok = []
    for t, fd in first_dates.items():
        if fd is None:
            not_listed.append((t, "데이터 없음"))
        else:
            # fd is a datetime.date; start_date is also a datetime.date (from st.date_input)
            if start_date < fd:
                not_listed.append((t, fd.isoformat()))
            else:
                listed_ok.append((t, fd.isoformat()))

    if len(not_listed) > 0:
        st.error("선택한 시작일에 상장되어 있지 않은 종목이 있습니다. 시작일을 조정하거나 해당 종목을 제거하세요.")
        df_nl = pd.DataFrame(not_listed, columns=['Ticker', 'First Available Date'])
        st.dataframe(df_nl)
        if len(listed_ok) > 0:
            st.success("아래 종목들은 시작일 이전에도 거래 데이터가 존재합니다.")
            st.dataframe(pd.DataFrame(listed_ok, columns=['Ticker', 'First Available Date']))
        return

    # 다운로드 (사용자가 입력한 기간만 다운로드)
    with st.spinner("선택 기간 데이터 다운로드 중..."):
        prices = download_data(tickers, pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1))
        if benchmark_option.startswith(BENCHMARK_TICKER):
            benchmark_prices = download_data([BENCHMARK_TICKER], pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1))
        else:
            benchmark_prices = prices.copy()

    if prices is None or prices.empty:
        st.error("종목 데이터 다운로드 실패 또는 기간 내 데이터가 없습니다. 날짜 범위를 조정하거나 티커를 확인하세요.")
        return

    # 이하 기존 로직 (백테스트 등)
    lookback_days = max(5, int(lookback_months * 21))

    with st.spinner("백테스팅 중..."):
        portfolio_values, weight_history = backtest_strategy(
            prices,
            lookback_days,
            rebalance_freq,
            threshold,
            weight_split,
            min_weight_change
        )

    if portfolio_values is None or portfolio_values.empty:
        st.error("백테스트 결과가 없습니다. 파라미터를 조정해보세요.")
        return

    # 벤치마크 시리즈 생성
    if benchmark_option.startswith(BENCHMARK_TICKER):
        if benchmark_prices is None or benchmark_prices.empty or BENCHMARK_TICKER not in benchmark_prices.columns:
            st.error(f"벤치마크 {BENCHMARK_TICKER} 데이터를 가져올 수 없습니다.")
            return
        bench_vals = benchmark_prices[BENCHMARK_TICKER] / benchmark_prices[BENCHMARK_TICKER].iloc[0] * 100.0
    else:
        returns = prices.pct_change().fillna(0)
        bench_returns = returns.mean(axis=1)
        bench_vals = (1 + bench_returns).cumprod() * 100.0

    # 성과 계산
    strategy_metrics = calculate_performance_metrics(portfolio_values, bench_vals)
    benchmark_metrics = calculate_performance_metrics(bench_vals, portfolio_values)
    monthly_turnover, annual_turnover = calculate_turnover(weight_history, rebalance_freq)

    # returns & cum
    strat_returns = portfolio_values.pct_change().fillna(0)
    bench_returns = bench_vals.pct_change().fillna(0)
    strat_cum = (1 + strat_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod()

    def drawdown_ts(cum_series: pd.Series) -> pd.Series:
        running_max = cum_series.expanding().max()
        dd = (cum_series - running_max) / running_max
        return dd

    strat_dd = drawdown_ts(strat_cum)
    bench_dd = drawdown_ts(bench_cum)

    # ========== 출력 (기존 UI를 유지) ==========
    st.subheader("성과 개요 및 차트")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strat_cum.index, y=(strat_cum - 1) * 100, name="Strategy Cumulative (%)", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=bench_cum.index, y=(bench_cum - 1) * 100, name="Benchmark Cumulative (%)", line=dict(width=2, dash='dash')))
        fig.update_layout(title="누적수익률 (%)", xaxis_title="Date", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(x=np.log(strat_cum).index, y=np.log(strat_cum).values, name="Strategy Log Cumulative", line=dict(width=2)))
        fig_log.add_trace(go.Scatter(x=np.log(bench_cum).index, y=np.log(bench_cum).values, name="Benchmark Log Cumulative", line=dict(width=2, dash='dash')))
        fig_log.update_layout(title="로그 누적수익률 (log cumulative)", xaxis_title="Date", yaxis_title="Log(Value)")
        st.plotly_chart(fig_log, use_container_width=True)

    with col2:
        st.markdown("### 주요 지표")
        if strategy_metrics is not None:
            strat_df = pd.DataFrame.from_dict(strategy_metrics, orient='index', columns=['Strategy'])
        else:
            strat_df = pd.DataFrame()
        if benchmark_metrics is not None:
            bench_df = pd.DataFrame.from_dict(benchmark_metrics, orient='index', columns=['Benchmark'])
        else:
            bench_df = pd.DataFrame()
        ordered_index = ['Total Return (%)', 'CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Tracking Error (%)', 'Calmar Ratio']
        metrics_df = pd.DataFrame(index=ordered_index)
        if not strat_df.empty:
            metrics_df = metrics_df.join(strat_df)
        if not bench_df.empty:
            metrics_df = metrics_df.join(bench_df)
        metrics_df = metrics_df.round(3).fillna("-")
        st.dataframe(metrics_df, use_container_width=True)
        st.markdown("### 회전율")
        st.metric(label="평균 월간 회전율 (%)", value=f"{monthly_turnover:.2f}")
        st.metric(label="예상 연간 회전율 (%)", value=f"{annual_turnover:.2f}")

    st.subheader("Maximum Drawdown (전략 vs 벤치마크)")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values * 100, name="Strategy DD (%)", line=dict(color='crimson')))
    fig_dd.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values * 100, name="Benchmark DD (%)", line=dict(color='royalblue', dash='dash')))
    fig_dd.update_layout(title="Drawdown (%) over time", xaxis_title="Date", yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    # Portfolio Update 출력 (간단)
    st.subheader("포트폴리오 업데이트 (최근 리밸런싱 기준)")
    if weight_history is None or len(weight_history) == 0:
        st.info("리밸런싱 이력이 없습니다.")
    else:
        wh = weight_history.copy()
        if 'date' in wh.columns:
            wh = wh.set_index('date')
        wh = wh.sort_index()
        last_date = wh.index[-1]
        st.markdown(f"**최신 리밸런싱 날짜:** {last_date.date().isoformat()}")
        current_weights = wh.iloc[-1].fillna(0)
        if len(wh) >= 2:
            prev_weights = wh.iloc[-2].fillna(0)
        else:
            prev_weights = pd.Series(0.0, index=wh.columns)
        change_weights = (current_weights - prev_weights).fillna(0)
        display_df = pd.DataFrame({
            'Ticker': current_weights.index,
            'Name': [fetch_ticker_name(t) for t in current_weights.index],
            'Weight (%)': (current_weights.values * 100).round(2),
            'Change (%)': (change_weights.values * 100).round(2)
        }).sort_values('Weight (%)', ascending=False)
        st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.caption("변경사항: 시작일 상장 여부 판정 로직을 안전하게 수정(전체 히스토리 조회 및 date 비교)했고 M7 종목을 기본 티커로 설정했습니다.")

if __name__ == "__main__":
    main()
