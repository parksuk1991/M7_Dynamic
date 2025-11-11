import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from typing import List, Tuple, Optional

warnings.filterwarnings('ignore')

# 페이지 설정 (이 호출은 가능한 파일 최상단에 위치해야 함)
st.set_page_config(
    page_title="U.S. Contrarian Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 기본 파라미터
DEFAULT_PARAMS = {
    'lookback_months': 3,
    'lookback_days': 63,  # 3개월 ~ 63영업일
    'rebalance_freq': 'M',  # 'M' or 'W'
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

    # 실제 사용 가능한 날짜로 맞춤
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
    # unique and sorted
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
                # no rebalance
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
    """
    주요 지표 계산. Returns a dict of metrics.
    Ordering requested:
    Total Return, CAGR, Volatility, Sharpe, Max Drawdown, Tracking Error, Calmar
    """
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
    # CAGR
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
        # align returns
        bvals = benchmark_series.reindex(values.index).dropna()
        if len(bvals) > 1:
            bret = bvals.pct_change().dropna()
            # align lengths
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
        tickers_input = st.text_area("티커 목록", value="", placeholder="예: AAPL, MSFT, TSLA (M7이 자동으로 들어가지 않음)", height=80)
        tickers = [t.strip().upper() for t in tickers_input.replace(';', ',').split(',') if t.strip() != ""]

        st.subheader("📅 기간 설정")
        default_start = datetime(2017, 1, 1)
        default_end = datetime.now()
        start_date = st.date_input("시작일", value=default_start, min_value=datetime(1990,1,1), max_value=default_end)
        end_date = st.date_input("종료일", value=default_end, min_value=start_date, max_value=default_end)

        st.subheader("📈 벤치마크")
        benchmark_option = st.selectbox("벤치마크 선택", options=["Equal Weight (tickers)", f"{BENCHMARK_TICKER} (Nasdaq 100)"], index=0)

        st.subheader("🔧 전략 파라미터 (선택)")
        lookback_months = st.number_input("Lookback (months)", min_value=1, max_value=24, value=DEFAULT_PARAMS['lookback_months'])
        rebalance_freq = st.selectbox("Rebalance Frequency", options=['M','W'], format_func=lambda x: "Monthly" if x=='M' else "Weekly", index=0)
        threshold = st.slider("Threshold (negative drawdown, %)", min_value=-100.0, max_value=0.0, value=DEFAULT_PARAMS['threshold']*100) / 100.0
        weight_split = st.slider("Weight split to deep drawdown group (%)", min_value=0.0, max_value=100.0, value=int(DEFAULT_PARAMS['weight_split']*100)) / 100.0
        min_weight_change = st.number_input("Min weight change to trigger rebalance (absolute sum)", min_value=0.0, max_value=1.0, value=DEFAULT_PARAMS['min_weight_change'])

        st.markdown("---")
        # 자동 실행 비활성화 (요청사항1)
        st.info("자동 실행이 비활성화되어 있습니다. 반드시 '포트폴리오 분석 실행' 버튼을 눌러주세요.")
        run_button = st.button("🚀 포트폴리오 분석 실행", type="primary", use_container_width=True)

    if not run_button:
        st.info("사이드바에서 티커 및 기간을 설정한 뒤 '포트폴리오 분석 실행'을 눌러 결과를 보세요.")
        return

    # 입력 검증: tickers non-empty
    if len(tickers) == 0:
        st.error("티커 목록이 비어 있습니다. 하나 이상의 티커를 입력하세요.")
        return

    # 다운로드
    with st.spinner("데이터 다운로드 중..."):
        prices = download_data(tickers, pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1))
        # benchmark
        if benchmark_option.startswith(BENCHMARK_TICKER):
            benchmark_prices = download_data([BENCHMARK_TICKER], pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1))
        else:
            benchmark_prices = prices.copy()

    if prices is None or prices.empty:
        st.error("종목 데이터 다운로드 실패 또는 기간 내 데이터가 없습니다. 날짜 범위를 조정하거나 티커를 확인하세요.")
        return

    # 시작일 상장 여부 검사 (요청사항3)
    not_listed = []
    for t in tickers:
        col = prices.get(t)
        if col is None:
            not_listed.append((t, "데이터 없음"))
            continue
        # find first valid date
        first_valid = col.first_valid_index()
        if first_valid is None:
            not_listed.append((t, "데이터 없음"))
        else:
            # if first valid date is after chosen start_date, then not listed at start_date
            if pd.Timestamp(start_date) < first_valid.normalize():
                not_listed.append((t, first_valid.date().isoformat()))

    if len(not_listed) > 0:
        # 팝업 형태로 오류표시: Streamlit에는 modal 없음. 에러와 상세 표로 안내.
        st.error("선택한 시작일에 상장되어 있지 않은 종목이 있습니다. 시작일을 조정하거나 해당 종목을 제거하세요.")
        df_nl = pd.DataFrame(not_listed, columns=['Ticker', 'First Available Date'])
        st.dataframe(df_nl)
        return

    # 전략 파라미터 보정
    lookback_days = max(5, int(lookback_months * 21))

    # 백테스트
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

    # 벤치마크 시리즈 생성 (동일한 날짜 인덱스)
    if benchmark_option.startswith(BENCHMARK_TICKER):
        if benchmark_prices is None or benchmark_prices.empty or BENCHMARK_TICKER not in benchmark_prices.columns:
            st.error(f"벤치마크 {BENCHMARK_TICKER} 데이터를 가져올 수 없습니다.")
            return
        bench_vals = benchmark_prices[BENCHMARK_TICKER] / benchmark_prices[BENCHMARK_TICKER].iloc[0] * 100.0
    else:
        # equal weight across user tickers
        returns = prices.pct_change().fillna(0)
        bench_returns = returns.mean(axis=1)
        bench_vals = (1 + bench_returns).cumprod() * 100.0

    # ------------- 성과 계산 -------------
    strategy_metrics = calculate_performance_metrics(portfolio_values, bench_vals)
    benchmark_metrics = calculate_performance_metrics(bench_vals, portfolio_values)
    monthly_turnover, annual_turnover = calculate_turnover(weight_history, rebalance_freq)

    # Prepare returns series
    strat_returns = portfolio_values.pct_change().fillna(0)
    bench_returns = bench_vals.pct_change().fillna(0)

    # Max drawdown time series for strategy & benchmark (요청사항4)
    # Compute cumulative returns series (1 + returns).cumprod()
    strat_cum = (1 + strat_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod()

    def drawdown_ts(cum_series: pd.Series) -> pd.Series:
        running_max = cum_series.expanding().max()
        dd = (cum_series - running_max) / running_max
        return dd

    strat_dd = drawdown_ts(strat_cum)
    bench_dd = drawdown_ts(bench_cum)

    # ---------------- UI: 상단 요약 & 차트 ----------------
    st.subheader("성과 개요 및 차트")

    # Left: cumulative returns & log cumulative
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strat_cum.index, y=(strat_cum - 1) * 100, name="Strategy Cumulative (%)", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=bench_cum.index, y=(bench_cum - 1) * 100, name="Benchmark Cumulative (%)", line=dict(width=2, dash='dash')))
        fig.update_layout(title="누적수익률 (%)", xaxis_title="Date", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

        # 로그 누적수익률
        fig_log = go.Figure()
        # to avoid log of zero or negative, use np.log(cum_series)
        fig_log.add_trace(go.Scatter(x=np.log(strat_cum).index, y=np.log(strat_cum).values, name="Strategy Log Cumulative", line=dict(width=2)))
        fig_log.add_trace(go.Scatter(x=np.log(bench_cum).index, y=np.log(bench_cum).values, name="Benchmark Log Cumulative", line=dict(width=2, dash='dash')))
        fig_log.update_layout(title="로그 누적수익률 (log cumulative)", xaxis_title="Date", yaxis_title="Log(Value)")
        st.plotly_chart(fig_log, use_container_width=True)

    # Right: 주요 지표 (요청사항6: specific order)
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

        # Ensure ordering: Total Return, CAGR, Volatility, Sharpe, Max Drawdown, Tracking Error, Calmar
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

    # ---------------- Max Drawdown Chart ---------------- (요청사항4)
    st.subheader("Maximum Drawdown (전략 vs 벤치마크)")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values * 100, name="Strategy DD (%)", line=dict(color='crimson')))
    fig_dd.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values * 100, name="Benchmark DD (%)", line=dict(color='royalblue', dash='dash')))
    fig_dd.update_layout(title="Drawdown (%) over time", xaxis_title="Date", yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    # ---------------- Portfolio Update (요청사항7) ----------------
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

        # 현재 가중치
        current_weights = wh.iloc[-1].fillna(0)
        # 이전 리밸런싱 가중치 (있다면)
        if len(wh) >= 2:
            prev_weights = wh.iloc[-2].fillna(0)
        else:
            prev_weights = pd.Series(0.0, index=wh.columns)

        change_weights = (current_weights - prev_weights).fillna(0)

        # Build display dataframe with ticker, name, weight, change
        display_df = pd.DataFrame({
            'Ticker': current_weights.index,
            'Name': [fetch_ticker_name(t) for t in current_weights.index],
            'Weight': current_weights.values,
            'Change vs Prev Rebal': change_weights.values
        })
        display_df['Weight (%)'] = (display_df['Weight'] * 100).round(2)
        display_df['Change (%)'] = (display_df['Change vs Prev Rebal'] * 100).round(2)
        display_df = display_df[['Ticker', 'Name', 'Weight (%)', 'Change (%)']].sort_values('Weight (%)', ascending=False)
        st.dataframe(display_df, use_container_width=True)

        # 시각화: 바차트 (weights) 및 변화(색상)
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(
            x=display_df['Ticker'],
            y=display_df['Weight (%)'],
            name='Current Weight (%)',
            marker_color='teal'
        ))
        fig_w.add_trace(go.Bar(
            x=display_df['Ticker'],
            y=display_df['Change (%)'],
            name='Change vs Prev (%)',
            marker_color=['crimson' if v < 0 else 'darkgreen' for v in display_df['Change (%)']]
        ))
        fig_w.update_layout(barmode='group', title='현재 가중치 및 직전 리밸런싱 대비 변화 (%)', xaxis_title='Ticker', yaxis_title='%')
        st.plotly_chart(fig_w, use_container_width=True)

    # ---------------- Weight change list + visualization per last rebalance (요청사항4) ----------------
    st.subheader("직전 리밸런싱 대비 종목별 변화 (리스트 및 차트)")
    if weight_history is None or len(weight_history) < 2:
        st.info("이전 리밸런싱 데이터가 부족하여 비교를 표시할 수 없습니다.")
    else:
        wh2 = weight_history.copy()
        if 'date' in wh2.columns:
            wh2 = wh2.set_index('date')
        wh2 = wh2.sort_index()
        last = wh2.iloc[-1].fillna(0)
        prev = wh2.iloc[-2].fillna(0)
        delta = (last - prev)
        delta_df = pd.DataFrame({
            'Ticker': last.index,
            'Prev Weight (%)': (prev * 100).round(3).values,
            'Last Weight (%)': (last * 100).round(3).values,
            'Delta (%)': (delta * 100).round(3).values
        }).sort_values('Delta (%)', ascending=False)
        st.dataframe(delta_df, use_container_width=True)

        # delta waterfall-style bar chart
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Bar(x=delta_df['Ticker'], y=delta_df['Delta (%)'], marker_color=['green' if v >= 0 else 'red' for v in delta_df['Delta (%)']]))
        fig_delta.update_layout(title="종목별 리밸런싱 변화 (Last - Prev) %", xaxis_title='Ticker', yaxis_title='Delta (%)')
        st.plotly_chart(fig_delta, use_container_width=True)

    # ---------------- Monthly return distribution & 12-month rolling Sharpe (요청사항8) ----------------
    st.subheader("월별 수익률 분포 및 12개월 롤링 샤프비율")

    # compute monthly returns
    strat_monthly = (1 + strat_returns).resample('M').prod() - 1
    bench_monthly = (1 + bench_returns).resample('M').prod() - 1

    col3, col4 = st.columns([1,1])
    with col3:
        # histogram / boxplot
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=strat_monthly.values * 100, name='Strategy', opacity=0.6))
        fig_hist.add_trace(go.Histogram(x=bench_monthly.values * 100, name='Benchmark', opacity=0.6))
        fig_hist.update_layout(barmode='overlay', title='월별 수익률 분포 (%)', xaxis_title='Monthly Return (%)')
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=strat_monthly.values * 100, name='Strategy'))
        fig_box.add_trace(go.Box(y=bench_monthly.values * 100, name='Benchmark'))
        fig_box.update_layout(title='월별 수익률 분포 (Box)', yaxis_title='%')
        st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        # 12-month rolling Sharpe computed on monthly returns (window=12)
        def rolling_sharpe(monthly_ret: pd.Series, window: int = 12):
            if monthly_ret is None or len(monthly_ret) < window:
                return pd.Series(dtype=float)
            mu = monthly_ret.rolling(window).mean()
            sigma = monthly_ret.rolling(window).std()
            return (mu / sigma) * np.sqrt(12)

        strat_rs = rolling_sharpe(strat_monthly, 12)
        bench_rs = rolling_sharpe(bench_monthly, 12)
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=strat_rs.index, y=strat_rs.values, name='Strategy 12M Rolling Sharpe', line=dict(color='crimson')))
        fig_rs.add_trace(go.Scatter(x=bench_rs.index, y=bench_rs.values, name='Benchmark 12M Rolling Sharpe', line=dict(color='royalblue', dash='dash')))
        fig_rs.update_layout(title='12개월 롤링 샤프비율 (월별 데이터 기준)', xaxis_title='Date', yaxis_title='Sharpe')
        st.plotly_chart(fig_rs, use_container_width=True)

    # ---------------- 연도별 성과 비교 및 월별(최근24개월) 비교 (요청사항9) ----------------
    st.subheader("연도별 및 최근 24개월 성과 비교")

    # yearly cumulative returns: group by year and compute total return for each year
    def yearly_perf(returns_series: pd.Series) -> pd.Series:
        monthly = (1 + returns_series).resample('M').prod() - 1
        # convert monthly to per-year aggregation
        yearly = (1 + returns_series).resample('Y').apply(lambda s: (1 + s).prod() - 1)
        return yearly

    strat_yearly = (1 + strat_returns).resample('Y').apply(lambda s: (1 + s).prod() - 1)
    bench_yearly = (1 + bench_returns).resample('Y').apply(lambda s: (1 + s).prod() - 1)

    # Yearly bar chart
    years = strat_yearly.index.union(bench_yearly.index).sort_values()
    df_year = pd.DataFrame({
        'Strategy': strat_yearly.reindex(years).fillna(0).values * 100,
        'Benchmark': bench_yearly.reindex(years).fillna(0).values * 100
    }, index=[d.year for d in years])
    fig_year = go.Figure()
    fig_year.add_trace(go.Bar(x=df_year.index.astype(str), y=df_year['Strategy'], name='Strategy'))
    fig_year.add_trace(go.Bar(x=df_year.index.astype(str), y=df_year['Benchmark'], name='Benchmark'))
    fig_year.update_layout(barmode='group', title='연도별 성과 비교 (%)', xaxis_title='Year', yaxis_title='%')
    st.plotly_chart(fig_year, use_container_width=True)

    # Monthly last 24 months
    strat_monthly_all = strat_monthly.copy()
    bench_monthly_all = bench_monthly.copy()
    combined_months = strat_monthly_all.index.union(bench_monthly_all.index).sort_values()
    last_24 = combined_months[-24:]
    df_m24 = pd.DataFrame({
        'Strategy': strat_monthly_all.reindex(last_24).fillna(0).values * 100,
        'Benchmark': bench_monthly_all.reindex(last_24).fillna(0).values * 100
    }, index=[d.strftime('%Y-%m') for d in last_24])
    fig_m24 = go.Figure()
    fig_m24.add_trace(go.Bar(x=df_m24.index, y=df_m24['Strategy'], name='Strategy'))
    fig_m24.add_trace(go.Bar(x=df_m24.index, y=df_m24['Benchmark'], name='Benchmark'))
    fig_m24.update_layout(title='최근 24개월 월별 성과 비교 (%)', xaxis_title='Month', yaxis_title='Monthly Return (%)', barmode='group', xaxis_tickangle=-45)
    st.plotly_chart(fig_m24, use_container_width=True)

    # ---------------- 포트폴리오 구성 히스토리 (최근 6개월, 월별) (요청사항10) ----------------
    st.subheader("포트폴리오 구성 히스토리 (최근 6개월, 월별)")
    if weight_history is None or len(weight_history) == 0:
        st.info("가중치 히스토리가 없습니다.")
    else:
        wh_hist = weight_history.copy()
        if 'date' in wh_hist.columns:
            wh_hist = wh_hist.set_index('date')
        wh_hist = wh_hist.sort_index()
        # pick last 6 monthly rebalancing points (unique months)
        # convert index to month period
        wh_hist['month'] = wh_hist.index.to_period('M')
        # get last 6 months where we have a rebalancing entry
        last_months = wh_hist['month'].unique()[-6:]
        if len(last_months) == 0:
            st.info("최근 6개월의 리밸런싱 기록이 부족합니다.")
        else:
            for m in last_months:
                month_df = wh_hist[wh_hist['month'] == m]
                # take the final rebalancing in that month
                row = month_df.iloc[-1].drop(labels=['month'], errors='ignore').fillna(0)
                st.markdown(f"#### {m.strftime('%Y-%m')}")
                table = pd.DataFrame({
                    'Ticker': row.index,
                    'Weight (%)': (row.values * 100).round(3)
                }).sort_values('Weight (%)', ascending=False)
                st.dataframe(table, use_container_width=True)
                # pie chart
                fig_p = px.pie(table, names='Ticker', values='Weight (%)', title=f"Composition {m.strftime('%Y-%m')}")
                st.plotly_chart(fig_p, use_container_width=True)

    # ---------------- 추가 유용 기능 (요청사항11) ----------------
    st.subheader("추가 도구 및 내보내기")
    cold1, cold2 = st.columns([1,1])
    with cold1:
        # download CSVs
        csv_port = portfolio_values.rename("portfolio").to_frame().to_csv().encode('utf-8')
        st.download_button("포트폴리오 가치(시계열) CSV 다운로드", data=csv_port, file_name="portfolio_values.csv", mime="text/csv")
        if weight_history is not None and len(weight_history) > 0:
            wh_dl = weight_history.copy()
            wh_dl['date'] = wh_dl['date'].astype(str) if 'date' in wh_dl.columns else wh_dl.index.astype(str)
            st.download_button("가중치 히스토리 CSV 다운로드", data=wh_dl.to_csv(index=False).encode('utf-8'), file_name="weight_history.csv", mime="text/csv")

    with cold2:
        st.markdown("### 데이터/파라미터 요약")
        st.write(f"Tickers: {', '.join(tickers)}")
        st.write(f"기간: {start_date} ~ {end_date}")
        st.write(f"Lookback (days): {lookback_days}")
        st.write(f"Rebalance Frequency: {'Monthly' if rebalance_freq=='M' else 'Weekly'}")
        st.write(f"Threshold: {threshold}")
        st.write(f"Weight Split: {weight_split}")
        st.write(f"Min Weight Change: {min_weight_change}")

    st.markdown("---")
    st.caption("앱 개선 요청이 있으면 알려주세요. 티커 이름 불러오기나 벤치마크 변경 등 추가 요청을 반영할 수 있습니다.")

if __name__ == "__main__":
    main()
