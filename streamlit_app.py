import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import warnings
from typing import List, Tuple, Optional, Dict

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="U.S. Contrarian Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 고정(하드코딩) 전략 파라미터
# =========================
OPTIMAL_PARAMS = {
    'lookback_months': 3,
    'lookback_days': 63,     # 3개월 * 약 21영업일
    'rebalance_freq': 'M',   # 'M' 또는 'W'
    'threshold': -0.3,       # drawdown threshold
    'weight_split': 0.60,    # deep drawdown 그룹에 부여할 비중
    'min_weight_change': 0.0
}

# 기본/디폴트 티커 (M7)
M7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
BENCHMARK_TICKER = 'QQQ'

# Color theme
PRIMARY_COLOR = 'deeppink'   # accent
SECONDARY_COLOR = 'royalblue'  # secondary
# pastel palette for pies
PASTEL_PALETTE = px.colors.qualitative.Pastel

# -------------------------
# 캐시 / 유틸리티 함수
# -------------------------
@st.cache_data(ttl=3600)
def download_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """주가 데이터 다운로드 (종가). 반환된 DataFrame의 컬럼은 입력 tickers 순서로 정렬됩니다."""
    try:
        if isinstance(tickers, str):
            tickers = [tickers]
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        # convert Series -> DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        # Ensure columns are in requested order (may introduce NaNs)
        data = data.reindex(columns=tickers)
        # forward/backfill to handle missing days
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
    티커의 전체 이용 가능한 데이터에서 첫 거래 가능일(종가가 존재하는 첫 날짜)을 datetime.date로 반환.
    """
    try:
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        series = hist['Close'] if 'Close' in hist.columns else hist.iloc[:, 0]
        first = series.first_valid_index()
        if first is None:
            return None
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

    # align rebalance candidates to available trading dates
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
            cur_dd = drawdowns.loc[date] if isinstance(drawdowns, pd.DataFrame) else drawdowns
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
                # no rebalance - record current weights
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
    요청된 지표 순서:
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
    """회전율 계산 (월간 및 연간 %)"""
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
# Helper functions for UI
# -------------------------
def weights_history_to_composition_dict(weight_history: pd.DataFrame, rebalance_freq: str = 'M') -> Dict[date, Dict[str, float]]:
    """
    weight_history(DataFrame with 'date' column or date index) -> {date: {ticker: weight}}
    For monthly rebalance ('M'), map entries to their month-end date (period end).
    If multiple entries fall in same month, keep the last one (chronological).
    """
    comp = {}
    if weight_history is None or len(weight_history) == 0:
        return comp
    wh = weight_history.copy()
    # normalize date column/index
    if 'date' in wh.columns:
        wh['date'] = pd.to_datetime(wh['date'])
        wh = wh.set_index('date')
    wh = wh.sort_index()

    for idx, row in wh.iterrows():
        ts = pd.to_datetime(idx)
        if rebalance_freq == 'M':
            key = ts.to_period('M').to_timestamp('M').date()  # month-end date
        else:
            key = ts.date()
        # extract numeric columns only (tickers)
        weights = {}
        for col in wh.columns:
            try:
                val = float(row[col])
            except Exception:
                continue
            weights[col] = val
        # if same month already present, replace (we iterate in chronological order)
        comp[key] = weights
    return comp

def get_rebalancing_changes(current: Dict[str,float], previous: Dict[str,float]) -> Dict[str, Dict]:
    """두 가중치 dict 비교해서 변화 리턴 (previous/current/change/action)"""
    all_keys = sorted(set(current.keys()) | set(previous.keys()))
    changes = {}
    for k in all_keys:
        prev = previous.get(k, 0.0)
        cur = current.get(k, 0.0)
        change = cur - prev
        if abs(change) < 1e-8:
            action = 'NO_CHANGE'
        elif change > 0:
            action = 'INCREASE'
        else:
            action = 'DECREASE'
        changes[k] = {'previous': prev, 'current': cur, 'change': change, 'action': action}
    return changes

def create_performance_charts(portfolio_returns: pd.Series, benchmark_returns: pd.Series, benchmark_name: str):
    """연도별/월별 비교 차트(Plotly) 생성"""
    # yearly returns (%)
    strat_yearly = (1 + portfolio_returns).resample('Y').apply(lambda s: (1 + s).prod() - 1)
    bench_yearly = (1 + benchmark_returns).resample('Y').apply(lambda s: (1 + s).prod() - 1)
    years = strat_yearly.index.union(bench_yearly.index).sort_values()
    df_year = pd.DataFrame({
        'Strategy': strat_yearly.reindex(years).fillna(0).values * 100,
        'Benchmark': bench_yearly.reindex(years).fillna(0).values * 100
    }, index=[d.year for d in years])
    fig_year = go.Figure()
    fig_year.add_trace(go.Bar(x=df_year.index.astype(str), y=df_year['Strategy'], name='Strategy', marker_color=PRIMARY_COLOR))
    fig_year.add_trace(go.Bar(x=df_year.index.astype(str), y=df_year['Benchmark'], name=benchmark_name, marker_color=SECONDARY_COLOR))
    fig_year.update_layout(barmode='group', title='연도별 성과 비교 (%)', xaxis_title='Year', yaxis_title='%', template="plotly_white")

    # monthly last 24 months
    strat_monthly = (1 + portfolio_returns).resample('M').prod() - 1
    bench_monthly = (1 + benchmark_returns).resample('M').prod() - 1
    combined = strat_monthly.index.union(bench_monthly.index).sort_values()
    last_24 = combined[-24:]
    df_m24 = pd.DataFrame({
        'Strategy': strat_monthly.reindex(last_24).fillna(0).values * 100,
        'Benchmark': bench_monthly.reindex(last_24).fillna(0).values * 100
    }, index=[d.strftime('%Y-%m') for d in last_24])
    fig_m24 = go.Figure()
    fig_m24.add_trace(go.Bar(x=df_m24.index, y=df_m24['Strategy'], name='Strategy', marker_color=PRIMARY_COLOR))
    fig_m24.add_trace(go.Bar(x=df_m24.index, y=df_m24['Benchmark'], name='Benchmark', marker_color=SECONDARY_COLOR))
    fig_m24.update_layout(barmode='group', title='최근 24개월 월별 성과 비교 (%)', xaxis_tickangle=-45, template="plotly_white")

    return fig_year, fig_m24

# -------------------------
# 스트림릿 UI (입력은 티커/기간/벤치/실행 버튼만)
# -------------------------
def main():
    st.title("📈 U.S. Contrarian Strategy")
    st.markdown("동적 리밸런싱(고정 파라미터)을 기반으로 한 컨트래리언 포트폴리오 분석 및 시각화")

    # 사이드바: 티커 입력, 기간, 벤치마크, 실행
    with st.sidebar:
        st.header("⚙️ 설정")
        st.subheader("종목 티커 (콤마로 구분)")
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

        st.markdown("---")
       
        # 파라미터 표시
        st.subheader("🎯 최적 파라미터")
        st.info(f"""
        **Lookback:** {OPTIMAL_PARAMS['lookback_months']}개월  
        **Rebalancing:** {"Weekly" if OPTIMAL_PARAMS['rebalance_freq']=='W' else "Monthly"}  
        **Threshold:** {abs(OPTIMAL_PARAMS['threshold'])*100:.0f}%  
        **Weight Split:** {OPTIMAL_PARAMS['weight_split']*100:.0f}%  
        **Min Weight Change:** {OPTIMAL_PARAMS['min_weight_change']*100:.0f}%
        """)
        run_button = st.button("🚀 포트폴리오 분석 실행", type="primary", use_container_width=True)
    
    if not run_button:
        st.info("사이드바에서 티커 및 기간을 설정한 뒤 '포트폴리오 분석 실행'을 눌러 결과를 보세요.")
        return

    # 기본 입력 확인
    if len(tickers) == 0:
        st.error("티커 목록이 비어 있습니다. 하나 이상의 티커를 입력하세요.")
        return

    # 시작일 상장 여부 검사: 전체 히스토리 기준 first available date 사용
    with st.spinner("티커별 전체 사용가능한 첫 거래일을 조회 중..."):
        first_dates = {t: get_first_available_date(t) for t in tickers}

    not_listed = []
    listed_ok = []
    for t, fd in first_dates.items():
        if fd is None:
            not_listed.append((t, "데이터 없음"))
        else:
            if start_date < fd:
                not_listed.append((t, fd.isoformat()))
            else:
                listed_ok.append((t, fd.isoformat()))

    if len(not_listed) > 0:
        st.error("선택한 시작일에 상장되어 있지 않은 종목이 있습니다. 시작일을 조정하거나 해당 종목을 제거하세요.")
        st.dataframe(pd.DataFrame(not_listed, columns=['Ticker', 'First Available Date']))
        if len(listed_ok) > 0:
            st.success("아래 종목들은 시작일 이전에도 거래 데이터가 존재합니다.")
            st.dataframe(pd.DataFrame(listed_ok, columns=['Ticker', 'First Available Date']))
        return

    # 데이터 다운로드
    with st.spinner("선택 기간 데이터 다운로드 중..."):
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        prices = download_data(tickers, start_dt, end_dt)
        if benchmark_option.startswith(BENCHMARK_TICKER):
            benchmark_prices = download_data([BENCHMARK_TICKER], start_dt, end_dt)
        else:
            benchmark_prices = prices.copy()

    if prices is None or prices.empty:
        st.error("종목 데이터 다운로드 실패 또는 기간 내 데이터가 없습니다. 날짜 범위를 조정하거나 티커를 확인하세요.")
        return

    # 백테스트 (고정 파라미터 사용)
    lookback_days = OPTIMAL_PARAMS['lookback_days']
    rebalance_freq = OPTIMAL_PARAMS['rebalance_freq']
    threshold = OPTIMAL_PARAMS['threshold']
    weight_split = OPTIMAL_PARAMS['weight_split']
    min_weight_change = OPTIMAL_PARAMS['min_weight_change']

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

    # 벤치마크 시리즈 생성 (초기 100 기준)
    if benchmark_option.startswith(BENCHMARK_TICKER):
        if benchmark_prices is None or benchmark_prices.empty or BENCHMARK_TICKER not in benchmark_prices.columns:
            st.error(f"벤치마크 {BENCHMARK_TICKER} 데이터를 가져올 수 없습니다.")
            return
        bench_vals = benchmark_prices[BENCHMARK_TICKER] / benchmark_prices[BENCHMARK_TICKER].iloc[0] * 100.0
    else:
        returns = prices.pct_change().fillna(0)
        bench_returns = returns.mean(axis=1)
        bench_vals = (1 + bench_returns).cumprod() * 100.0

    # 지표 계산
    strategy_metrics = calculate_performance_metrics(portfolio_values, bench_vals)
    benchmark_metrics = calculate_performance_metrics(bench_vals, portfolio_values)
    monthly_turnover, annual_turnover = calculate_turnover(weight_history, rebalance_freq)

    # 수익률 시리즈
    strat_returns = portfolio_values.pct_change().fillna(0)
    bench_returns = bench_vals.pct_change().fillna(0)
    strat_cum = (1 + strat_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod()

    # drawdown 시리즈
    def drawdown_ts(cum_series: pd.Series) -> pd.Series:
        running_max = cum_series.expanding().max()
        dd = (cum_series - running_max) / running_max
        return dd

    strat_dd = drawdown_ts(strat_cum)
    bench_dd = drawdown_ts(bench_cum)

    # -------------------------- UI 출력 --------------------------
    st.subheader("성과 개요 및 차트")

    # Put cumulative and log-cumulative side-by-side
    col_left, col_right = st.columns(2)
    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strat_cum.index, y=(strat_cum - 1) * 100, name="Strategy Cumulative (%)", line=dict(color=PRIMARY_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=bench_cum.index, y=(bench_cum - 1) * 100, name="Benchmark Cumulative (%)", line=dict(color=SECONDARY_COLOR, width=2, dash='dash')))
        fig.update_layout(title="누적수익률 (%)", xaxis_title="Date", yaxis_title="%", template="plotly_white", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(x=strat_cum.index, y=np.log(np.maximum(strat_cum.values, 1e-8)), name="Strategy Log Cumulative", line=dict(color=PRIMARY_COLOR, width=2)))
        fig_log.add_trace(go.Scatter(x=bench_cum.index, y=np.log(np.maximum(bench_cum.values, 1e-8)), name="Benchmark Log Cumulative", line=dict(color=SECONDARY_COLOR, width=2, dash='dash')))
        fig_log.update_layout(title="로그 누적수익률", template="plotly_white", hovermode='x unified')
        st.plotly_chart(fig_log, use_container_width=True)

    # Major metrics table under the charts (restored per request)
    st.subheader("주요 지표")
    ordered_index = ['Total Return (%)', 'CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Tracking Error (%)', 'Calmar Ratio']
    metrics_df = pd.DataFrame(index=ordered_index)
    if strategy_metrics is not None:
        metrics_df = metrics_df.join(pd.DataFrame.from_dict(strategy_metrics, orient='index', columns=['Strategy']))
    if benchmark_metrics is not None:
        metrics_df = metrics_df.join(pd.DataFrame.from_dict(benchmark_metrics, orient='index', columns=['Benchmark']))
    metrics_df = metrics_df.round(3).fillna("-")
    st.dataframe(metrics_df, use_container_width=True)

    # Drawdown area chart (filled)
    st.subheader("낙폭 (Drawdown) 비교 (영역형)")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=strat_dd.index,
        y=strat_dd.values * 100,
        fill='tozeroy',
        mode='none',
        name='Strategy DD (%)',
        fillcolor='rgba(255,20,147,0.25)'  # pale deeppink
    ))
    fig_dd.add_trace(go.Scatter(
        x=bench_dd.index,
        y=bench_dd.values * 100,
        fill='tozeroy',
        mode='none',
        name='Benchmark DD (%)',
        fillcolor='rgba(65,105,225,0.18)'  # pale royalblue
    ))
    fig_dd.update_layout(title="Drawdown (%) over time (area)", xaxis_title="Date", yaxis_title="Drawdown (%)", template="plotly_white", hovermode='x unified')
    st.plotly_chart(fig_dd, use_container_width=True)

    # ---------------- 리밸런싱 시점별 가중치 히스토리 (히트맵 + 테이블) ----------------
    st.subheader("리밸런싱 시점별 가중치 히스토리")
    if weight_history is None or len(weight_history) == 0:
        st.info("리밸런싱 가중치 이력이 없습니다.")
        weights_composition = {}
    else:
        wh = weight_history.copy()
        if 'date' in wh.columns:
            wh['date'] = pd.to_datetime(wh['date'])
            wh = wh.set_index('date')
        wh = wh.sort_index()
        # show whole table (percent)
        st.markdown("### 리밸런싱별 가중치 표")
        wh_pct = (wh * 100).round(3)
        st.dataframe(wh_pct, use_container_width=True)

        # heatmap - changed to pinkish sequential color scale (user requested)
        try:
            heat_df = wh.fillna(0).T
            heat_df.columns = [pd.to_datetime(c).strftime('%Y-%m-%d') if not isinstance(c, str) else c for c in heat_df.columns]
            # Use a pink/purple sequential scale
            fig_heat = px.imshow(heat_df, labels=dict(x="Rebalance Date", y="Ticker", color="Weight"),
                                 x=heat_df.columns, y=heat_df.index, color_continuous_scale='RdPu', aspect="auto")
            fig_heat.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception:
            st.warning("히트맵 생성 중 문제가 발생했습니다. 위 표를 확인하세요.")

        # build weights_composition mapping, mapping monthly rebalance entries to month-end (fix for reported Nov10 -> Oct31)
        weights_composition = weights_history_to_composition_dict(weight_history, rebalance_freq=rebalance_freq)

    # ---------------- 포트폴리오 업데이트 (최근 리밸런싱 기준) ----------------
    st.subheader(f"📰 포트폴리오 업데이트 ({date.today().strftime('%Y-%m')} 기준)")
    if weights_composition:
        recent_dates = sorted(weights_composition.keys())
        latest_date = recent_dates[-1]
        previous_date = recent_dates[-2] if len(recent_dates) > 1 else None
        current_weights = weights_composition[latest_date]
        previous_weights = weights_composition[previous_date] if previous_date else None

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**📕 {latest_date.strftime('%Y-%m-%d')} 리밸런싱 안**")
            current_df = pd.DataFrame([
                {'종목': stock, '비중': f"{weight:.2%}"}
                for stock, weight in sorted(current_weights.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(current_df, use_container_width=True, hide_index=True)

            # pastel pie
            fig_pie = px.pie(
                names=list(current_weights.keys()),
                values=list(current_weights.values()),
                title="📒 현재 비중 분포",
                color_discrete_sequence=PASTEL_PALETTE
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            if previous_weights:
                st.write(f"**📙 전월 대비 리밸런싱 변화** ({previous_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')})")
                changes = get_rebalancing_changes(current_weights, previous_weights)
                # sort by absolute change desc
                sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]['change']), reverse=True)
                rebalancing_data = []
                for stock, change_info in sorted_changes:
                    action_emoji = "📈" if change_info['action'] == 'INCREASE' else "📉" if change_info['action'] == 'DECREASE' else "➡️"
                    rebalancing_data.append({
                        '종목': f"{action_emoji} {stock}",
                        '이전 비중': f"{change_info['previous']:.2%}",
                        '현재 비중': f"{change_info['current']:.2%}",
                        '변화': f"{change_info['change']:+.2%}"
                    })
                rebalancing_df = pd.DataFrame(rebalancing_data)
                st.dataframe(rebalancing_df, use_container_width=True, hide_index=True)

                # bar chart with color mapping
                stocks = [r['종목'] for r in rebalancing_data]
                # extract numeric change values
                changes_values = [float(r['변화'].replace('%',''))/100.0 for r in rebalancing_data]
                colors = [PRIMARY_COLOR if v > 0 else SECONDARY_COLOR for v in changes_values]
                fig_rebal = go.Figure(data=[
                    go.Bar(x=stocks, y=[x*100 for x in changes_values],
                           marker_color=colors,
                           text=[f"{x:+.2%}" for x in changes_values],
                           textposition='auto')
                ])
                fig_rebal.update_layout(
                    title="📗 리밸런싱 변화 (%p)",
                    xaxis_title="종목",
                    yaxis_title="비중 변화 (%p)",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig_rebal, use_container_width=True)
            else:
                st.info("비교할 이전 포트폴리오 데이터가 없습니다.")
    else:
        st.info("리밸런싱 구성 데이터가 없습니다.")

    # ---------------- 월별 수익률 분포 및 12개월 롤링 샤프비율 ----------------
    st.subheader("월별 수익률 분포 및 12개월 롤링 샤프비율")
    strat_monthly = (1 + strat_returns).resample('M').prod() - 1
    bench_monthly = (1 + bench_returns).resample('M').prod() - 1

    colm1, colm2 = st.columns([1,1])
    with colm1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=strat_monthly.values * 100, name='포트폴리오', opacity=0.7, marker_color=PRIMARY_COLOR, nbinsx=20))
        fig_hist.add_trace(go.Histogram(x=bench_monthly.values * 100, name='벤치마크', opacity=0.7, marker_color=SECONDARY_COLOR, nbinsx=20))
        fig_hist.update_layout(title="월별 수익률 분포", xaxis_title="월별 수익률 (%)", yaxis_title="빈도", barmode='overlay', template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with colm2:
        def rolling_sharpe(monthly_ret: pd.Series, window: int = 12):
            if monthly_ret is None or len(monthly_ret) < window:
                return pd.Series(dtype=float)
            mu = monthly_ret.rolling(window).mean()
            sigma = monthly_ret.rolling(window).std()
            return (mu / sigma) * np.sqrt(12)

        strat_rs = rolling_sharpe(strat_monthly, 12)
        bench_rs = rolling_sharpe(bench_monthly, 12)
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=strat_rs.index, y=strat_rs.values, mode='lines', name='포트폴리오', line=dict(color=PRIMARY_COLOR, width=2)))
        fig_rs.add_trace(go.Scatter(x=bench_rs.index, y=bench_rs.values, mode='lines', name='벤치마크', line=dict(color=SECONDARY_COLOR, width=2, dash='dash')))
        fig_rs.update_layout(title='12개월 롤링 샤프비율', xaxis_title='Date', yaxis_title='Sharpe', template="plotly_white")
        st.plotly_chart(fig_rs, use_container_width=True)

    # ---------------- 연도별 & 최근 24개월 비교 ----------------
    st.subheader("연도별 및 최근 24개월 성과 비교")
    fig_yearly, fig_monthly = create_performance_charts(strat_returns, bench_returns, BENCHMARK_TICKER)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_yearly, use_container_width=True)
    with c2:
        st.plotly_chart(fig_monthly, use_container_width=True)

    # ---------------- 포트폴리오 구성 히스토리 (최근 6개월, 월별) ----------------
    st.subheader("포트폴리오 구성 히스토리 (최근 6개월)")
    if weights_composition:
        recent_dates = sorted(weights_composition.keys())[-6:]
        for date_key in recent_dates:
            weights = weights_composition[date_key]
            with st.expander(f"{date_key.strftime('%Y-%m-%d')} 포트폴리오 구성"):
                weights_df = pd.DataFrame([
                    {'종목': stock, '가중치': f"{weight:.2%}"}
                    for stock, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)
                ])
                colA, colB = st.columns([2,1])
                with colA:
                    st.dataframe(weights_df, use_container_width=True, hide_index=True)
                with colB:
                    fig_pie = px.pie(
                        names=list(weights.keys()),
                        values=list(weights.values()),
                        title="가중치 분포",
                        color_discrete_sequence=PASTEL_PALETTE
                    )
                    fig_pie.update_traces(textinfo='percent+label')
                    fig_pie.update_layout(height=300, template="plotly_white")
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("가중치 히스토리가 없습니다.")

    # ---------------- 추가 도구 및 다운로드 ----------------
    st.subheader("추가 도구 및 내보내기")
    c1, c2 = st.columns([1,1])
    with c1:
        csv_port = portfolio_values.rename("portfolio").to_frame().to_csv().encode('utf-8')
        st.download_button("포트폴리오 가치(시계열) CSV 다운로드", data=csv_port, file_name="portfolio_values.csv", mime="text/csv")
        if weight_history is not None and len(weight_history) > 0:
            wh_dl = weight_history.copy()
            wh_dl['date'] = wh_dl['date'].astype(str) if 'date' in wh_dl.columns else wh_dl.index.astype(str)
            st.download_button("가중치 히스토리 CSV 다운로드", data=wh_dl.to_csv(index=False).encode('utf-8'), file_name="weight_history.csv", mime="text/csv")
    with c2:
        st.markdown("### 데이터/파라미터 요약")
        st.write(f"Tickers: {', '.join(tickers)}")
        st.write(f"기간: {start_date} ~ {end_date}")
        st.write(f"Lookback (days): {lookback_days}")
        st.write(f"Rebalance Frequency: {'Monthly' if rebalance_freq=='M' else 'Weekly'}")
        st.write(f"Threshold: {threshold}")
        st.write(f"Weight Split: {weight_split}")
        st.write(f"Min Weight Change: {min_weight_change}")

    st.markdown("---")
    st.caption("변경사항: (1) 백테스팅 정보 블록 제거, 주요 지표 섹션 복구(누적/로그 아래에 배치). (2) 히트맵을 핑크 계열로 변경, 낙폭 차트를 영역형으로 표시. (3) 월간 리밸런싱의 최신 리밸런싱 날짜는 '월말(직전월말)' 기준으로 매핑하여 Nov 10 같은 비정상적 날짜 대신 10월 말 등으로 표시하도록 수정했습니다. 파이차트는 파스텔 톤을 사용합니다.")

if __name__ == "__main__":
    main()
