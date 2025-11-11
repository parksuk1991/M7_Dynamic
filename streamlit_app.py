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
PASTEL_PALETTE = px.colors.qualitative.Pastel

# -------------------------
# 유틸리티 함수
# -------------------------
@st.cache_data(ttl=3600)
def download_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    try:
        if isinstance(tickers, str):
            tickers = [tickers]
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        data = data.reindex(columns=tickers)
        return data.ffill().bfill()
    except Exception as e:
        st.error(f"데이터 다운로드 실패: {str(e)}")
        return None


@st.cache_data(ttl=86400)
def fetch_ticker_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        name = info.get('shortName') or info.get('longName') or ticker
        return name
    except Exception:
        return ticker


@st.cache_data(ttl=86400)
def get_first_available_date(ticker: str) -> Optional[date]:
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
        w_tm1 = wh.iloc[i - 1].fillna(0)
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


def weights_history_to_composition_dict(weight_history: pd.DataFrame, rebalance_freq: str = 'M') -> Dict[date, Dict[str, float]]:
    comp: Dict[date, Dict[str, float]] = {}
    if weight_history is None or len(weight_history) == 0:
        return comp

    wh = weight_history.copy()
    if 'date' in wh.columns:
        wh['date'] = pd.to_datetime(wh['date'])
        wh = wh.set_index('date')
    wh = wh.sort_index()

    for idx, row in wh.iterrows():
        ts = pd.to_datetime(idx)
        if rebalance_freq == 'M':
            key = ts.to_period('M').to_timestamp('M').date()
        else:
            key = ts.date()
        weights: Dict[str, float] = {}
        for col in wh.columns:
            try:
                val = float(row[col])
            except Exception:
                continue
            weights[col] = val
        comp[key] = weights

    if rebalance_freq == 'M':
        today_date = date.today()
        first_of_this_month = today_date.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        comp = {k: v for k, v in comp.items() if k <= last_month_end}

    return comp


def get_rebalancing_changes(current: Dict[str, float], previous: Dict[str, float]) -> Dict[str, Dict]:
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


def ensure_returns_from_series(s: pd.Series) -> pd.Series:
    """If s looks like price/level series, convert to simple returns. Otherwise, assume it's returns."""
    if s is None:
        return pd.Series(dtype=float)
    s = s.dropna()
    if s.empty:
        return s
    # Heuristic: if typical magnitude > 1 (e.g., values >> 1), treat as levels
    if s.abs().mean() > 5:
        return s.pct_change().fillna(0)
    # If max > 10 and min > 0, also likely levels/cumulative levels
    if s.max() > 10 and s.min() > 0:
        return s.pct_change().fillna(0)
    return s


def monthly_returns(series: pd.Series) -> pd.Series:
    """
    Given a daily returns series (decimal, e.g., 0.01 for 1%), compute month-end returns:
    (1 + r_daily).resample('M').prod() - 1
    Input may be levels (prices) or returns; function will detect and convert if needed.
    Returns a Series indexed by month-end Timestamp with decimal returns.
    """
    if series is None:
        return pd.Series(dtype=float)
    # normalize index to DatetimeIndex
    s = pd.to_numeric(series, errors='coerce')
    s.index = pd.to_datetime(series.index)
    # ensure returns
    r = ensure_returns_from_series(series)
    # if ensure_returns_from_series returned levels converted, index preserved
    r.index = pd.to_datetime(r.index)
    # compute month-end returns
    monthly = (1 + r).resample('M').apply(lambda x: x.add(1).prod() - 1 if len(x) > 0 else np.nan)
    monthly.name = series.name if series.name else 'ret'
    return monthly


def monthly_excess_table(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    """
    Compute monthly returns for strategy and benchmark, then return a pivot table (rows: year, cols: month names)
    of the excess = (strategy_monthly - benchmark_monthly) in percent with 2 decimals.
    """
    strat_m = monthly_returns(strategy_returns)
    bench_m = monthly_returns(benchmark_returns)
    # align indexes
    idx = strat_m.index.union(bench_m.index).sort_values()
    strat_m = strat_m.reindex(idx)
    bench_m = bench_m.reindex(idx)
    excess = strat_m - bench_m
    if excess.empty:
        return pd.DataFrame()
    df = excess.to_frame(name='excess')
    df['year'] = df.index.year
    df['month'] = df.index.month
    pivot = df.pivot_table(index='year', columns='month', values='excess', aggfunc='first')
    pivot = pivot.reindex(columns=range(1, 13)).fillna(np.nan)
    pivot_pct = (pivot * 100).round(2)
    pivot_pct.columns = [datetime(1900, m, 1).strftime('%b') for m in pivot_pct.columns]
    return pivot_pct


# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("📈 U.S. Contrarian Strategy (Monthly Excess Table)")
    st.markdown("포트폴리오와 벤치마크의 월별 초과성과(포트폴리오 - 벤치마크) 표를 제공합니다. 월간 리밸런싱은 직전 월말까지 반영됩니다.")

    # Sidebar
    with st.sidebar:
        st.header("설정")
        st.subheader("종목 티커 (콤마로 구분)")
        tickers_default = ", ".join(M7_TICKERS)
        tickers_input = st.text_area("티커 목록", value=tickers_default, height=120)
        tickers = [t.strip().upper() for t in tickers_input.replace(';', ',').split(',') if t.strip() != ""]

        st.subheader("기간")
        default_start = datetime(2017, 1, 1)
        default_end = datetime.now()
        start_date = st.date_input("시작일", value=default_start.date(), min_value=datetime(1990, 1, 1).date(), max_value=default_end.date())
        end_date = st.date_input("종료일", value=default_end.date(), min_value=start_date, max_value=default_end.date())

        st.subheader("벤치마크")
        benchmark_option = st.selectbox("벤치마크 선택", options=["Equal Weight (tickers)", f"{BENCHMARK_TICKER} (Nasdaq 100)"], index=0)

        st.markdown("---")
        st.subheader("고정 파라미터 (최적)")
        st.write(f"Lookback: {OPTIMAL_PARAMS['lookback_months']} months, Rebalance: {'Monthly' if OPTIMAL_PARAMS['rebalance_freq']=='M' else 'Weekly'}")
        run_button = st.button("포트폴리오 분석 실행")

    if not run_button:
        st.info("사이드바에서 설정을 마친 후 '포트폴리오 분석 실행'을 눌러주세요.")
        return

    if len(tickers) == 0:
        st.error("티커를 하나 이상 입력하세요.")
        return

    # Check first available dates
    with st.spinner("티커 첫 사용 가능일 조회 중..."):
        first_dates = {t: get_first_available_date(t) for t in tickers}

    not_listed = []
    for t, fd in first_dates.items():
        if fd is None:
            not_listed.append((t, "데이터 없음"))
        else:
            if start_date < fd:
                not_listed.append((t, fd.isoformat()))
    if len(not_listed) > 0:
        st.error("다음 종목이 선택 시작일에 데이터가 없습니다:")
        st.dataframe(pd.DataFrame(not_listed, columns=['Ticker', 'First Available Date']))
        return

    # Download price data
    with st.spinner("데이터 다운로드 중..."):
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        prices = download_data(tickers, start_dt, end_dt)
        if benchmark_option.startswith(BENCHMARK_TICKER):
            benchmark_prices = download_data([BENCHMARK_TICKER], start_dt, end_dt)
            benchmark_name = BENCHMARK_TICKER
        else:
            benchmark_prices = None
            benchmark_name = "Equal Weight (tickers)"

    if prices is None or prices.empty:
        st.error("종목 가격 데이터를 가져올 수 없습니다. 날짜 범위 또는 티커를 확인하세요.")
        return

    # Backtest to produce portfolio values and weight history
    with st.spinner("백테스트 실행 중..."):
        portfolio_values, weight_history = backtest_strategy(prices, OPTIMAL_PARAMS['lookback_days'],
                                                            OPTIMAL_PARAMS['rebalance_freq'],
                                                            OPTIMAL_PARAMS['threshold'],
                                                            OPTIMAL_PARAMS['weight_split'],
                                                            OPTIMAL_PARAMS['min_weight_change'])

    if portfolio_values is None or portfolio_values.empty:
        st.error("백테스트 결과가 없습니다.")
        return

    # Create benchmark series (level series starting at 100)
    if benchmark_option.startswith(BENCHMARK_TICKER):
        if benchmark_prices is None or BENCHMARK_TICKER not in benchmark_prices.columns:
            st.error("벤치마크 데이터를 가져올 수 없습니다.")
            return
        bench_vals = benchmark_prices[BENCHMARK_TICKER] / benchmark_prices[BENCHMARK_TICKER].iloc[0] * 100.0
    else:
        # Equal weight: average of ticker returns (daily)
        returns = prices.pct_change().fillna(0)
        bench_returns_eq = returns.mean(axis=1)
        bench_vals = (1 + bench_returns_eq).cumprod() * 100.0
        # Name the series for clarity
        bench_vals.name = 'Equal Weight (tickers)'

    # Ensure portfolio_values and bench_vals have DatetimeIndex and are aligned daily series of levels
    portfolio_values.index = pd.to_datetime(portfolio_values.index)
    bench_vals.index = pd.to_datetime(bench_vals.index)

    # Compute daily returns series (decimals)
    strat_daily_returns = portfolio_values.pct_change().fillna(0)
    bench_daily_returns = bench_vals.pct_change().fillna(0)

    # Compute monthly excess table
    excess_table = monthly_excess_table(strat_daily_returns, bench_daily_returns)

    # Display results
    st.subheader("포트폴리오 vs 벤치마크: 월별 초과성과 표 (행: 연도, 열: 월, 단위: %)")
    if excess_table.empty:
        st.info("월별 초과성과를 계산할 충분한 데이터가 없습니다.")
    else:
        # Show a helpful note about month mapping: we use month-end returns and, for monthly rebalancing, only data up to previous month-end is considered.
        st.caption("월별 수익률은 일간 수익률을 월단위로 곱하여 계산했습니다. 표의 값은 '포트폴리오 월수익률 - 벤치마크 월수익률' (단위 %) 입니다.")
        # Replace NaN with '-' for display
        display_df = excess_table.fillna(np.nan).astype(object)
        display_df = display_df.where(pd.notnull(display_df), '-')
        st.dataframe(display_df, use_container_width=True)

    # Also show major metrics for reference (strategy vs benchmark)
    strategy_metrics = calculate_performance_metrics(portfolio_values, bench_vals)
    benchmark_metrics = calculate_performance_metrics(bench_vals, portfolio_values)
    st.subheader("요약 지표 (전략 vs 벤치마크)")
    ordered_index = ['Total Return (%)', 'CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Tracking Error (%)', 'Calmar Ratio']
    metrics_df = pd.DataFrame(index=ordered_index)
    if strategy_metrics is not None:
        metrics_df = metrics_df.join(pd.DataFrame.from_dict(strategy_metrics, orient='index', columns=['Strategy']))
    if benchmark_metrics is not None:
        metrics_df = metrics_df.join(pd.DataFrame.from_dict(benchmark_metrics, orient='index', columns=[benchmark_name]))
    metrics_df = metrics_df.round(3).fillna("-")
    st.dataframe(metrics_df, use_container_width=True)

    # Drawdown chart: area + boundary, legend outside right with only two entries
    def drawdown_ts(cum_series: pd.Series) -> pd.Series:
        running_max = cum_series.expanding().max()
        dd = (cum_series - running_max) / running_max
        return dd

    strat_cum = (1 + strat_daily_returns).cumprod()
    bench_cum = (1 + bench_daily_returns).cumprod()
    strat_dd = drawdown_ts(strat_cum)
    bench_dd = drawdown_ts(bench_cum)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values * 100, fill='tozeroy', mode='none',
                                name='Strategy DD (area)', showlegend=False, fillcolor='rgba(255,20,147,0.12)'))
    fig_dd.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values * 100, mode='lines', name='Strategy DD',
                                line=dict(color=PRIMARY_COLOR, width=1)))
    fig_dd.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values * 100, fill='tozeroy', mode='none',
                                name='Benchmark DD (area)', showlegend=False, fillcolor='rgba(65,105,225,0.12)'))
    fig_dd.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values * 100, mode='lines', name='Benchmark DD',
                                line=dict(color=SECONDARY_COLOR, width=1, dash='dash')))
    fig_dd.update_layout(title='Drawdown (%)', xaxis_title='Date', yaxis_title='Drawdown (%)',
                         legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top'),
                         template='plotly_white', hovermode='x unified')
    st.plotly_chart(fig_dd, use_container_width=True)

    # Portfolio composition (most recent month-end)
    weights_composition = weights_history_to_composition_dict(weight_history, rebalance_freq=OPTIMAL_PARAMS['rebalance_freq'])
    st.subheader("포트폴리오 업데이트 (최근 월말 기준)")
    if weights_composition:
        recent_dates = sorted(weights_composition.keys())
        latest_date = recent_dates[-1]
        prev_date = recent_dates[-2] if len(recent_dates) > 1 else None
        st.write(f"최신 리밸런싱(월말 기준): {latest_date.isoformat()}")
        current_weights = weights_composition[latest_date]
        if prev_date:
            previous_weights = weights_composition[prev_date]
        else:
            previous_weights = None

        col1, col2 = st.columns(2)
        with col1:
            cur_df = pd.DataFrame([{'Ticker': k, 'Weight (%)': f"{v:.2%}"} for k, v in sorted(current_weights.items(), key=lambda x: x[1], reverse=True)])
            st.dataframe(cur_df, use_container_width=True, hide_index=True)
            pie = px.pie(names=list(current_weights.keys()), values=list(current_weights.values()), color_discrete_sequence=PASTEL_PALETTE)
            pie.update_traces(textinfo='percent+label')
            st.plotly_chart(pie, use_container_width=True)
        with col2:
            if previous_weights:
                st.write(f"전월(월말) 대비 변화: {prev_date.isoformat()} → {latest_date.isoformat()}")
                changes = get_rebalancing_changes(current_weights, previous_weights)
                rows = []
                for tck, info in sorted(changes.items(), key=lambda x: abs(x[1]['change']), reverse=True):
                    rows.append({
                        'Ticker': tck,
                        'Previous (%)': f"{info['previous']:.2%}",
                        'Current (%)': f"{info['current']:.2%}",
                        'Change (%p)': f"{info['change']:+.2%}"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("비교할 이전 월말 데이터가 없습니다.")
    else:
        st.info("가중치 이력이 없습니다.")

    # Download buttons
    st.subheader("내보내기")
    csv_port = portfolio_values.rename("portfolio").to_frame().to_csv().encode('utf-8')
    st.download_button("포트폴리오 가치 CSV", data=csv_port, file_name="portfolio_values.csv", mime="text/csv")
    if weight_history is not None and len(weight_history) > 0:
        wh_dl = weight_history.copy()
        wh_dl['date'] = wh_dl['date'].astype(str) if 'date' in wh_dl.columns else wh_dl.index.astype(str)
        st.download_button("가중치 히스토리 CSV", data=wh_dl.to_csv(index=False).encode('utf-8'), file_name="weight_history.csv", mime="text/csv")

    st.caption("설명: 월별 수익률은 '일간 수익률을 월별로 곱한' 표준 방식으로 계산합니다. 표는 포트폴리오 월수익률 - 벤치마크 월수익률(%)로 구성되어 있습니다.")


if __name__ == "__main__":
    main()
