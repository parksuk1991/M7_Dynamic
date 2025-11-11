import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 페이지 설정 (이 호출은 가능한 파일 최상단에 위치해야 함)
st.set_page_config(
    page_title="M7 Dynamic Portfolio Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 최적화된 파라미터 (하드코딩)
OPTIMAL_PARAMS = {
    'lookback_months': 3,
    'lookback_days': 63,  # 3개월 * 21일
    'rebalance_freq': 'M',  # 'M' 또는 'W'
    'threshold': -0.3,
    'weight_split': 0.60,
    'min_weight_change': 0.0
}

M7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
BENCHMARK_TICKER = 'QQQ'

# 캐시된 데이터 다운로드
@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers, start_date, end_date):
    """주가 데이터 다운로드 (종가)"""
    try:
        # yfinance accepts list or single ticker; ensure list
        if isinstance(tickers, str):
            tickers = [tickers]
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        # ensure columns are tickers (order)
        data = data.reindex(columns=tickers)
        return data.ffill().bfill()
    except Exception as e:
        st.error(f"데이터 다운로드 실패: {str(e)}")
        return None

def calculate_drawdown_from_peak(prices, lookback_days):
    """고점 대비 하락률 계산 (각종목 별로 최근 lookback_days 동안 고점 대비)"""
    rolling_max = prices.rolling(window=lookback_days, min_periods=1).max()
    return (prices - rolling_max) / rolling_max

def calculate_weights_by_drawdown(drawdowns, threshold, weight_split):
    """
    하락률 기반 가중치 계산
    - drawdowns: Series(index=tickers)에서 각 종목의 최근 고점 대비 하락률 (음수)
    - threshold: 예: -0.3 (30% 하락 시 deep drawdown)
    - weight_split: deep drawdown 그룹에 할당할 총 비중 (예: 0.6)
    """
    # ensure we operate on a Series indexed by ticker symbols
    if drawdowns is None or len(drawdowns.dropna()) == 0:
        # equal weight fallback
        idx = drawdowns.index if drawdowns is not None else M7_TICKERS
        return pd.Series(1.0 / len(idx), index=idx)

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

    # normalize (safe)
    if weights.sum() <= 0:
        weights = pd.Series(1.0 / len(idx), index=idx)
    else:
        weights = weights / weights.sum()
    return weights

def backtest_strategy(prices, lookback_days, rebalance_freq, threshold, weight_split, min_weight_change=0.0):
    """백테스팅 실행
    - prices: DataFrame (index: date, columns: tickers) 종가
    - 반환: portfolio_values (Series indexed by dates), weight_history (DataFrame with date column + tickers)
    """
    if prices is None or len(prices) == 0:
        return pd.Series(dtype=float), pd.DataFrame()

    # rebalance date candidates
    if rebalance_freq == 'W':
        rebalance_dates = prices.resample('W-MON').last().index
    else:
        rebalance_dates = prices.resample('M').last().index

    # align to actual available price dates
    rebalance_dates_actual = []
    for date in rebalance_dates:
        if date in prices.index:
            rebalance_dates_actual.append(date)
        else:
            # take next available index after date (if exists), otherwise take previous
            later = prices.index[prices.index >= date]
            if len(later) > 0:
                rebalance_dates_actual.append(later[0])
            else:
                earlier = prices.index[prices.index <= date]
                if len(earlier) > 0:
                    rebalance_dates_actual.append(earlier[-1])
    rebalance_dates_actual = sorted(list(dict.fromkeys(rebalance_dates_actual)))  # unique preserving order

    portfolio_value = 100.0
    portfolio_values = []
    portfolio_dates = []
    weight_history = []
    current_holdings = pd.Series(0.0, index=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        # update portfolio value by current holdings
        if i > 0 and (current_holdings > 0).any():
            portfolio_value = (current_holdings * prices.loc[date]).sum()

        portfolio_values.append(portfolio_value)
        portfolio_dates.append(date)

        if date in rebalance_dates_actual:
            prices_up_to_rebal = prices.loc[:date]
            drawdowns = calculate_drawdown_from_peak(prices_up_to_rebal, lookback_days)
            # drawdowns is dataframe where each column is drawdown series per ticker
            # pick most recent drawdown values
            if isinstance(drawdowns, pd.DataFrame):
                current_drawdowns = drawdowns.loc[date]
            else:
                current_drawdowns = drawdowns

            target_weights = calculate_weights_by_drawdown(current_drawdowns, threshold, weight_split)

            weight_change_sum = (target_weights.reindex_like(last_weights).fillna(0) - last_weights).abs().sum()

            if last_weights.sum() == 0 or weight_change_sum >= min_weight_change:
                current_prices = prices.loc[date]
                # if any target weight missing (e.g., ticker not present), align
                aligned_weights = target_weights.reindex(prices.columns).fillna(0)
                current_holdings = (portfolio_value * aligned_weights) / current_prices.replace(0, np.nan)
                current_holdings = current_holdings.fillna(0)
                last_weights = aligned_weights
                weight_history.append({'date': date, **{t: last_weights.get(t, 0.0) for t in prices.columns}})
            else:
                # no rebalancing - record existing weights
                if (current_holdings > 0).any():
                    current_value_per_stock = current_holdings * prices.loc[date]
                    if current_value_per_stock.sum() > 0:
                        current_weights = current_value_per_stock / current_value_per_stock.sum()
                    else:
                        current_weights = pd.Series(0.0, index=prices.columns)
                else:
                    current_weights = pd.Series(0.0, index=prices.columns)
                weight_history.append({'date': date, **{t: current_weights.get(t, 0.0) for t in prices.columns}})

    portfolio_series = pd.Series(portfolio_values, index=portfolio_dates).sort_index()
    weight_df = pd.DataFrame(weight_history)
    return portfolio_series, weight_df

def calculate_performance_metrics(value_series):
    """성과 지표 계산 (value_series: 가격/포트폴리오 가치 시계열)"""
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
    n_years = n_days / 252.0 if n_days > 0 else 0.0
    if n_years <= 0:
        cagr = 0.0
    else:
        cagr = ((final_value / initial_value) ** (1 / n_years) - 1) * 100

    volatility = float(returns.std() * np.sqrt(252) * 100)
    returns_std = float(returns.std())
    sharpe = float((returns.mean() * 252) / (returns_std * np.sqrt(252))) if returns_std > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    mdd = float(drawdown.min() * 100) if len(drawdown) > 0 else 0.0
    calmar = float(cagr / abs(mdd)) if abs(mdd) > 0.001 else 0.0

    return {
        'Total Return (%)': total_return,
        'CAGR (%)': cagr,
        'Volatility (%)': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': mdd,
        'Calmar Ratio': calmar
    }

def calculate_turnover(weight_history, rebalance_freq):
    """회전율 계산 (월간 및 연간)"""
    if weight_history is None or len(weight_history) < 2:
        return 0.0, 0.0

    weights_df = weight_history.copy()
    # ensure date index
    if 'date' in weights_df.columns:
        weights_df = weights_df.set_index('date')
    weights_df = weights_df.sort_index()
    total_turnover = 0.0
    rebalance_count = 0

    for i in range(1, len(weights_df)):
        w_t = weights_df.iloc[i].fillna(0)
        w_t_minus_1 = weights_df.iloc[i-1].fillna(0)
        turnover_i = (w_t - w_t_minus_1).abs().sum() / 2.0
        if w_t_minus_1.sum() > 0 or turnover_i > 0:
            total_turnover += turnover_i
            rebalance_count += 1

    if rebalance_count == 0:
        return 0.0, 0.0

    avg_rebal_turnover = total_turnover / rebalance_count
    annual_rebalances = 52 if rebalance_freq == 'W' else 12
    annual_turnover = avg_rebal_turnover * annual_rebalances
    monthly_turnover = annual_turnover / 12

    return monthly_turnover * 100, annual_turnover * 100

# ==================== 메인 앱 ====================
def main():
    st.title("📊 M7 Dynamic Portfolio Monitor")
    st.markdown("**최적화된 파라미터로 실시간 업데이트되는 M7 포트폴리오 모니터링**")

    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")

        # 날짜 선택
        st.subheader("📅 기간 설정")
        default_start = datetime(2017, 1, 1)
        default_end = datetime.now()

        start_date = st.date_input(
            "시작일",
            value=default_start,
            min_value=datetime(2010, 1, 1),
            max_value=default_end
        )

        end_date = st.date_input(
            "종료일",
            value=default_end,
            min_value=start_date,
            max_value=default_end
        )

        # 벤치마크 선택
        st.subheader("📈 비교 벤치마크")
        benchmark_option = st.radio(
            "벤치마크 선택",
            options=["M7 Equal Weight", "QQQ (Nasdaq 100)"],
            index=0
        )

        # 자동 실행 옵션 (버튼을 누르지 않아도 자동으로 실행되게 하는 옵션)
        run_on_load = st.checkbox("자동 실행 (페이지 로드 시 자동 백테스트)", value=True)

        # 파라미터 표시
        st.subheader("🎯 최적 파라미터")
        st.info(f"""
**Lookback:** {OPTIMAL_PARAMS['lookback_months']}개월  
**Rebalancing:** {"Weekly" if OPTIMAL_PARAMS['rebalance_freq']=='W' else "Monthly"}  
**Threshold:** {abs(OPTIMAL_PARAMS['threshold'])*100:.0f}%  
**Weight Split:** {OPTIMAL_PARAMS['weight_split']*100:.0f}%  
**Min Weight Change:** {OPTIMAL_PARAMS['min_weight_change']*100:.0f}%
""")

        # 실행 버튼 (사용자가 수동으로 돌릴 수 있게)
        run_button = st.button("🚀 포트폴리오 분석 실행", type="primary", use_container_width=True)

    # run if user clicked or auto-run requested
    run = run_button or run_on_load

    if not run:
        st.info("사이드바에서 기간/벤치마크를 설정한 후 '포트폴리오 분석 실행'을 누르거나 '자동 실행'을 켜면 결과가 표시됩니다.")
        return

    # 실행 흐름
    with st.spinner("데이터 다운로드 중... (yfinance)"):
        m7_prices = download_data(M7_TICKERS, start_date, end_date)

        if benchmark_option == "M7 Equal Weight":
            benchmark_prices = m7_prices.copy()
        else:
            benchmark_prices = download_data([BENCHMARK_TICKER], start_date, end_date)

        if m7_prices is None or benchmark_prices is None or m7_prices.empty:
            st.error("데이터 다운로드에 실패했거나 기간 내 데이터가 없습니다. 날짜 범위를 조정해보세요.")
            return

    with st.spinner("포트폴리오 백테스팅 중..."):
        portfolio_values, weight_history = backtest_strategy(
            m7_prices,
            OPTIMAL_PARAMS['lookback_days'],
            OPTIMAL_PARAMS['rebalance_freq'],
            OPTIMAL_PARAMS['threshold'],
            OPTIMAL_PARAMS['weight_split'],
            OPTIMAL_PARAMS['min_weight_change']
        )

        # 벤치마크 계산
        if benchmark_option == "M7 Equal Weight":
            benchmark_returns = m7_prices.pct_change().fillna(0).mean(axis=1)
            benchmark_values = (1 + benchmark_returns).cumprod() * 100.0
            benchmark_name = "M7 Equal Weight"
        else:
            if BENCHMARK_TICKER not in benchmark_prices.columns:
                st.error(f"벤치마크 데이터 {BENCHMARK_TICKER}를 가져올 수 없습니다.")
                return
            benchmark_values = benchmark_prices[BENCHMARK_TICKER] / benchmark_prices[BENCHMARK_TICKER].iloc[0] * 100.0
            benchmark_name = BENCHMARK_TICKER

        strategy_metrics = calculate_performance_metrics(portfolio_values)
        benchmark_metrics = calculate_performance_metrics(benchmark_values)
        monthly_turnover, annual_turnover = calculate_turnover(weight_history, OPTIMAL_PARAMS['rebalance_freq'])

    # ==================== 결과 표시 ====================
    st.subheader("성과 요약")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        if not portfolio_values.empty:
            fig.add_trace(go.Scatter(x=portfolio_values.index, y=portfolio_values.values,
                                     mode='lines', name='Strategy', line=dict(width=2)))
        if not benchmark_values.empty:
            fig.add_trace(go.Scatter(x=benchmark_values.index, y=benchmark_values.values,
                                     mode='lines', name=benchmark_name, line=dict(width=2, dash='dash')))
        fig.update_layout(title="포트폴리오 가치 (초기 100)", xaxis_title="날짜", yaxis_title="Value",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        # 누적수익률(%) 비교
        if not portfolio_values.empty and not benchmark_values.empty:
            p_ret = portfolio_values.pct_change().fillna(0)
            b_ret = benchmark_values.pct_change().fillna(0)
            cum_p = (1 + p_ret).cumprod() - 1
            cum_b = (1 + b_ret).cumprod() - 1
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=cum_p.index, y=cum_p.values * 100, name='Strategy %', line=dict(width=2)))
            fig2.add_trace(go.Scatter(x=cum_b.index, y=cum_b.values * 100, name=f'{benchmark_name} %', line=dict(width=2, dash='dash')))
            fig2.update_layout(title="누적수익률 (%)", xaxis_title="날짜", yaxis_title="%")
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### 주요 지표")
        if strategy_metrics is not None:
            sm_df = pd.DataFrame.from_dict(strategy_metrics, orient='index', columns=['Strategy']).round(3)
        else:
            sm_df = pd.DataFrame(index=['No Data'])
        if benchmark_metrics is not None:
            bm_df = pd.DataFrame.from_dict(benchmark_metrics, orient='index', columns=[benchmark_name]).round(3)
        else:
            bm_df = pd.DataFrame(index=['No Data'])

        metrics_df = sm_df.join(bm_df, how='outer').fillna("-")
        st.dataframe(metrics_df, use_container_width=True)

        st.markdown("### 회전율")
        st.metric(label="평균 월간 회전율 (%)", value=f"{monthly_turnover:.2f}")
        st.metric(label="예상 연간 회전율 (%)", value=f"{annual_turnover:.2f}")

    # 가중치 히트맵/표
    st.subheader("리밸런싱 시점별 가중치 히스토리")
    if weight_history is None or len(weight_history) == 0:
        st.info("리밸런싱이 수행되지 않았거나 가중치 이력이 존재하지 않습니다.")
    else:
        wh = weight_history.copy()
        if 'date' in wh.columns:
            wh = wh.set_index('date')
        wh = wh.sort_index()
        # heatmap
        fig_heat = px.imshow(wh.T.fillna(0), labels=dict(x="Date", y="Ticker", color="Weight"),
                             x=wh.index.strftime('%Y-%m-%d'), y=wh.columns, aspect="auto",
                             color_continuous_scale='Blues')
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
        # show last weights table
        st.markdown("마지막 리밸런싱 가중치")
        st.dataframe(wh.iloc[-1].to_frame(name='Weight').sort_values('Weight', ascending=False).style.format("{:.2%}"))

    # 개별 종목 가격표 및 간단 통계
    st.subheader("개별 종목 시계열 (종가)")
    st.write("아래 표는 선택 기간 내 각 종목의 종가 시계열입니다.")
    st.dataframe(m7_prices.tail(250), use_container_width=True)

    # CSV 다운로드 링크
    st.subheader("데이터 내보내기")
    csv_port = portfolio_values.rename("portfolio").to_frame().to_csv().encode('utf-8')
    st.download_button("포트폴리오 가치 CSV 다운로드", data=csv_port, file_name="portfolio_values.csv", mime="text/csv")

    if not (weight_history is None or len(weight_history) == 0):
        csv_weights = weight_history.copy()
        csv_weights['date'] = csv_weights['date'].astype(str)
        st.download_button("가중치 히스토리 CSV 다운로드", data=csv_weights.to_csv(index=False).encode('utf-8'),
                           file_name="weight_history.csv", mime="text/csv")

if __name__ == "__main__":
    main()
