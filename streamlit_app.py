import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 페이지 설정
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
    'rebalance_freq': 'M',
    'threshold': -0.3,
    'weight_split': 0.60,
    'min_weight_change': 0.0
}

M7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
BENCHMARK_TICKER = 'QQQ'

# 캐시된 데이터 다운로드
@st.cache_data(ttl=3600)
def download_data(tickers, start_date, end_date):
    """주가 데이터 다운로드"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.ffill().bfill()
    except Exception as e:
        st.error(f"데이터 다운로드 실패: {str(e)}")
        return None

def calculate_drawdown_from_peak(prices, lookback_days):
    """고점 대비 하락률 계산"""
    rolling_max = prices.rolling(window=lookback_days, min_periods=1).max()
    return (prices - rolling_max) / rolling_max

def calculate_weights_by_drawdown(drawdowns, threshold, weight_split):
    """하락률 기반 가중치 계산"""
    drawdowns = drawdowns.dropna()
    if len(drawdowns) == 0:
        return pd.Series(1.0 / len(M7_TICKERS), index=M7_TICKERS)

    deep_drawdown_mask = drawdowns <= threshold
    weights = pd.Series(0.0, index=drawdowns.index)

    if deep_drawdown_mask.sum() > 0:
        deep_dd_stocks = drawdowns[deep_drawdown_mask]
        other_stocks = drawdowns[~deep_drawdown_mask]

        deep_dd_abs = deep_dd_stocks.abs()
        if deep_dd_abs.sum() > 0:
            weights[deep_dd_stocks.index] = (deep_dd_abs / deep_dd_abs.sum()) * weight_split

        if len(other_stocks) > 0:
            other_dd_abs = other_stocks.abs()
            remaining = 1 - weight_split
            if other_dd_abs.sum() > 0:
                weights[other_stocks.index] = (other_dd_abs / other_dd_abs.sum()) * remaining
            else:
                weights[other_stocks.index] = remaining / len(other_stocks)
    else:
        drawdowns_abs = drawdowns.abs()
        if drawdowns_abs.sum() > 0:
            weights = drawdowns_abs / drawdowns_abs.sum()
        else:
            weights = pd.Series(1.0 / len(drawdowns), index=drawdowns.index)

    return weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(M7_TICKERS), index=M7_TICKERS)

def backtest_strategy(prices, lookback_days, rebalance_freq, threshold, weight_split, min_weight_change=0.0):
    """백테스팅 실행"""
    if rebalance_freq == 'W':
        rebalance_dates = prices.resample('W-MON').last().index
    else:
        rebalance_dates = prices.resample('M').last().index

    rebalance_dates_actual = []
    for date in rebalance_dates:
        if date in prices.index:
            rebalance_dates_actual.append(date)
        elif any(prices.index >= date):
            rebalance_dates_actual.append(prices.index[prices.index >= date][0])
    rebalance_dates_actual = sorted(list(set(rebalance_dates_actual)))

    portfolio_value = 100.0
    portfolio_values = []
    portfolio_dates = []
    weight_history = []
    current_holdings = pd.Series(0.0, index=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        if i > 0 and (current_holdings > 0).any():
            portfolio_value = (current_holdings * prices.loc[date]).sum()

        portfolio_values.append(portfolio_value)
        portfolio_dates.append(date)

        if date in rebalance_dates_actual:
            prices_up_to_rebal = prices.loc[:date]
            drawdowns = calculate_drawdown_from_peak(prices_up_to_rebal, lookback_days)
            current_drawdowns = drawdowns.loc[date]
            target_weights = calculate_weights_by_drawdown(current_drawdowns, threshold, weight_split)

            weight_change_sum = (target_weights - last_weights).abs().sum()

            if last_weights.sum() == 0 or weight_change_sum >= min_weight_change:
                current_prices = prices.loc[date]
                current_holdings = (portfolio_value * target_weights) / current_prices
                last_weights = target_weights

                weight_history.append({
                    'date': date,
                    **{ticker: target_weights.get(ticker, 0) for ticker in prices.columns}
                })
            else:
                if (current_holdings > 0).any():
                    current_value_per_stock = current_holdings * prices.loc[date]
                    current_weights = current_value_per_stock / current_value_per_stock.sum()
                else:
                    current_weights = pd.Series(0.0, index=prices.columns)

                weight_history.append({
                    'date': date,
                    **{ticker: current_weights.get(ticker, 0) for ticker in prices.columns}
                })

    portfolio_data = pd.DataFrame({'value': portfolio_values}, index=portfolio_dates)
    portfolio_data = portfolio_data[~portfolio_data.index.duplicated(keep='first')].sort_index()
    weight_df = pd.DataFrame(weight_history)

    return portfolio_data['value'], weight_df

def calculate_performance_metrics(returns_series):
    """성과 지표 계산"""
    if isinstance(returns_series, pd.DataFrame):
        returns_series = returns_series.squeeze()
    if len(returns_series) < 2:
        return None

    returns = returns_series.pct_change().dropna()
    if len(returns) == 0:
        return None

    final_value = float(returns_series.iloc[-1])
    initial_value = float(returns_series.iloc[0])
    total_return = (final_value / initial_value - 1) * 100
    n_years = len(returns) / 252

    if n_years <= 0:
        return None

    cagr = ((final_value / initial_value) ** (1 / n_years) - 1) * 100
    volatility = float(returns.std() * np.sqrt(252) * 100)

    returns_std = float(returns.std())
    sharpe = float((returns.mean() * 252) / (returns_std * np.sqrt(252))) if returns_std > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    mdd = float(drawdown.min() * 100)
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
    """회전율 계산"""
    if len(weight_history) < 2:
        return 0.0, 0.0

    weights_df = weight_history.set_index('date').drop(columns=['date'], errors='ignore')
    total_turnover = 0.0
    rebalance_count = 0

    for i in range(1, len(weights_df)):
        w_t = weights_df.iloc[i]
        w_t_minus_1 = weights_df.iloc[i-1]
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
        
        # 자동 시작일 계산 (2017-01-01부터 시작하는 것으로 가정)
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
        
        # 파라미터 표시
        st.subheader("🎯 최적 파라미터")
        st.info(f"""
        **Lookback:** {OPTIMAL_PARAMS['lookback_months']}개월  
        **Rebalancing:** Monthly  
        **Threshold:** {abs(OPTIMAL_PARAMS['threshold'])*100:.0f}%  
        **Weight Split:** {OPTIMAL_PARAMS['weight_split']*100:.0f}%  
        **Min Weight Change:** {OPTIMAL_PARAMS['min_weight_change']*100:.0f}%
        """)
        
        # 실행 버튼
        run_button = st.button("🚀 포트폴리오 분석 실행", type="primary", use_container_width=True)
    
    if run_button:
        with st.spinner("데이터 다운로드 중..."):
            # 데이터 다운로드
            m7_prices = download_data(M7_TICKERS, start_date, end_date)
            
            if benchmark_option == "M7 Equal Weight":
                benchmark_prices = m7_prices.copy()
            else:
                benchmark_prices = download_data([BENCHMARK_TICKER], start_date, end_date)
            
            if m7_prices is None or benchmark_prices is None:
                st.error("데이터 다운로드에 실패했습니다.")
                return
        
        with st.spinner("포트폴리오 백테스팅 중..."):
            # 전략 백테스팅
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
                benchmark_values = (1 + benchmark_returns).cumprod() * 100
                benchmark_name = "M7 Equal Weight"
            else:
                benchmark_values = (benchmark_prices[BENCHMARK_TICKER] / benchmark_prices[BENCHMARK_TICKER].iloc[0] * 100)
                benchmark_name = "QQQ"
            
            # 성과 지표 계산
            strategy_metrics = calculate_performance_metrics(portfolio_values)
            benchmark_metrics = calculate_performance_metrics(benchmark_values)
            monthly_turnover, annual_turnover = calculate_turnover(weight_history, OPTIMAL_PARAMS['rebalance_freq'])
        
        # ==================== 결과 표시 ====================
        