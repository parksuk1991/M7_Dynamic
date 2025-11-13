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

import requests
from PIL import Image
from io import BytesIO

# 페이지 설정
st.set_page_config(
    page_title="M7 Contrarian Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 고정(하드코딩) 전략 파라미터
# =========================
OPTIMAL_PARAMS = {
    'lookback_months': 3,
    'lookback_days': 63,
    'rebalance_freq': 'M',
    'threshold': -0.3,
    'weight_split': 0.60,
    'min_weight_change': 0.0,
    'cap_weight': 0.60
}

# 기본/디폴트 티커 (M7)
M7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
BENCHMARK_TICKER = 'QQQ'

# Color theme
PRIMARY_COLOR = 'deeppink'
SECONDARY_COLOR = 'royalblue'
PASTEL_PALETTE = px.colors.sequential.RdPu_r

# -------------------------
# 캐시 / 유틸리티 함수
# -------------------------
@st.cache_data(ttl=3600)
def download_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """주가 데이터 다운로드 (종가)."""
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
def get_first_available_date(ticker: str) -> Optional[date]:
    """티커의 첫 거래 가능일 반환."""
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

def calculate_weights_by_drawdown(drawdowns: pd.Series, threshold: float, weight_split: float, cap_weight: float = 1.0) -> pd.Series:
    """하락률 기반 가중치 계산 with Cap Weight 제한."""
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
    
    # Cap Weight 적용
    max_iterations = 10
    iteration = 0
    
    while weights.max() > cap_weight and iteration < max_iterations:
        max_ticker = weights.idxmax()
        excess = weights[max_ticker] - cap_weight
        weights[max_ticker] = cap_weight
        other_tickers = weights.index[weights.index != max_ticker]
        if len(other_tickers) > 0 and weights[other_tickers].sum() > 0:
            weights[other_tickers] = weights[other_tickers] * (1 + excess / weights[other_tickers].sum())
        else:
            weights[other_tickers] = excess / len(other_tickers)
        iteration += 1
    
    if weights.sum() > 0:
        weights = weights / weights.sum()
    
    return weights

def backtest_strategy(prices: pd.DataFrame, lookback_days: int, rebalance_freq: str, threshold: float,
                      weight_split: float, min_weight_change: float = 0.0, cap_weight: float = 1.0,
                      use_loss_cut: bool = False, individual_loss_threshold: float = -0.15) -> Tuple[pd.Series, pd.DataFrame, List[Dict]]:
    """
    백테스트 수행 with Loss Cut 로직.
    
    Loss Cut 로직:
    1. 리밸런싱 시점에만 체크 (매일 체크 X)
    2. 전월부터 보유한 종목 중 손실률이 threshold 이하인 경우만 매도
    3. 신규 편입 종목은 Loss Cut 체크 제외
    4. 손절된 비중은 나머지 보유 종목에 Pro-rata로 재분배
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float), pd.DataFrame(), []

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
    loss_cut_events = []
    current_holdings = pd.Series(0.0, index=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)
    entry_prices = pd.Series(0.0, index=prices.columns)  # 각 종목의 진입가
    previous_period_tickers = set()  # 전월 보유 종목

    for i, date in enumerate(prices.index):
        # 포트폴리오 가치 업데이트
        if i > 0 and (current_holdings > 0).any():
            portfolio_value = (current_holdings * prices.loc[date]).sum()

        pv_list.append(portfolio_value)
        pv_dates.append(date)

        # 리밸런싱 날짜 체크
        if date in reb_actual:
            current_prices = prices.loc[date]
            
            # 1단계: 목표 가중치 계산 (기존 전략)
            prices_up_to = prices.loc[:date]
            drawdowns = calculate_drawdown_from_peak(prices_up_to, lookback_days)
            cur_dd = drawdowns.loc[date] if isinstance(drawdowns, pd.DataFrame) else drawdowns
            target_weights = calculate_weights_by_drawdown(cur_dd, threshold, weight_split, cap_weight)
            aligned_target = target_weights.reindex(prices.columns).fillna(0)
            
            # 2단계: Loss Cut 체크 (활성화된 경우만)
            if use_loss_cut and len(previous_period_tickers) > 0:
                tickers_to_cut = []
                
                # 전월부터 보유한 종목만 체크
                for ticker in previous_period_tickers:
                    # 이번 달에도 편입 예정이고
                    if aligned_target[ticker] > 0:
                        # 진입가가 기록되어 있고
                        if entry_prices[ticker] > 0:
                            # 현재가와 진입가 비교
                            holding_return = (current_prices[ticker] - entry_prices[ticker]) / entry_prices[ticker]
                            
                            # 손실률이 threshold 이하인 경우
                            if holding_return <= individual_loss_threshold:
                                tickers_to_cut.append(ticker)
                                
                                loss_cut_events.append({
                                    'date': date,
                                    'ticker': ticker,
                                    'entry_price': entry_prices[ticker],
                                    'current_price': current_prices[ticker],
                                    'loss': holding_return * 100,
                                    'portfolio_value': portfolio_value
                                })
                
                # Loss Cut 종목이 있으면 가중치 재조정
                if len(tickers_to_cut) > 0:
                    # Loss Cut 종목 제거
                    for ticker in tickers_to_cut:
                        aligned_target[ticker] = 0.0
                    
                    # 남은 종목들에 Pro-rata 재분배
                    if aligned_target.sum() > 0:
                        aligned_target = aligned_target / aligned_target.sum()
                    else:
                        # 모든 종목이 Loss Cut된 경우 현금 보유
                        aligned_target = pd.Series(0.0, index=prices.columns)
            
            # 3단계: 가중치 변화 체크 및 리밸런싱
            weight_change_sum = (aligned_target - last_weights).abs().sum()

            if last_weights.sum() == 0 or weight_change_sum >= min_weight_change:
                # 리밸런싱 실행
                current_holdings = (portfolio_value * aligned_target) / current_prices.replace(0, np.nan)
                current_holdings = current_holdings.fillna(0)
                last_weights = aligned_target.copy()
                
                # 진입가 업데이트 (신규 편입 또는 비중 증가한 종목만)
                current_period_tickers = set(aligned_target[aligned_target > 0].index)
                for ticker in current_period_tickers:
                    # 신규 편입이거나 전월에 없었던 종목
                    if ticker not in previous_period_tickers:
                        entry_prices[ticker] = current_prices[ticker]
                    # 전월에도 있었지만 Loss Cut으로 제외되지 않은 경우는 진입가 유지
                
                # 다음 달을 위해 현재 보유 종목 기록
                previous_period_tickers = current_period_tickers.copy()
                
                weight_history.append({'date': date, **{t: last_weights.get(t, 0.0) for t in prices.columns}})
            else:
                # 리밸런싱 스킵 - 현재 비중 기록
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
    return portfolio_series, weight_df, loss_cut_events

def calculate_performance_metrics(value_series: pd.Series, benchmark_series: Optional[pd.Series] = None, is_benchmark: bool = False) -> dict:
    """성과 지표 계산."""
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
    if not is_benchmark:
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

def weights_history_to_composition_dict(weight_history: pd.DataFrame, rebalance_freq: str = 'M') -> Dict[date, Dict[str, float]]:
    """weight_history를 {date: {ticker: weight}} 형태로 변환."""
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

def get_rebalancing_changes(current: Dict[str,float], previous: Dict[str,float]) -> Dict[str, Dict]:
    """두 가중치 dict 비교해서 변화 리턴."""
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

def create_excess_return_heatmap(strat_returns: pd.Series, bench_returns: pd.Series) -> pd.DataFrame:
    """초과성과 히트맵용 데이터 생성"""
    strat_monthly = (1 + strat_returns).resample('M').prod() - 1
    bench_monthly = (1 + bench_returns).resample('M').prod() - 1
    
    common_idx = strat_monthly.index.intersection(bench_monthly.index)
    strat_monthly = strat_monthly.reindex(common_idx)
    bench_monthly = bench_monthly.reindex(common_idx)
    
    excess = strat_monthly - bench_monthly
    
    df_excess = excess.to_frame(name='ret')
    df_excess['year'] = df_excess.index.year
    df_excess['month'] = df_excess.index.month
    
    pivot_excess = df_excess.pivot_table(index='year', columns='month', values='ret', aggfunc='first')
    pivot_excess = pivot_excess.reindex(columns=range(1,13))
    pivot_excess_pct = (pivot_excess * 100).round(2)
    
    month_names = [datetime(1900, m, 1).strftime('%b') for m in range(1,13)]
    pivot_excess_pct.columns = month_names
    
    return pivot_excess_pct

# -------------------------
# 스트림릿 UI
# -------------------------
def main():
    # -------------------- 상단 레이아웃 ---------------------
    col_title, col_img_credit = st.columns([8, 1])
    with col_title:
        st.title("📈 M7 Contrarian Strategy")
        st.markdown("낙폭 과대 기준 Mean Reversion 포트폴리오 (with Loss Cut)")
    with col_img_credit:
        image_url = "https://amateurphotographer.com/wp-content/uploads/sites/7/2017/08/Screen-Shot-2017-08-23-at-22.29.18.png?w=600.jpg"
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            st.image(img, width=150, caption=None)
        except Exception:
            st.info("이미지를 불러올 수 없습니다.")
            
        st.markdown(
        "<div style='margin-top: -1px; text-align:center;'>"
        "<span style='font-size:0.75rem; color:#888;'>Made by CP3</span>"
        "</div>",
        unsafe_allow_html=True
        )
       
        st.markdown(
            '<div style="text-align: left; margin-bottom: 3px; font-size:0.9rem;">'
            'Data 출처: <a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a>'
            '</div>',
            unsafe_allow_html=True
        )

    with st.expander("📋 전략 로직 자세히 보기", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            #### 🎯 <font color='blueviolet'>주요 파라미터 <font color='black'>
            - <font color='black'>**Drawdown(3M)**: 고점 대비 하락률로 최근 3개월 고점 기준 하락폭이 클수록 저평가 판단
            - <font color='black'>**Threshold(-30%)**: 심각한 하락의 기준  
            - <font color='black'>**Weight Split(60%)**: 심각한 하락 종목(-30% 이하)에 60%를 배분하고 나머지 40%은 다른 종목에 분산
            - <font color='black'>**Cap Weight(60%)**: 단일 종목 최대 비중 제한 (초과 시 pro-rata 재분배)
            
            #### 🛡️ <font color='blueviolet'>Loss Cut 로직 <font color='black'>
            - <font color='black'>**개별 종목 손실 제한**: 리밸런싱 시점에 체크
            - <font color='black'>**전월 보유 종목만 체크**: 신규 편입 종목은 Loss Cut 제외
            - <font color='black'>**손실 기준 초과 시**: 해당 종목 전액 매도 후 나머지 보유 종목에 Pro-rata 재분배
            - <font color='black'>**기준 미달 시**: 다음 달에도 동일 종목 보유 시 진입가 기준 유지
            
            #### ✔️ <font color='blueviolet'>전략 요약 <font color='black'>
            - <font color='black'>Drawdown 기준 Threshold 이하 하락 종목에 Weight Split% 배분 | 나머지 종목에 (1-Weight Split)% 배분
            - <font color='black'>Threshold 이하로 하락한 종목이 없을 경우 전체를 하락폭 비례로 배분
            - <font color='black'>Loss Cut 활성화 시 개별 종목 손실 제한 적용 (리밸런싱 시점에만)
            - <font color='black'>모든 파라미터는 Walk Forward 최적화로 Look-ahead Bias 통제 하에 Pre-trained 완료
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="
                height: 905px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                margin-top: 40px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                "></div>
            </div>
            """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ 설정")
        tickers_default = ", ".join(M7_TICKERS)
        
        tickers_input = st.text_area(
            "종목 티커 (콤마로 구분)", 
            value=tickers_default, 
            placeholder="예: AAPL, MSFT, TSLA", 
            height=100,
            help="Default Tickers: M7" 
        )
        
        tickers = [t.strip().upper() for t in tickers_input.replace(';', ',').split(',') if t.strip() != ""]
      
        st.subheader("📅 기간 ")
        default_start = datetime(2015, 1, 1)
        default_end = datetime.now()
        start_date = st.date_input("시작일", value=default_start.date(), min_value=datetime(1990,1,1).date(), max_value=default_end.date())
        end_date = st.date_input("종료일", value=default_end.date(), min_value=start_date, max_value=default_end.date())

        st.subheader("📈 벤치마크")
        benchmark_option = st.selectbox("벤치마크 선택", options=["동일 가중 포트폴리오", f"{BENCHMARK_TICKER} (Nasdaq 100)"], index=0)
        
        # Loss Cut 옵션
        st.subheader("🛡️ Loss Cut 설정")
        use_loss_cut = st.checkbox("Loss Cut 활성화", value=False, help="개별 종목 손실 제한 적용 (리밸런싱 시점에만 체크)")
        
        individual_loss_threshold = -0.15  # 기본값
        if use_loss_cut:
            loss_pct = st.slider(
                "개별 종목 손실 제한 (%)",
                min_value=-50.0,
                max_value=-5.0,
                value=-15.0,
                step=1.0,
                help="전월부터 보유한 종목이 이 비율 이하로 하락하면 매도"
            )
            individual_loss_threshold = loss_pct / 100.0
            
            st.info(f"""
            **개별 종목 손실 제한:** {abs(individual_loss_threshold)*100:.0f}%  
            
            - 리밸런싱 시점에만 체크
            - 전월 보유 종목만 적용
            - 신규 편입 종목 제외
            """)
       
        st.subheader("🎯 최적 파라미터\n(Pre-trained Parameters)")
        st.info(f"""
        **Lookback:** {OPTIMAL_PARAMS['lookback_months']}개월  
        **Rebalancing:** {"Weekly" if OPTIMAL_PARAMS['rebalance_freq']=='W' else "Monthly"}  
        **Threshold:** {abs(OPTIMAL_PARAMS['threshold'])*100:.0f}%  
        **Weight Split:** {OPTIMAL_PARAMS['weight_split']*100:.0f}%  
        **Cap Weight:** {OPTIMAL_PARAMS['cap_weight']*100:.0f}%  
        **Min Weight Change:** {OPTIMAL_PARAMS['min_weight_change']*100:.0f}%
        """)
        run_button = st.button("🚀 포트폴리오 생성", type="primary", use_container_width=True)
    
    if not run_button:
        st.info("사이드바에서 티커, 기간, 벤치마크 설정 후 '포트폴리오 생성' 클릭")
        return

    if len(tickers) == 0:
        st.error("티커 목록이 비어 있습니다. 하나 이상의 티커를 입력하세요.")
        return

    with st.spinner("데이터 처리 중..."):
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

    with st.spinner("선택 기간 데이터 다운로드 중..."):
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        prices = download_data(tickers, start_dt, end_dt)
        if benchmark_option.startswith(BENCHMARK_TICKER):
            benchmark_prices = download_data([BENCHMARK_TICKER], start_dt, end_dt)
            benchmark_name = BENCHMARK_TICKER
        else:
            benchmark_prices = prices.copy()
            benchmark_name = "Equal Weight Portfolio"

    if prices is None or prices.empty:
        st.error("종목 데이터 다운로드 실패 또는 기간 내 데이터가 없습니다.")
        return

    lookback_days = OPTIMAL_PARAMS['lookback_days']
    rebalance_freq = OPTIMAL_PARAMS['rebalance_freq']
    threshold = OPTIMAL_PARAMS['threshold']
    weight_split = OPTIMAL_PARAMS['weight_split']
    min_weight_change = OPTIMAL_PARAMS['min_weight_change']
    cap_weight = OPTIMAL_PARAMS['cap_weight']

    with st.spinner("로딩중..."):
        portfolio_values, weight_history, loss_cut_events = backtest_strategy(
            prices, lookback_days, rebalance_freq, threshold, weight_split, min_weight_change, cap_weight,
            use_loss_cut=use_loss_cut,
            individual_loss_threshold=individual_loss_threshold
        )

    if portfolio_values is None or portfolio_values.empty:
        st.error("백테스트 결과가 없습니다.")
        return

    if benchmark_option.startswith(BENCHMARK_TICKER):
        if benchmark_prices is None or benchmark_prices.empty or BENCHMARK_TICKER not in benchmark_prices.columns:
            st.error(f"벤치마크 {BENCHMARK_TICKER} 데이터를 가져올 수 없습니다.")
            return
        bench_vals = benchmark_prices[BENCHMARK_TICKER] / benchmark_prices[BENCHMARK_TICKER].iloc[0] * 100.0
    else:
        returns = prices.pct_change().fillna(0)
        bench_returns = returns.mean(axis=1)
        bench_vals = (1 + bench_returns).cumprod() * 100.0

    strategy_metrics = calculate_performance_metrics(portfolio_values, bench_vals, is_benchmark=False)
    benchmark_metrics = calculate_performance_metrics(bench_vals, portfolio_values, is_benchmark=True)
    monthly_turnover, annual_turnover = calculate_turnover(weight_history, rebalance_freq)

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

    # Loss Cut 이벤트 표시
    if use_loss_cut and len(loss_cut_events) > 0:
        st.subheader("🛡️ Loss Cut 이벤트")
        
        col_summary, col_detail = st.columns([1, 2])
        
        with col_summary:
            st.metric("총 Loss Cut 이벤트", f"{len(loss_cut_events)}회")
            
            # 종목별 집계
            ticker_counts = {}
            for event in loss_cut_events:
                ticker = event['ticker']
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            
            st.write("**종목별 Loss Cut 횟수**")
            for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {ticker}: {count}회")
        
        with col_detail:
            st.write("**Loss Cut 상세 내역**")
            loss_cut_df = pd.DataFrame(loss_cut_events)
            loss_cut_df['date'] = pd.to_datetime(loss_cut_df['date']).dt.strftime('%Y-%m-%d')
            loss_cut_df['entry_price'] = loss_cut_df['entry_price'].apply(lambda x: f"${x:.2f}")
            loss_cut_df['current_price'] = loss_cut_df['current_price'].apply(lambda x: f"${x:.2f}")
            loss_cut_df['loss'] = loss_cut_df['loss'].apply(lambda x: f"{x:.2f}%")
            loss_cut_df['portfolio_value'] = loss_cut_df['portfolio_value'].apply(lambda x: f"${x:.2f}")
            
            display_df = loss_cut_df[['date', 'ticker', 'entry_price', 'current_price', 'loss']]
            display_df.columns = ['날짜', '종목', '진입가', '현재가', '손실률']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    elif use_loss_cut:
        st.info("🛡️ Loss Cut 활성화 상태 - 손실 제한 기준을 충족한 이벤트가 없습니다.")

    # UI 출력
    st.subheader("성과")

    col_left, col_right = st.columns(2)
    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strat_cum.index, y=(strat_cum - 1) * 100, name="Strategy", line=dict(color=PRIMARY_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=bench_cum.index, y=(bench_cum - 1) * 100, name=f"{benchmark_name}", line=dict(color=SECONDARY_COLOR, width=2, dash='dash')))
        
        # Loss Cut 이벤트 표시
        if use_loss_cut and len(loss_cut_events) > 0:
            for event in loss_cut_events:
                event_date = pd.Timestamp(event['date'])
                if event_date in strat_cum.index:
                    event_value = (strat_cum.loc[event_date] - 1) * 100
                    fig.add_trace(go.Scatter(
                        x=[event_date], y=[event_value],
                        mode='markers',
                        marker=dict(size=10, color='orange', symbol='triangle-down'),
                        name='Loss Cut',
                        showlegend=False,
                        hovertemplate=f"<b>Loss Cut</b><br>Date: {event_date.strftime('%Y-%m-%d')}<br>Ticker: {event['ticker']}<br>Loss: {event['loss']:.2f}%<extra></extra>"
                    ))
        
        title_suffix = " (Loss Cut 이벤트 표시)" if use_loss_cut and len(loss_cut_events) > 0 else ""
        fig.update_layout(title=f"누적수익률 (%){title_suffix}", template="plotly_white", hovermode='x unified',
                          legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.6)'))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(x=strat_cum.index, y=np.log(np.maximum(strat_cum.values, 1e-8)), name="Strategy (log)", line=dict(color=PRIMARY_COLOR, width=2)))
        fig_log.add_trace(go.Scatter(x=bench_cum.index, y=np.log(np.maximum(bench_cum.values, 1e-8)), name=f"{benchmark_name} (log)", line=dict(color=SECONDARY_COLOR, width=2, dash='dash')))
        fig_log.update_layout(title="로그 누적수익률", template="plotly_white", hovermode='x unified',
                              legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.6)'))
        st.plotly_chart(fig_log, use_container_width=True)

    st.subheader("주요 지표")
    ordered_index = ['Total Return (%)', 'CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Tracking Error (%)', 'Calmar Ratio', 'Annual Turnover (%)']
    metrics_df = pd.DataFrame(index=ordered_index)
    
    if strategy_metrics is not None:
        strat_dict = strategy_metrics.copy()
        strat_dict['Annual Turnover (%)'] = annual_turnover
        strategy_label = f"Strategy (Loss Cut {abs(individual_loss_threshold)*100:.0f}%)" if use_loss_cut else "Strategy"
        metrics_df = metrics_df.join(pd.DataFrame.from_dict(strat_dict, orient='index', columns=[strategy_label]))
    
    if benchmark_metrics is not None:
        bench_dict = benchmark_metrics.copy()
        bench_dict['Annual Turnover (%)'] = np.nan
        metrics_df = metrics_df.join(pd.DataFrame.from_dict(bench_dict, orient='index', columns=[benchmark_name]))
    
    metrics_df = metrics_df.replace({np.nan: "-"})
    
    for col in metrics_df.columns:
        metrics_df[col] = metrics_df[col].apply(lambda x: round(x, 3) if isinstance(x, (int, float)) else x)
    
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("낙폭 (Drawdown)")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values * 100, fill='tozeroy', mode='none', name='Strategy DD (area)',
                                 showlegend=False, fillcolor='rgba(255,20,147,0.18)', hoverinfo='x+y'))
    fig_dd.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values * 100, mode='lines', name='Strategy DD',
                                 line=dict(color=PRIMARY_COLOR, width=1)))
    fig_dd.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values * 100, fill='tozeroy', mode='none', name='Benchmark DD (area)',
                                 showlegend=False, fillcolor='rgba(65,105,225,0.12)', hoverinfo='x+y'))
    fig_dd.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values * 100, mode='lines', name='Benchmark DD',
                                 line=dict(color=SECONDARY_COLOR, width=1, dash='dash')))
    
    # Loss Cut 이벤트 표시
    if use_loss_cut and len(loss_cut_events) > 0:
        for event in loss_cut_events:
            event_date = pd.Timestamp(event['date'])
            if event_date in strat_dd.index:
                event_dd = strat_dd.loc[event_date] * 100
                fig_dd.add_trace(go.Scatter(
                    x=[event_date], y=[event_dd],
                    mode='markers',
                    marker=dict(size=10, color='orange', symbol='triangle-down'),
                    name='Loss Cut',
                    showlegend=False,
                    hovertemplate=f"<b>Loss Cut</b><br>Date: {event_date.strftime('%Y-%m-%d')}<br>Ticker: {event['ticker']}<extra></extra>"
                ))
    
    fig_dd.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)",
                         template="plotly_white", hovermode='x unified',
                         legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top'))
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("비중 히스토리")
    if weight_history is None or len(weight_history) == 0:
        st.info("리밸런싱 가중치 이력이 없습니다.")
        weights_composition = {}
    else:
        wh = weight_history.copy()
        if 'date' in wh.columns:
            wh['date'] = pd.to_datetime(wh['date'])
            wh = wh.set_index('date')
        wh = wh.sort_index()
        
        wh_pct = (wh * 100).round(3)
        st.dataframe(wh_pct, use_container_width=True)

        try:
            heat_df = wh.fillna(0).T
            heat_df.columns = [pd.to_datetime(c).strftime('%Y-%m-%d') if not isinstance(c, str) else c for c in heat_df.columns]
            fig_heat = px.imshow(heat_df, labels=dict(x="Rebalance Date", y="Ticker", color="Weight"),
                                x=heat_df.columns, y=heat_df.index, color_continuous_scale='RdPu', aspect="auto")
            fig_heat.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception:
            st.warning("히트맵 생성 중 문제가 발생했습니다.")

        weights_composition = weights_history_to_composition_dict(weight_history, rebalance_freq=rebalance_freq)

    st.subheader(f"📰 포트폴리오 업데이트 ({date.today().strftime('%Y-%m')} 기준)")
    if weights_composition:
        recent_dates = sorted(weights_composition.keys())
        if len(recent_dates) == 0:
            st.info("월말 기준으로 사용할 리밸런싱 데이터가 없습니다.")
        else:
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

                fig_pie = px.pie(names=list(current_weights.keys()), values=list(current_weights.values()),
                                title="📒 현재 비중", color_discrete_sequence=PASTEL_PALETTE, template='plotly_dark')
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                if previous_weights:
                    st.write(f"**📙 전월 대비 리밸런싱 추이** ({previous_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')})")
                    changes = get_rebalancing_changes(current_weights, previous_weights)
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

                    stocks = [r['종목'] for r in rebalancing_data]
                    changes_values = [float(r['변화'].replace('%',''))/100.0 for r in rebalancing_data]
                    colors = [PRIMARY_COLOR if v > 0 else SECONDARY_COLOR for v in changes_values]
                    fig_rebal = go.Figure(data=[go.Bar(x=stocks, y=[x*100 for x in changes_values],
                                                       marker_color=colors, text=[f"{x:+.2%}" for x in changes_values],
                                                       textposition='auto')])
                    fig_rebal.update_layout(title="📗 리밸런싱 추이 (%p)", xaxis_title="종목", yaxis_title="비중 변화 (%p)",
                                           template="plotly_white", height=400)
                    st.plotly_chart(fig_rebal, use_container_width=True)
                else:
                    st.info("비교할 이전 포트폴리오 데이터가 없습니다.")
    else:
        st.info("리밸런싱 구성 데이터가 없습니다.")

    st.subheader("월별 수익률 분포")
    strat_monthly = (1 + strat_returns).resample('M').prod() - 1
    bench_monthly = (1 + bench_returns).resample('M').prod() - 1

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=strat_monthly.values * 100, name='포트폴리오', opacity=0.75,
                                    marker_color=PRIMARY_COLOR, nbinsx=24))
    fig_hist.add_trace(go.Histogram(x=bench_monthly.values * 100, name='벤치마크', opacity=0.5,
                                    marker_color=SECONDARY_COLOR, nbinsx=24))
    fig_hist.update_layout(title="월별 수익률 분포 (%)", xaxis_title="월별 수익률 (%)", yaxis_title="빈도",
                          barmode='overlay', template="plotly_white", height=520,
                          legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.6)'))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("초과성과 히트맵")
    excess_heatmap = create_excess_return_heatmap(strat_returns, bench_returns)
    if not excess_heatmap.empty:
        st.markdown("### 월별 초과성과 (%) - Portfolio vs Benchmark")
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=excess_heatmap.values,
            x=excess_heatmap.columns,
            y=excess_heatmap.index,
            colorscale='RdPu',
            text=excess_heatmap.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Excess Return (%)")
        ))
        
        fig_heatmap.update_layout(
            xaxis_title="Month",
            yaxis_title="Year",
            height=max(400, len(excess_heatmap) * 40),
            template="plotly_white",
            xaxis=dict(
                side='top',
                tickmode='linear',
                dtick=1
            ),
            yaxis=dict(
                tickmode='linear',
                dtick=1,
                autorange='reversed'
            )
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)

    else:
        st.info("초과성과 데이터가 부족합니다.")

    st.subheader("포트폴리오 구성 히스토리 (최근 6개월)")
    if 'weights_composition' in locals() and weights_composition:
        recent_dates_comp = sorted(weights_composition.keys())[-6:]
        for date_key in recent_dates_comp:
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
                    fig_pie = px.pie(names=list(weights.keys()), values=list(weights.values()),
                                   title="가중치 분포", color_discrete_sequence=PASTEL_PALETTE, template='plotly_dark')
                    fig_pie.update_traces(textinfo='percent+label')
                    fig_pie.update_layout(height=300, template="plotly_white")
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("가중치 히스토리가 없습니다.")

    st.markdown("---")

if __name__ == "__main__":
    main()
