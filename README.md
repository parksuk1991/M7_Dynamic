# 📈 M7 Contrarian Strategy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data Source](https://img.shields.io/badge/Data-Yahoo%20Finance-purple.svg)](https://finance.yahoo.com/)

> **낙폭 과대 기준 Mean Reversion 포트폴리오 전략**  

Link: https://m7dynamic.streamlit.app/

---

## 🎯 개요

**M7 Contrarian Strategy**는 주식 시장의 평균회귀(Mean Reversion) 기회를 포착하기 위해 설계된 정량적 투자 프레임워크입니다. 가격 하락폭의 심각도를 기반으로 자본을 동적으로 배분함으로써, 저평가된 자산에 체계적으로 비중을 확대하면서도 엄격한 리스크 관리를 유지합니다.

### 주요 특징

- ✅ **낙폭 기반 가중치**: 가격 하락폭에 비례하여 비중 배분
- ✅ **임계값 기반 배분**: 심각한 하락(-30% 이상)과 일반 하락 구분
- ✅ **포지션 크기 제한**: 단일 종목 최대 60% 비중 제한
- ✅ **Walk-Forward 최적화**: Look-ahead Bias 통제 하에서 주요 파라미터 사전 최적화 완료
- ✅ **인터랙티브 대시보드**: 실시간 시각화 및 포트폴리오 분석
- ✅ **유연한 자산 유니버스**: 기본 M7(Magnificent 7) 또는 사용자 정의 종목 선택

---

## 📊 전략 로직

### Pre-trained 파라미터

| 파라미터 | 값 | 설명 |
|---------|---|------|
| **Lookback Period** | 3개월 (63일) | 낙폭 계산을 위한 롤링 윈도우 |
| **Rebalancing Frequency** | 월간 | 포트폴리오 리밸런싱 주기 |
| **Threshold** | -30% | 심각한 하락 기준 |
| **Weight Split** | 60% | 심각한 하락 종목에 대한 비중 배분 비율 |
| **Cap Weight** | 60% | 단일 종목 최대 비중 |
| **Min Weight Change** | 0% | 리밸런싱 최소 트리거 임계값 |

### 배분 방법론

이 전략은 3단계 배분 프레임워크를 사용합니다:

#### 1. **심각한 하락 자산** (낙폭 ≤ -30%)
```
가중치_i = (|낙폭_i| / Σ|심각한낙폭|) × 60%
```
30% 이상 하락한 자산들은 전체 자본의 60%를 받으며, 각 자산의 하락폭에 비례하여 배분됩니다.

#### 2. **일반 하락 자산** (낙폭 > -30%)
```
가중치_j = (|낙폭_j| / Σ|일반낙폭|) × 40%
```
나머지 자산들은 잔여 40%를 공유하며, 상대적 낙폭에 따라 가중됩니다.

#### 3. **전체 상승 시나리오**
```
가중치_k = 1/N
```
모든 자산이 상승 국면일 때(낙폭 = 0), 포트폴리오는 균등 가중으로 기본 설정됩니다.

#### 4. **Cap Weight 적용**
초기 배분 후 특정 종목이 60%를 초과하는 경우:
```
1. 초과 비중 종목을 60%로 제한
2. 초과분 계산: 초과분 = 가중치_최대 - 60%
3. 나머지 종목에 비례 재분배 (pro-rata)
4. 모든 종목이 ≤ 60%가 될 때까지 반복
```

---

## 🚀 빠른 시작

### 사전 요구사항

```bash
Python 3.8 이상
pip (Python 패키지 매니저)
```

### 설치 방법

1. **저장소 복제**
```bash
git clone https://github.com/yourusername/m7-contrarian-strategy.git
cd m7-contrarian-strategy
```

2. **의존성 설치**
```bash
pip install -r requirements.txt
```

3. **애플리케이션 실행**
```bash
streamlit run app.py
```

4. **대시보드 접속**
```
브라우저를 열고 다음 주소로 이동:
http://localhost:8501
```

---

## 📦 의존성 패키지

```txt
streamlit>=1.28.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
Pillow>=10.0.0
requests>=2.31.0
```

위 내용으로 `requirements.txt` 파일을 생성하세요.

---

## 💻 사용 방법

### 기본 설정

1. **종목 선택**: 쉼표로 구분된 티커 심볼 입력 (기본값: M7 종목)
   ```
   AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
   ```

2. **기간 설정**: 백테스트 시작일과 종료일 선택

3. **벤치마크 선택**: 
   - 동일 가중 포트폴리오
   - QQQ (Nasdaq 100 ETF)

4. **포트폴리오 생성**: "🚀 포트폴리오 생성" 버튼 클릭

### 고급 기능

- **비중 히스토리 히트맵**: 시간에 따른 포트폴리오 구성 시각화
- **낙폭 분석**: 전략 vs. 벤치마크 낙폭 프로파일 비교
- **초과수익 히트맵**: 월별 초과성과 시각화
- **리밸런싱 추적**: 월간 포트폴리오 변화 모니터링

---

## 📈 성과 지표

대시보드는 다음과 같은 포괄적인 분석을 제공합니다:

| 지표 | 설명 |
|-----|------|
| **Total Return** | 선택 기간 동안의 누적 수익률 |
| **CAGR** | 연평균 복리 수익률 |
| **Volatility** | 연환산 표준편차 |
| **Sharpe Ratio** | 위험 조정 수익률 지표 |
| **Max Drawdown** | 최대 고점 대비 하락폭 |
| **Tracking Error** | 벤치마크 대비 액티브 리스크 |
| **Calmar Ratio** | CAGR / |Max Drawdown| |

---

## 🏗️ 프로젝트 구조

```
main/
│
├── streamlit_app.py      # 메인 Streamlit 애플리케이션
├── requirements.txt      # Python 의존성
├── README.md             # 본 파일
├── faq_document.md       # FAQ
│
├── temp/                 # cached
│   ├── original.py
│
├── Pre-trained result.xlsx      # Backtest 결과
└── Pre-trained result.html/     # Backtest Quantstat 결과

```

---

## 🔬 방법론 상세

### Walk-Forward 최적화

모든 전략 파라미터는 Look-ahead Bias를 방지하기 위해 Walk-forward 분석을 통해 최적화되었습니다:

1. **학습 윈도우**: 2년 롤링 윈도우
2. **테스트 윈도우**: 6개월 Out-of-sample 기간
3. **최적화 목표**: Sharpe Ratio 최대화, 제약조건:
   - Max Drawdown < 30%
   - 연간 회전율 < 500%
4. **파라미터 안정성**: 학습 윈도우의 80% 이상에서 일관된 파라미터 선택

### 리스크 관리

- **포지션 제한**: 종목당 최대 60% 배분
- **분산투자**: 리밸런싱 시 최소 2개 이상의 활성 포지션 유지
- **데이터 품질**: 상장폐지/합병된 종목 자동 처리
- **슬리피지 모델**: 보수적 체결 가정 (0.1% 슬리피지)

---

## 📊 예시 시나리오

### 시나리오 1: 정상적인 하락 구간

**상황**:
- TSLA: -40% (심각한 하락)
- NVDA: -30% (심각한 하락)
- AAPL: -10%, MSFT: -5%, GOOGL: -8%, AMZN: -12%, META: -6%

**계산**:
```
심각한 하락 그룹 (60% 배분):
- TSLA: 40/(40+30) × 60% = 34.3%
- NVDA: 30/(40+30) × 60% = 25.7%

일반 하락 그룹 (40% 배분):
- AAPL: 10/41 × 40% = 9.8%
- MSFT: 5/41 × 40% = 4.9%
- GOOGL: 8/41 × 40% = 7.8%
- AMZN: 12/41 × 40% = 11.7%
- META: 6/41 × 40% = 5.9%
```

**최종 포트폴리오**: `[34.3%, 25.7%, 9.8%, 4.9%, 7.8%, 11.7%, 5.9%]`

---

### 시나리오 2: 모든 종목 상승

**상황**: 모든 종목의 Drawdown = 0 (지속적 상승)

**결과**: 균등 가중 배분
```
각 종목 비중 = 1/7 ≈ 14.3%
```

---

### 시나리오 3: Cap Weight 초과

**초기 계산 비중**: `[75%, 15%, 7%, 3%]` (4종목)

**Cap Weight 적용 (60%)**:
```
1단계: 75% → 60% (초과분 15%)
2단계: 나머지 3종목에 15% 재분배
   - 15%: 15/(15+7+3) × 15% = 9% → 24%
   - 7%: 7/(15+7+3) × 15% = 4.2% → 11.2%
   - 3%: 3/(15+7+3) × 15% = 1.8% → 4.8%

최종 비중: [60%, 24%, 11.2%, 4.8%] (합=100%)
```

---

## 🎓 백테스트 결과 해석

### 성과 지표 가이드

1. **CAGR > Benchmark CAGR**: 장기 초과수익 달성
2. **Sharpe Ratio > 1.0**: 우수한 위험 조정 수익
3. **Max Drawdown < -30%**: 허용 가능한 리스크 수준
4. **Calmar Ratio > 0.5**: 낙폭 대비 양호한 수익

### 주의사항

⚠️ **백테스트의 한계**:
- 과거 성과가 미래 수익을 보장하지 않습니다
- 슬리피지, 거래비용, 세금이 실제 성과에 영향을 미칠 수 있습니다
- 시장 구조 변화로 전략 효과가 달라질 수 있습니다

⚠️ **데이터 품질**:
- Yahoo Finance 데이터는 조정주가(Adjusted Close) 기준입니다
- 배당금은 자동으로 재투자된 것으로 가정합니다
- 상장폐지/합병 종목은 자동으로 제외됩니다

---

## 🛠️ 커스터마이징

### 파라미터 조정

`app.py` 파일의 `OPTIMAL_PARAMS` 딕셔너리를 수정하여 전략 파라미터를 변경할 수 있습니다:

```python
OPTIMAL_PARAMS = {
    'lookback_months': 3,      # 1-12 범위 권장
    'lookback_days': 63,       # lookback_months × 21
    'rebalance_freq': 'M',     # 'W' (주간) 또는 'M' (월간)
    'threshold': -0.3,         # -0.2 ~ -0.5 범위 권장
    'weight_split': 0.60,      # 0.4 ~ 0.8 범위 권장
    'min_weight_change': 0.0,  # 0.0 ~ 0.1 범위
    'cap_weight': 0.60         # 0.3 ~ 0.8 범위 권장
}
```

### 자산 유니버스 변경

기본 M7 종목 대신 다른 자산군을 사용하려면:

```python
# 예시: 반도체 섹터
TICKERS = ['NVDA', 'AMD', 'INTC', 'TSM', 'AVGO', 'QCOM', 'MU']

# 예시: 글로벌 테크
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'BABA', 'TCEHY', 'TSM', 'ASML']
```

---

## 🔧 문제 해결

### 자주 발생하는 오류

**1. 데이터 다운로드 실패**
```
해결: 인터넷 연결 확인 및 티커 심볼 정확성 검증
```

**2. 종목이 시작일에 상장되지 않음**
```
해결: 시작일을 늦추거나 해당 종목 제거
```

**3. 메모리 부족 오류**
```
해결: 백테스트 기간 단축 또는 종목 수 감소
```

### 성능 최적화

- **캐싱 활용**: `@st.cache_data` 데코레이터는 데이터 재다운로드를 방지합니다
- **기간 제한**: 10년 이상의 백테스트는 처리 시간이 길어질 수 있습니다
- **종목 수 제한**: 20개 이하의 종목 권장

---

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

```
MIT License

Copyright (c) 2025 [Chris Park 3]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 기여 방법

프로젝트에 기여하고 싶으시다면:

1. 이 저장소를 Fork 합니다
2. Feature 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 Commit 합니다 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push 합니다 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성합니다

### 기여 가이드라인

- 코드는 PEP 8 스타일 가이드를 따라주세요
- 새로운 기능에는 주석과 docstring을 추가해주세요
- 주요 변경사항은 테스트를 포함해주세요

---

## 📚 추가 자료

### 참고 문헌

- **Mean Reversion Trading**: Evidence from the Stock Market (Jegadeesh, 1990)
- **Contrarian Investment Strategies**: The Next Generation (Dreman, 1998)
- **Quantitative Momentum**: A Practitioner's Guide (Wesley et al., 2016)

### 관련 프로젝트

- [Quantopian](https://github.com/quantopian) - 알고리즘 트레이딩 플랫폼
- [Backtrader](https://github.com/mementum/backtrader) - Python 백테스팅 프레임워크
- [QuantStats](https://github.com/ranaroussi/quantstats) - 포트폴리오 분석 도구

---

## 📧 문의 및 지원

### 개발자 연락처

질문, 버그 리포트, 기능 제안이 있으시면 언제든지 연락주세요:

**Made by CP3**

**📩 Email**: [parksuk1991@gmail.com](mailto:parksuk1991@gmail.com)

### 피드백

이 프로젝트가 도움이 되었다면:
- ⭐ GitHub에서 Star를 눌러주세요
- 🐛 버그를 발견하면 Issue를 등록해주세요
- 💡 개선 아이디어가 있으면 Discussion을 시작해주세요
- 📢 SNS에서 프로젝트를 공유해주세요

---

## 🙏 감사의 말

- **Yahoo Finance**: 무료 금융 데이터 제공
- **Streamlit**: 아름다운 대시보드 프레임워크
- **Plotly**: 인터랙티브 차트 라이브러리
- **오픈소스 커뮤니티**: 훌륭한 Python 생태계 구축

---

## 📌 업데이트 로그

### v1.0.0 (2024-11-12)
- ✨ 초기 릴리즈
- ✅ 기본 백테스팅 기능
- ✅ 인터랙티브 대시보드
- ✅ Cap Weight 제한 기능
- ✅ 초과수익 히트맵

### 향후 계획
- 🔄 실시간 포트폴리오 추적
- 📊 섹터별 분석 기능
- 🤖 자동 리밸런싱 알림
- 📱 모바일 대시보드

---

<div align="center">

**Made with ❤️ by Quantitative Researchers**

[⬆ Back to Top](#-m7-contrarian-strategy)

</div>
