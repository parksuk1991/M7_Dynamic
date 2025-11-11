# M7 Portfolio Monitor - 프로젝트 구조 📁

## 전체 파일 구조

```
m7-portfolio-monitor/
│
├── 📄 app.py                          # 메인 Streamlit 애플리케이션
├── 📄 requirements.txt                # Python 패키지 의존성
├── 📄 packages.txt                    # 시스템 패키지 (비어있음)
├── 📄 README.md                       # 프로젝트 메인 문서
├── 📄 .gitignore                      # Git 제외 파일 목록
│
├── 📁 .streamlit/
│   └── 📄 config.toml                 # Streamlit 앱 설정
│
├── 📁 .github/                        # (선택사항)
│   └── 📁 workflows/
│       └── 📄 update_data.yml         # 자동 데이터 업데이트 워크플로우
│
└── 📁 docs/                           # 문서 폴더 (선택사항)
    ├── 📄 DEPLOYMENT_GUIDE.md         # 배포 가이드
    ├── 📄 USAGE_EXAMPLES.md           # 사용 예시
    ├── 📄 QUICK_START.md              # 빠른 시작 가이드
    └── 📄 PROJECT_STRUCTURE.md        # 이 파일
```

---

## 📄 파일별 상세 설명

### 1. `app.py` (핵심 파일)

**역할**: Streamlit 웹 애플리케이션의 메인 코드

**주요 기능**:
- 사용자 인터페이스 구성
- 데이터 다운로드 및 처리
- 포트폴리오 백테스팅
- 차트 및 지표 시각화
- CSV 다운로드 기능

**코드 구조**:
```python
1. 임포트 및 설정
2. 상수 정의 (M7_TICKERS, OPTIMAL_PARAMS)
3. 헬퍼 함수들
   - download_data()
   - calculate_drawdown_from_peak()
   - calculate_weights_by_drawdown()
   - backtest_strategy()
   - calculate_performance_metrics()
   - calculate_turnover()
4. main() 함수
   - 사이드바 UI
   - 백테스팅 실행
   - 결과 시각화
```

**주요 라이브러리**:
- `streamlit`: 웹 UI
- `yfinance`: 주가 데이터
- `pandas/numpy`: 데이터 처리
- `plotly`: 인터랙티브 차트

---

### 2. `requirements.txt`

**역할**: Python 패키지 의존성 명시

**내용**:
```
streamlit==1.28.0      # 웹 프레임워크
yfinance==0.2.32       # 주가 데이터 API
pandas==2.1.3          # 데이터 프레임
numpy==1.26.2          # 수치 계산
plotly==5.18.0         # 차트 라이브러리
scikit-learn==1.3.2    # 유틸리티 함수
```

**버전 관리**:
- 안정적인 버전으로 고정
- 주기적으로 업데이트 권장
- 호환성 테스트 필수

---

### 3. `.streamlit/config.toml`

**역할**: Streamlit 앱 테마 및 설정

**주요 설정**:
```toml
[theme]
primaryColor = "#1f77b4"    # 메인 색상 (파란색)
backgroundColor = "#ffffff"  # 배경색 (흰색)
textColor = "#262730"       # 텍스트 색상

[server]
headless = true            # 헤드리스 모드
port = 8501               # 포트 번호
```

**커스터마이징**:
- 색상 변경 가능
- 다크 모드 설정 가능
- 폰트 변경 가능

---

### 4. `.gitignore`

**역할**: Git에서 제외할 파일 지정

**주요 항목**:
```
__pycache__/          # Python 캐시
*.pkl                 # 피클 파일
*.csv                 # 데이터 파일
.streamlit/           # 로컬 Streamlit 설정
.DS_Store            # macOS 파일
```

---

### 5. `README.md`

**역할**: 프로젝트 소개 및 메인 문서

**포함 내용**:
- 프로젝트 개요
- 주요 기능
- 배포 방법
- 사용 방법
- 라이선스 및 면책조항

---

### 6. `.github/workflows/update_data.yml` (선택사항)

**역할**: GitHub Actions 자동화 워크플로우

**기능**:
- 매일 장 마감 후 데이터 업데이트
- 수동 트리거 가능

**참고**: Streamlit Cloud는 요청 시마다 최신 데이터를 가져오므로 선택사항입니다.

---

## 🔄 데이터 흐름

```
1. 사용자 입력
   ↓
2. yfinance API로 주가 데이터 다운로드
   ↓
3. 백테스팅 엔진 실행
   - 고점 대비 하락률 계산
   - 가중치 결정
   - 포트폴리오 시뮬레이션
   ↓
4. 성과 지표 계산
   - CAGR, Sharpe, MDD 등
   ↓
5. 시각화
   - Plotly 차트 생성
   - 테이블 표시
   ↓
6. 결과 제공
   - 화면 표시
   - CSV 다운로드
```

---

## 🎨 UI 구조

```
┌─────────────────────────────────────────────┐
│  📊 M7 Dynamic Portfolio Monitor            │
├─────────────────────────────────────────────┤
│                                             │
│  [사이드바]           [메인 콘텐츠]          │
│  ┌─────────┐         ┌────────────────┐    │
│  │ ⚙️ 설정  │         │ 📊 성과 요약    │    │
│  │         │         │ ┌────┬────────┐│    │
│  │ 📅 기간  │         │ │CAGR│Sharpe ││    │
│  │         │         │ └────┴────────┘│    │
│  │ 📈 벤치  │         │                │    │
│  │         │         │ 📈 누적 수익률  │    │
│  │ 🎯 파라  │         │ [차트]         │    │
│  │         │         │                │    │
│  │ 🚀 실행  │         │ 📉 Drawdown    │    │
│  └─────────┘         │ [차트]         │    │
│                      │                │    │
│                      │ 🎯 최신 리밸런싱│    │
│                      │ [파이차트]     │    │
│                      │                │    │
│                      │ 📊 비중 변화   │    │
│                      │ [Area 차트]    │    │
│                      │                │    │
│                      │ 💾 다운로드    │    │
│                      └────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## 📦 배포 환경

### Streamlit Cloud 사양

```
리소스:
- RAM: 1GB (무료 플랜)
- CPU: 공유
- 스토리지: 제한적

제약사항:
- 세션 타임아웃: 30분
- 동시 사용자: 제한적
- 실행 시간: 최대 10분
```

### 최적화 전략

```python
# 1. 캐싱
@st.cache_data(ttl=3600)
def download_data(...):
    # 1시간 캐싱

# 2. 지연 로딩
if run_button:  # 버튼 클릭 시만 실행
    # 무거운 계산

# 3. 데이터 압축
# 불필요한 컬럼 제거
# 날짜 범위 제한
```

---

## 🔐 보안 고려사항

### 현재 구현

```
✅ API 키 불필요 (yfinance 무료 API)
✅ 사용자 데이터 저장 안 함
✅ HTTPS 자동 제공 (Streamlit Cloud)
```

### 향후 추가 시 고려사항

```
- API 키 관리: Streamlit Secrets 사용
- 사용자 인증: OAuth 통합
- 데이터 암호화: 민감 정보 보호
```

---

## 📈 확장 가능성

### 쉬운 추가 기능

1. **더 많은 벤치마크**
```python
BENCHMARKS = {
    'SPY': 'S&P 500',
    'IWM': 'Russell 2000',
    'EFA': 'EAFE'
}
```

2. **커스텀 파라미터**
```python
lookback = st.slider("Lookback (months)", 1, 12, 3)
```

3. **알림 기능**
```python
if mdd < -20:
    st.warning("⚠️ 큰 하락이 발생했습니다!")
```

### 고급 기능 (별도 개발 필요)

- 실시간 포트폴리오 추적
- 이메일/텔레그램 알림
- 백테스팅 파라미터 최적화 UI
- 복수 전략 비교

---

## 🧪 테스트 가이드

### 로컬 테스트

```bash
# 1. 환경 설정
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 앱 실행
streamlit run app.py

# 3. 테스트 시나리오
- 다양한 날짜 범위 테스트
- 벤치마크 전환 테스트
- CSV 다운로드 테스트
- 에러 처리 확인
```

### 배포 전 체크리스트

```
□ 로컬에서 에러 없이 실행
□ 다양한 날짜 범위 테스트
□ 모든 차트 정상 표시
□ CSV 다운로드 작동
□ README.md 업데이트
□ .gitignore 확인
□ requirements.txt 검증
```

---

## 📞 유지보수

### 정기 업데이트

```
월간:
- yfinance API 상태 확인
- 성과 지표 검증
- 사용자 피드백 수집

분기:
- 패키지 버전 업데이트
- 보안 패치 적용
- 기능 개선

연간:
- 전략 파라미터 재검토
- 코드 리팩토링
- 문서 업데이트
```

### 문제 해결 프로세스

```
1. 로그 확인
   → Streamlit Cloud 대시보드

2. 로컬 재현
   → 동일 환경에서 테스트

3. 디버깅
   → st.write()로 중간 값 확인

4. 수정 및 배포
   → GitHub 푸시 → 자동 재배포
```

---

**이 구조를 기반으로 프로젝트를 확장하고 커스터마이징하세요! 🚀**