# M7 Portfolio Monitor - 배포 가이드 🚀

## 📁 파일 구조

배포 전에 다음과 같은 파일 구조를 확인하세요:

```
your-repo/
├── app.py                      # 메인 애플리케이션
├── requirements.txt            # Python 패키지 의존성
├── packages.txt               # 시스템 패키지 (비어있음)
├── README.md                  # 프로젝트 설명
├── .gitignore                 # Git 제외 파일
├── .streamlit/
│   └── config.toml           # Streamlit 설정
└── DEPLOYMENT_GUIDE.md       # 이 파일
```

## 🔧 GitHub 저장소 생성 및 업로드

### 1. GitHub에서 새 저장소 생성

1. GitHub에 로그인
2. 우측 상단 "+" → "New repository" 클릭
3. Repository name: `m7-portfolio-monitor` (원하는 이름)
4. Public 또는 Private 선택
5. "Create repository" 클릭

### 2. 로컬에서 Git 초기화 및 업로드

```bash
# 프로젝트 폴더로 이동
cd your-project-folder

# Git 초기화
git init

# 모든 파일 추가
git add .

# 첫 커밋
git commit -m "Initial commit: M7 Portfolio Monitor"

# GitHub 저장소 연결 (YOUR-USERNAME과 YOUR-REPO를 실제 값으로 변경)
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git

# 푸시
git branch -M main
git push -u origin main
```

## 🌐 Streamlit Cloud 배포

### 1. Streamlit Cloud 접속

1. [https://streamlit.io/cloud](https://streamlit.io/cloud) 접속
2. GitHub 계정으로 로그인

### 2. 앱 배포

1. **"New app"** 버튼 클릭
2. **Deploy an app** 선택
3. 다음 정보 입력:
   - **Repository**: `your-username/your-repo`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. **"Deploy!"** 클릭

### 3. 배포 확인

- 배포는 보통 2-5분 소요됩니다
- 배포 진행 상황은 실시간으로 표시됩니다
- 완료되면 자동으로 앱 URL이 생성됩니다 (예: `your-app-name.streamlit.app`)

## ⚙️ 환경 설정 (선택사항)

Streamlit Cloud에서 추가 설정이 필요한 경우:

1. 앱 대시보드에서 "Settings" 클릭
2. "Secrets" 탭에서 환경 변수 추가 (현재는 불필요)
3. "General" 탭에서 Python 버전 확인 (3.9+ 권장)

## 🔄 앱 업데이트

코드를 수정한 후 앱을 업데이트하려면:

```bash
# 변경사항 커밋
git add .
git commit -m "Update: 변경 내용 설명"

# GitHub에 푸시
git push origin main
```

**Streamlit Cloud가 자동으로 감지하여 재배포합니다!**

## 🐛 문제 해결

### 1. 배포 실패 시

**로그 확인**:
- Streamlit Cloud 대시보드에서 "Manage app" → "Logs" 확인
- requirements.txt의 패키지 버전 충돌 확인

**흔한 오류**:
```
ModuleNotFoundError: No module named 'xxx'
→ requirements.txt에 패키지 추가 및 재배포
```

### 2. 데이터 다운로드 오류

```python
# yfinance API 제한 문제
→ 캐시 시간(ttl) 조정: @st.cache_data(ttl=7200)
```

### 3. 메모리 부족

Streamlit Cloud 무료 플랜 제한:
- **RAM**: 1GB
- **CPU**: 공유

**해결책**:
- 데이터 기간 단축
- 캐싱 최적화
- 유료 플랜 고려

## 📊 성능 최적화 팁

### 1. 캐싱 활용

```python
@st.cache_data(ttl=3600)  # 1시간 캐싱
def download_data(tickers, start_date, end_date):
    # ...
```

### 2. 데이터 로딩 표시

```python
with st.spinner("데이터 로딩 중..."):
    # 시간이 걸리는 작업
```

### 3. 조건부 계산

```python
if run_button:  # 버튼 클릭시만 실행
    # 무거운 계산
```

## 🔒 보안 고려사항

### API 키 관리 (현재 불필요하지만 참고용)

Streamlit Secrets 사용:

1. `.streamlit/secrets.toml` 생성 (로컬)
```toml
API_KEY = "your-api-key"
```

2. 코드에서 사용
```python
import streamlit as st
api_key = st.secrets["API_KEY"]
```

3. Streamlit Cloud에서 설정
   - Settings → Secrets → TOML 형식으로 입력

## 📱 커스텀 도메인 연결 (Pro 플랜)

1. Streamlit Cloud Pro 플랜 구독
2. DNS 설정에서 CNAME 레코드 추가
3. Streamlit Cloud에서 도메인 연결

## 📈 모니터링

### 앱 사용량 확인

- Streamlit Cloud 대시보드에서 "Analytics" 확인
- 방문자 수, 세션 시간 등 추적

### 오류 모니터링

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # 코드
except Exception as e:
    logger.error(f"Error: {e}")
    st.error(f"오류 발생: {e}")
```

## 🎨 UI 커스터마이징

### config.toml에서 테마 변경

```toml
[theme]
primaryColor = "#FF4B4B"  # 메인 색상
backgroundColor = "#0E1117"  # 배경색 (다크모드)
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

## 📞 지원

- **Streamlit 문서**: [docs.streamlit.io](https://docs.streamlit.io)
- **커뮤니티**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: 프로젝트 저장소의 Issues 탭

## ✅ 배포 체크리스트

- [ ] GitHub 저장소 생성 및 코드 업로드
- [ ] requirements.txt 확인
- [ ] .gitignore 설정
- [ ] README.md 작성
- [ ] Streamlit Cloud 계정 생성
- [ ] 앱 배포
- [ ] 테스트 (다양한 날짜 범위, 벤치마크)
- [ ] 에러 로그 확인
- [ ] 성능 모니터링
- [ ] 문서 업데이트

---

**축하합니다! 🎉 이제 M7 Portfolio Monitor가 온라인에서 실행됩니다!**