# M7 Portfolio Monitor - 빠른 시작 가이드 ⚡

## 🚀 5분 안에 배포하기

### 단계 1: 파일 준비 (1분)

다음 파일들을 프로젝트 폴더에 생성하세요:

```
m7-portfolio-monitor/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

### 단계 2: GitHub 업로드 (2분)

```bash
# 터미널에서 실행
cd m7-portfolio-monitor
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR-USERNAME/m7-portfolio-monitor.git
git push -u origin main
```

### 단계 3: Streamlit Cloud 배포 (2분)

1. [streamlit.io/cloud](https://streamlit.io/cloud) 접속
2. GitHub 로그인
3. "New app" 클릭
4. 저장소 선택, `app.py` 입력
5. "Deploy!" 클릭

**완료! 🎉**

---

## 💻 로컬에서 먼저 테스트하기

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 앱 실행
streamlit run app.py

# 3. 브라우저에서 열기
# 자동으로 http://localhost:8501 열림
```

---

## 📋 필수 체크리스트

배포 전 확인사항:

- [ ] `requirements.txt`에 모든 패키지 포함
- [ ] `app.py`가 에러 없이 실행됨
- [ ] `.gitignore`로 불필요한 파일 제외
- [ ] GitHub 저장소가 Public으로 설정됨
- [ ] Streamlit Cloud 계정이 준비됨

---

## 🔧 첫 실행 시 해야 할 것

1. **기본 테스트**
   - 시작일: 2023-01-01
   - 종료일: 2024-12-31
   - 벤치마크: M7 Equal Weight
   - "분석 실행" 클릭

2. **차트 확인**
   - 누적 수익률 차트 로딩 확인
   - Drawdown 차트 표시 확인
   - 최신 리밸런싱 정보 확인

3. **데이터 다운로드 테스트**
   - CSV 다운로드 버튼 클릭
   - 파일이 정상적으로 다운로드되는지 확인

---

## 🐛 흔한 오류 해결

### 오류 1: "No module named 'yfinance'"

**해결**:
```bash
pip install yfinance
# 또는
pip install -r requirements.txt
```

### 오류 2: Streamlit Cloud 배포 실패

**해결**:
1. Streamlit Cloud 로그 확인
2. `requirements.txt` 패키지 버전 확인
3. Python 버전 확인 (3.9+ 필요)

### 오류 3: 데이터 다운로드 느림

**정상입니다!**
- yfinance API는 초기 로딩 시 시간이 걸립니다
- 캐싱 덕분에 두 번째부터는 빠릅니다

---

## 📱 사용 팁

### 팁 1: 날짜 범위 선택

```
단기 분석 (1-2년): 최근 트렌드 파악
장기 분석 (5년+): 전략 안정성 확인
```

### 팁 2: 벤치마크 비교

```
M7 Equal Weight: 동적 리밸런싱 효과 측정
QQQ: 시장 대비 성과 비교
```

### 팁 3: 리밸런싱 모니터링

```
매월 말 확인하여 실제 포트폴리오 조정에 활용
(단, 거래 비용 고려 필수!)
```

---

## 🎯 다음 단계

배포 완료 후:

1. **URL 공유**
   - 생성된 Streamlit 앱 URL 저장
   - 친구나 동료와 공유

2. **정기 모니터링**
   - 매월 말 리밸런싱 정보 확인
   - 성과 지표 추적

3. **커스터마이징**
   - 색상 테마 변경 (`.streamlit/config.toml`)
   - 추가 차트나 지표 구현

---

## 📞 도움이 필요하신가요?

- **문서**: [README.md](README.md)
- **상세 가이드**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **사용 예시**: [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
- **GitHub Issues**: 프로젝트 저장소

---

**시작하셨다면, 즐거운 분석 되세요! 📊✨**