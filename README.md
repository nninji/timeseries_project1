# 시계열 예측 자동화 웹앱

단변량 시계열 CSV 파일을 업로드하면 자동으로 데이터를 분석하고, 다양한 모델로 예측을 수행한 뒤, 여러 평가지표를 통해 적절성을 판단할 수 있는 인터랙티브 웹앱입니다.

## 핵심 기능 (풀 패키지)

### 데이터 처리
| 기능 | 설명 |
|---|---|
| 자동 분석 | CSV 인코딩, 구분자, 날짜·값 컬럼, 빈도, 계절성 자동 감지 |
| 임의 CSV 지원 | 단변량 시계열이면 모든 형식 OK |
| 이상치 탐지 및 처리 | IQR 기반 / 보간 또는 윈저화 |
| 변동점 탐지 | CUSUM 기반 자동 탐지 |
| 변환 | log, Box-Cox 자동 적용 |
| 다운샘플링 | 일→주/월/분기/연 자동 변환 |

### 예측 모델 (12종)
| 카테고리 | 모델 |
|---|---|
| Naive (4) | Last value, Mean, Seasonal Naive, Drift |
| 평활법 (5) | SES, Holt, Holt-Winters (가법/곱셈), Theta |
| ARIMA (3) | 수동 (p,d,q), Auto-ARIMA, sktime NaiveForecaster |
| 앙상블 | 단순 평균 + RMSE 역수 가중 평균 |

### 평가지표 (8종)
MAE · MSE · RMSE · MAPE · sMAPE · R² · MASE · Bias

### 통계 진단
| 영역 | 내용 |
|---|---|
| 정상성 검정 | ADF (Augmented Dickey-Fuller) |
| 자기상관 | ACF, PACF, Ljung-Box |
| 정규성 | Shapiro-Wilk, QQ plot |
| 분해 | 추세 / 계절성 / 잔차 |
| 차분 | d=0, 1, 2 시각화 (ARIMA d 결정) |

### 시각화 대시보드
- 예측 vs 실제 비교 차트
- 평가지표 막대 + 레이더 차트
- 잔차 4분할 (시계열/분포/산점도/누적합) + QQ plot
- 데이터 특성 게이지 (추세/계절/노이즈/자기상관)
- 95% 신뢰구간 음영
- 시계열 분해 4분할
- 백테스팅 시뮬레이션
- 롤링 교차검증
- 이상치/변동점 표시

### 고급 기능
- 자동 진단 메시지 (✅/⚠️/❌)
- 데이터 특성 기반 모델 자동 추천
- 신뢰구간 적중률(Coverage) 검증
- 시평 구간별 성능 (단기/중기/장기)
- 롤링 교차검증 + 안정성 진단
- 백테스팅 시뮬레이션
- 데이터 품질 점수 (A~F 등급)
- 평가지표 메타데이터 포함 CSV 다운로드

## 빠른 시작 (로컬 실행)

```bash
# 1) 이 폴더로 이동
cd ts_forecast_app

# 2) 가상환경 권장 (선택)
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# 3) 의존성 설치
pip install -r requirements.txt

# 4) 실행 (브라우저가 자동으로 열림)
streamlit run app.py
```

기본 주소: <http://localhost:8501>

## 온라인 배포 (Streamlit Community Cloud)

1. 이 폴더 전체를 본인의 GitHub 리포지토리에 푸시 (예: `ts-forecast-app`)
2. <https://share.streamlit.io> 에 GitHub로 로그인
3. **New app** → 리포지토리 선택 → 메인 파일은 `app.py`
4. **Deploy** 클릭 (1~3분이면 빌드 완료)
5. 받은 URL을 그대로 제출하면 됩니다 (예: `https://xxxx.streamlit.app`)

### 배포 시 주의 사항
- `requirements.txt`의 `pmdarima`는 빌드가 무거울 수 있습니다.
  Streamlit Cloud에서 빌드가 실패하면 `requirements.txt`에서 `pmdarima` 한 줄만 제거하세요.
  (Auto-ARIMA는 자동으로 statsmodels 폴백으로 동작합니다.)
- Python 버전은 3.10 또는 3.11 권장 (`runtime.txt`에 `python-3.11` 기재 가능).

## 사용 방법

1. 사이드바에서 **CSV 업로드** 또는 **샘플 데이터 사용** 체크
2. **시평(Horizon)** 입력 (예: 12)
3. 결측치 처리 방법 선택 (보통 "선형 보간" 권장)
4. (선택) 자동 감지가 잘못된 경우 컬럼명 수동 지정
5. 본문의 **모델 선택** 체크박스에서 비교할 모델 선택
6. **결과 확인**:
   - **1. 데이터 개요**: 자동 감지된 빈도/계절성/통계 요약
   - **2. 모델 선택**: 카테고리별로 그룹핑된 모델 체크박스
   - **3. 예측 결과**: 모델별 예측 곡선 + 실제값 비교
   - **4. 평가지표 대시보드**: 8개 지표 표 + 막대그래프 + 레이더 차트
   - **5. 잔차 분석**: 잔차 시계열 / 분포 / 산점도 / 누적합
   - **6. 미래 예측**: 전체 데이터로 재학습 후 미래 시평 예측 + CSV 다운로드

## 샘플 데이터

`sample_data/` 폴더에 5종의 시계열이 포함되어 있어 즉시 테스트할 수 있습니다:

| 파일 | 빈도 | 길이 | 특성 |
|---|---|---|---|
| `monthly_sales.csv` | 월별 | 60 | 추세 + 연간 계절성 |
| `daily_traffic.csv` | 일별 | 365 | 주간 계절성 |
| `hourly_temperature.csv` | 시간별 | 720 | 24시간 주기 |
| `quarterly_gdp.csv` | 분기별 | 40 | 약한 추세 |
| `simple_series.csv` | 날짜 없음 | 100 | 인덱스 자동 생성 |

## 프로젝트 구조

```
ts_forecast_app/
├── app.py                    # Streamlit 메인 앱
├── modules/
│   ├── data_handler.py       # CSV 자동 분석/전처리
│   ├── models.py             # 9개 예측 모델
│   ├── metrics.py            # 8개 평가지표
│   └── visualizations.py     # Plotly 차트 (6종)
├── sample_data/              # 샘플 시계열 5종
├── .streamlit/config.toml    # Streamlit 설정
├── requirements.txt          # 의존성
└── README.md                 # 이 문서
```

## 사용된 기술

- **[Streamlit](https://streamlit.io)** — 웹 UI 프레임워크
- **[pandas](https://pandas.pydata.org)** + **[numpy](https://numpy.org)** — 데이터 처리
- **[statsmodels](https://www.statsmodels.org)** — 평활법(SES, Holt, Holt-Winters), ARIMA
- **[pmdarima](https://github.com/alkaline-ml/pmdarima)** — Auto-ARIMA (선택; 폴백 지원)
- **[sktime](https://www.sktime.net)** — 시계열 통합 인터페이스 (강의 자료 기반)
- **[Plotly](https://plotly.com/python/)** — 인터랙티브 시각화


## 트러블슈팅

**Q. Streamlit Cloud 배포가 실패합니다.**
- 보통 `pmdarima` 빌드 실패. `requirements.txt`에서 `pmdarima` 줄을 삭제하세요.

**Q. "시평이 데이터 길이보다 큽니다" 에러**
- 검증 구간이 너무 길어 학습 데이터가 부족합니다. 시평을 줄이세요.

**Q. R² 값이 음수로 나옵니다.**
- 정상입니다. R²의 음수는 "단순 평균보다 못함"을 의미합니다. Naive 모델은 추세/계절성이 강한 데이터에서 자주 음수가 나옵니다. 더 적합한 모델(평활법/ARIMA)을 선택해보세요.

**Q. 한글 CSV가 깨져서 읽힙니다.**
- 자동으로 utf-8, cp949, euc-kr 등 다양한 인코딩을 시도합니다. 실패할 경우 CSV를 utf-8로 다시 저장해보세요.

