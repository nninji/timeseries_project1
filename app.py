"""
시계열 예측 자동화 웹앱 
========================================
- 단변량 시계열 CSV를 업로드하면 자동으로 분석/예측을 수행합니다.
- 시평(horizon), 모델 선택, 결측치 처리 방법은 사이드바에서 변경 가능합니다.
- 파일이 바뀌거나 시평이 바뀌면 예측이 새로 계산됩니다.
- 다양한 평가지표(MAE, RMSE, MAPE, sMAPE, R², MASE, Bias)와 잔차 분석 차트를 제공합니다.

실행 방법:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# 모듈 경로
sys.path.insert(0, str(Path(__file__).parent))
from modules.data_handler import (
    read_csv_robust, auto_process, train_test_split_ts,
)
from modules.metrics import all_metrics, metric_descriptions
from modules.models import (
    MODEL_REGISTRY, get_default_models, run_forecast,
    forecast_with_ci, rolling_cv,
    forecast_arima_manual,
    ensemble_simple_mean, ensemble_inverse_rmse,
    evaluate_multi_horizon,
)
from modules.visualizations import (
    plot_raw_series, plot_train_test, plot_forecasts,
    plot_metric_comparison, plot_metrics_radar, plot_residuals,
    plot_future_forecast, plot_future_forecast_with_ci,
    plot_decomposition, plot_acf_pacf, plot_rolling_cv,
    plot_with_outliers, plot_change_points, plot_qq, plot_differencing,
    plot_multi_horizon, plot_feature_gauges, plot_backtest,
)
from modules.diagnostics import (
    adf_test, compute_acf_pacf, decompose, ljung_box_test,
    qq_data, prediction_interval_coverage, differencing_series,
    shapiro_test,
)
from modules.preprocessing import (
    handle_outliers, detect_outliers_iqr,
    detect_change_points, resample_series,
    apply_log_transform, inverse_log_transform,
    apply_boxcox_transform, inverse_boxcox_transform,
    compute_features, data_quality_score, recommend_models,
)


# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="시계열 예측 자동화 웹앱",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1f77b4;
    }
    .small-note { font-size: 0.85rem; color: #666; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 헤더
# =============================================================================
st.title("📈 시계열 예측 자동화 웹앱")
st.caption("단변량 시계열 CSV를 업로드하면 자동으로 분석 → 예측 → 평가까지 수행합니다.")


# =============================================================================
# 사이드바 — 입력 영역
# =============================================================================
with st.sidebar:
    st.header("⚙️ 설정")

    # 1) 파일 업로드
    st.subheader("1. 데이터 입력")
    uploaded = st.file_uploader(
        "단변량 시계열 CSV 파일",
        type=["csv", "tsv", "txt"],
        help="날짜 컬럼과 값 컬럼이 있는 CSV. 날짜가 없어도 동작합니다.",
    )
    use_sample = st.checkbox(
        "샘플 데이터 사용",
        value=(uploaded is None),
        help="업로드된 파일이 없으면 자동으로 샘플 데이터를 사용합니다.",
    )

    # 2) 시평
    st.subheader("2. 예측 시평 (Horizon)")
    horizon = st.number_input(
        "예측 시평 (몇 시점을 예측할지)",
        min_value=1, max_value=365, value=12, step=1,
        help="시평이 길수록 검증 구간이 길어지고, 미래 예측 기간도 같은 길이로 산출됩니다.",
    )

    # 3) 결측치 처리
    st.subheader("3. 전처리 옵션")
    fill_method = st.selectbox(
        "결측치 처리 방법",
        ["interpolate", "ffill", "mean", "drop"],
        index=0,
        format_func=lambda x: {
            "interpolate": "선형 보간 (권장)",
            "ffill": "직전값으로 채우기",
            "mean": "평균값으로 채우기",
            "drop": "결측 행 제거",
        }[x],
    )
    regularize = st.checkbox(
        "균일한 시점 간격으로 보정",
        value=True,
        help="감지된 빈도(freq)로 인덱스를 재샘플링하고 빈 값을 채웁니다.",
    )

    # 3-2) 이상치 처리
    st.markdown("**이상치 처리**")
    outlier_method = st.selectbox(
        "이상치 처리",
        ["keep", "interpolate", "winsorize"],
        index=0,
        format_func=lambda x: {
            "keep": "그대로 두기 (탐지만)",
            "interpolate": "보간으로 대체",
            "winsorize": "윈저화 (5/95% 클리핑)",
        }[x],
        help="IQR×1.5 기준 이상치를 처리합니다.",
    )

    # 3-3) 변환
    st.markdown("**값 변환 (선택)**")
    transform_method = st.selectbox(
        "변환",
        ["none", "log", "boxcox"],
        index=0,
        format_func=lambda x: {
            "none": "변환 없음",
            "log": "로그 변환 (지수적 성장 데이터)",
            "boxcox": "Box-Cox 변환 (자동 λ)",
        }[x],
    )

    # 3-4) 다운샘플링
    st.markdown("**다운샘플링 (선택)**")
    resample_target = st.selectbox(
        "재샘플링 빈도",
        ["none", "W", "MS", "QS", "YS"],
        index=0,
        format_func=lambda x: {
            "none": "원본 유지",
            "W": "주별로 변환",
            "MS": "월별로 변환",
            "QS": "분기별로 변환",
            "YS": "연도별로 변환",
        }[x],
        help="원본보다 큰 빈도로만 변환할 수 있습니다 (예: 일별 → 월별).",
    )

    # 4) 모델 선택은 데이터 분석 후로 미룸 (계절성 알아야 함)
    st.subheader("4. 컬럼 선택 (선택)")
    st.caption("자동 감지가 잘못되었다면 수동으로 지정하세요.")
    date_col_override_input = st.text_input("날짜 컬럼명 (자동 감지: 비워두기)", value="")
    value_col_override_input = st.text_input("값 컬럼명 (자동 감지: 비워두기)", value="")
    date_col_override = date_col_override_input.strip() or None
    value_col_override = value_col_override_input.strip() or None

    # 5) 고급 분석 옵션
    st.subheader("5. 고급 분석")
    enable_decomposition = st.checkbox(
        "시계열 분해 표시", value=True,
        help="추세/계절성/잔차로 분해 (계절성이 감지된 경우만)",
    )
    enable_diagnostics = st.checkbox(
        "통계 진단 (ADF, ACF) 표시", value=True,
        help="정상성 검정과 자기상관 함수를 표시합니다.",
    )
    enable_ci = st.checkbox(
        "예측 신뢰구간 표시", value=True,
        help="미래 예측에 95% 신뢰구간을 함께 표시합니다.",
    )
    enable_cv = st.checkbox(
        "롤링 교차검증 실행", value=False,
        help="여러 시점에서 학습/검증을 반복하여 신뢰성을 확인합니다 (시간 소요).",
    )
    if enable_cv:
        cv_splits = st.slider("CV fold 수", min_value=2, max_value=8, value=4)
    else:
        cv_splits = 4


# =============================================================================
# 데이터 로딩
# =============================================================================
@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    """샘플: 추세+계절성+노이즈가 있는 36개월 매출 데이터."""
    rng = np.random.default_rng(42)
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    trend = np.linspace(100, 180, n)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 4, n)
    values = trend + seasonal + noise
    return pd.DataFrame({"date": idx, "sales": values.round(2)})


@st.cache_data(show_spinner=False)
def load_csv_cached(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """파일 내용 + 이름을 키로 캐싱."""
    return read_csv_robust(io.BytesIO(file_bytes))


def get_dataframe():
    if uploaded is not None:
        try:
            file_bytes = uploaded.getvalue()
            return load_csv_cached(file_bytes, uploaded.name), uploaded.name
        except Exception as e:
            st.error(f"파일 로드 실패: {e}")
            return None, None
    elif use_sample:
        return load_sample(), "sample_monthly_sales.csv"
    else:
        return None, None


df, file_label = get_dataframe()

if df is None:
    st.info("👈 사이드바에서 CSV 파일을 업로드하거나 '샘플 데이터 사용'을 켜주세요.")
    st.stop()


# =============================================================================
# 자동 전처리
# =============================================================================
try:
    proc = auto_process(
        df,
        date_col_override=date_col_override,
        value_col_override=value_col_override,
        fill_method=fill_method,
        regularize=regularize,
    )
except Exception as e:
    st.error(f"❌ **데이터 처리 실패**: {e}")
    st.markdown("""
    ### 💡 해결 방법
    - **CSV의 첫 줄이 컬럼명인지** 확인해주세요.
    - **수치 데이터가 있는 컬럼**이 최소 1개는 있어야 합니다.
    - 컬럼명에 한글이 깨져서 들어가 있다면 **utf-8** 인코딩으로 다시 저장해보세요.
    - 사이드바 "4. 컬럼 선택"에서 **수동으로 날짜/값 컬럼명을 지정**해보세요.
    """)
    st.write("업로드된 데이터 미리보기:")
    st.dataframe(df.head(10), use_container_width=True)
    st.stop()

ts: pd.Series = proc["ts"]
freq = proc["freq"]
seasonal_period = proc["seasonal_period"]
summary = proc["summary"]

# 추가 전처리 단계들

# (1) 다운샘플링 (재샘플링 우선, 빈도 자동 재추정 위해)
resample_applied = False
if resample_target != "none" and isinstance(ts.index, pd.DatetimeIndex):
    try:
        ts_new = resample_series(ts, resample_target, agg="mean")
        if len(ts_new) >= 8:  # 너무 짧아지면 무시
            ts = ts_new
            resample_applied = True
            freq = resample_target
            # 계절성 재추정 (간단히: 월별이면 12, 분기면 4, 연도면 1)
            seasonal_period = {"W": 52, "MS": 12, "QS": 4, "YS": 1}.get(resample_target, seasonal_period)
        else:
            st.warning(f"⚠️ 재샘플링 결과가 너무 짧아 적용하지 않았습니다 ({len(ts_new)}개).")
    except Exception as e:
        st.warning(f"⚠️ 재샘플링 실패: {e}")

# (2) 이상치 처리
ts_pre_outlier = ts.copy()
outlier_mask = detect_outliers_iqr(ts)
n_outliers = int(outlier_mask.sum())
if outlier_method != "keep" and n_outliers > 0:
    ts, _ = handle_outliers(ts, method=outlier_method, detection="iqr")

# (3) 변환
transform_info = None
if transform_method == "log":
    try:
        ts, transform_info = apply_log_transform(ts)
    except Exception as e:
        st.warning(f"⚠️ 로그 변환 실패: {e}")
elif transform_method == "boxcox":
    try:
        ts, transform_info = apply_boxcox_transform(ts)
    except Exception as e:
        st.warning(f"⚠️ Box-Cox 변환 실패: {e}")

# 변동점 탐지 (변환 후 시계열에서)
change_points = detect_change_points(ts_pre_outlier, n_points=3)

# 시계열 특성 (변환 전 시계열로 계산해야 의미 있음)
features = compute_features(ts_pre_outlier, seasonal_period)

# 데이터 품질 점수
quality = data_quality_score(
    ts_pre_outlier, features,
    n_outliers=n_outliers,
    n_missing=summary["n_missing_before"],
)


# =============================================================================
# 데이터 개요 섹션
# =============================================================================
st.header("1. 데이터 개요")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("관측치 수", f"{summary['n_observations']:,}")
c2.metric("빈도(freq)", freq if freq else "—")
c3.metric("계절 주기(m)", seasonal_period if seasonal_period else "—")
c4.metric("결측 (보정 전)", f"{summary['n_missing_before']:,}")
c5.metric("평균값", f"{summary['mean']:,.2f}")

st.caption(
    f"📁 **파일**: `{file_label}`  |  "
    f"📅 **기간**: `{summary['start']} ~ {summary['end']}`  |  "
    f"📌 **날짜 컬럼**: `{proc['date_col'] or '없음(인덱스)'}`  |  "
    f"📊 **값 컬럼**: `{proc['value_col']}`"
)

with st.expander("🔍 원시 데이터 미리보기 (상위 10행)"):
    st.dataframe(df.head(10), use_container_width=True)

st.plotly_chart(plot_raw_series(ts, title=f"시계열: {proc['value_col']}"), use_container_width=True)


# =============================================================================
# 1-1. 데이터 품질 점수 + 특성 게이지
# =============================================================================
st.subheader("📋 데이터 품질 평가")
qcol1, qcol2 = st.columns([1, 3])
with qcol1:
    score = quality["score"]
    grade = quality["grade"]
    grade_color = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "F": "🔴"}.get(grade, "⚪")
    st.metric(f"품질 점수 ({grade_color} 등급 {grade})", f"{score:.1f} / 100")
    if grade in ("A", "B"):
        st.success(quality["interpretation"])
    elif grade == "C":
        st.warning(quality["interpretation"])
    else:
        st.error(quality["interpretation"])
with qcol2:
    st.markdown("**시계열 특성 정량화**")
    st.plotly_chart(plot_feature_gauges(features), use_container_width=True)
    feat_summary = (
        f"📊 추세 강도 **{features.get('trend_strength', 0):.2f}** · "
        f"계절성 강도 **{features.get('seasonal_strength', 0):.2f}** · "
        f"노이즈 **{features.get('noise_ratio', 0):.2f}** · "
        f"자기상관 **{features.get('autocorr_lag1', 0):+.2f}**"
    )
    st.caption(feat_summary)

# 1-1-2) 이상치 + 변동점 시각화
viz_c1, viz_c2 = st.columns(2)
with viz_c1:
    st.markdown(f"**🚨 이상치 탐지** ({n_outliers}개 발견, 처리: `{outlier_method}`)")
    st.plotly_chart(plot_with_outliers(ts_pre_outlier, outlier_mask), use_container_width=True)
with viz_c2:
    st.markdown(f"**📍 변동점 탐지** ({len(change_points)}개)")
    st.plotly_chart(plot_change_points(ts_pre_outlier, change_points), use_container_width=True)

# 변환 적용 정보
if transform_info is not None:
    method = transform_info.get("method", "")
    if method == "boxcox":
        st.info(f"🔄 **Box-Cox 변환 적용** (λ={transform_info.get('lambda', 0):.3f}) — 모든 분석은 변환된 값 기준입니다.")
    elif method == "log":
        st.info(f"🔄 **로그 변환 적용** — 모든 분석은 log(x) 기준입니다.")
if resample_applied:
    st.info(f"📅 **다운샘플링 적용**: 빈도가 `{resample_target}`로 변환됨")


# =============================================================================
# 1-2. 시계열 분해 (Decomposition)
# =============================================================================
if enable_decomposition:
    st.subheader("📐 시계열 분해 (추세 / 계절성 / 잔차)")
    if seasonal_period and seasonal_period > 1:
        decomp = decompose(ts, seasonal_period, model="additive")
        if decomp is not None:
            st.plotly_chart(plot_decomposition(decomp), use_container_width=True)
            with st.expander("💡 시계열 분해 해석"):
                st.markdown(f"""
- **추세(Trend)**: 데이터의 장기적인 방향성. 우상향이면 성장, 우하향이면 감소.
- **계절성(Seasonal)**: 계절 주기 m={seasonal_period}로 반복되는 패턴.
- **잔차(Residual)**: 추세와 계절성을 빼고 남은 무작위 변동. 0 주변에서 들쭉날쭉하면 정상.
                """)
        else:
            st.info(f"데이터 길이가 너무 짧아 분해 불가 (최소 {2 * seasonal_period}개 필요).")
    else:
        st.info("계절성이 감지되지 않아 분해를 건너뜁니다.")


# =============================================================================
# 1-3. 통계 진단 (ADF + ACF/PACF)
# =============================================================================
if enable_diagnostics:
    st.subheader("🔬 통계 진단")
    diag_c1, diag_c2 = st.columns([1, 2])
    with diag_c1:
        st.markdown("**ADF 정상성 검정**")
        adf_result = adf_test(ts)
        st.metric(
            "검정 통계량",
            f"{adf_result['statistic']:.4f}" if not pd.isna(adf_result["statistic"]) else "—",
        )
        st.metric(
            "p-value",
            f"{adf_result['p_value']:.4f}" if not pd.isna(adf_result["p_value"]) else "—",
        )
        st.markdown(adf_result["interpretation"])
    with diag_c2:
        st.markdown("**자기상관 함수 (ACF / PACF)**")
        acf_v, pacf_v, conf_band = compute_acf_pacf(ts, nlags=min(40, len(ts) // 2 - 1))
        if len(acf_v) > 0:
            st.plotly_chart(
                plot_acf_pacf(acf_v, pacf_v, conf_band),
                use_container_width=True,
            )
            with st.expander("💡 ACF/PACF 해석"):
                st.markdown("""
- **빨간 점선** 바깥의 막대 = 통계적으로 유의한 자기상관.
- **ACF에서 점진적 감소** + **PACF에서 특정 시차 후 절단**: AR 과정 시사.
- **ACF에서 계절 주기마다 큰 값** 반복: 계절성 존재.
- 정상화(차분)한 후에도 패턴이 남으면 ARIMA의 (p, q) 결정에 사용.
                """)
        else:
            st.info("ACF/PACF 계산 불가 (데이터 부족).")


# =============================================================================
# 1-4. 차분 시각화 (ARIMA 강의 연계)
# =============================================================================
if enable_diagnostics:
    with st.expander("🔢 차분(Differencing) 시각화 — ARIMA의 d 결정", expanded=False):
        st.markdown(
            "ARIMA에서 d=1, 2 차분 결과를 보여줍니다. "
            "**평균이 0 근처에서 일정한 변동**을 보이는 차분이 적절한 d입니다."
        )
        diffs = differencing_series(ts, max_d=2)
        st.plotly_chart(plot_differencing(diffs), use_container_width=True)


# =============================================================================
# 모델 선택 (데이터 분석 후) + 자동 추천
# =============================================================================
st.header("2. 모델 선택 및 학습")

# 자동 추천
recommended_keys, recommend_reasons = recommend_models(features, seasonal_period, len(ts))
recommended_keys = [k for k in recommended_keys if k in MODEL_REGISTRY]

with st.expander("💡 데이터 특성 기반 모델 추천", expanded=True):
    st.markdown("**이 데이터에 적합한 모델 (자동 추천):**")
    for k, why in zip(recommended_keys, recommend_reasons):
        if k in MODEL_REGISTRY:
            st.markdown(f"- ✅ **{MODEL_REGISTRY[k].name}** — {why}")

default_models = recommended_keys if recommended_keys else get_default_models(seasonal_period)
all_model_keys = list(MODEL_REGISTRY.keys())

# 카테고리별로 그룹핑하여 보여주기
with st.expander("🤖 사용할 모델 선택", expanded=True):
    cat_to_keys = {}
    for k, info in MODEL_REGISTRY.items():
        cat_to_keys.setdefault(info.category, []).append(k)

    selected_models: list[str] = []
    cols = st.columns(len(cat_to_keys))
    for i, (cat, keys) in enumerate(cat_to_keys.items()):
        with cols[i]:
            st.markdown(f"**{cat}**")
            for k in keys:
                info = MODEL_REGISTRY[k]
                checked = st.checkbox(
                    info.name,
                    value=(k in default_models),
                    key=f"model_{k}",
                    help=info.description,
                )
                if checked:
                    selected_models.append(k)

    if not selected_models:
        st.warning("⚠️ 최소 하나 이상의 모델을 선택해주세요.")
        st.stop()

# 2-2) ARIMA 수동 차수 (선택)
with st.expander("⚙️ ARIMA 수동 차수 직접 설정 (선택)", expanded=False):
    st.markdown(
        "위의 ACF/PACF와 차분 결과를 보고 직접 (p, d, q)를 선택해보세요. "
        "ARIMA(0,0,0)은 백색잡음, (0,1,0)은 random walk, (1,0,0)은 AR(1)을 의미합니다."
    )
    use_manual_arima = st.checkbox("ARIMA 수동 차수 추가", value=False)
    if use_manual_arima:
        arima_c1, arima_c2, arima_c3 = st.columns(3)
        with arima_c1:
            arima_p = st.number_input("p (AR 차수)", min_value=0, max_value=5, value=1)
        with arima_c2:
            arima_d = st.number_input("d (차분 차수)", min_value=0, max_value=2, value=1)
        with arima_c3:
            arima_q = st.number_input("q (MA 차수)", min_value=0, max_value=5, value=1)
    else:
        arima_p = arima_d = arima_q = 1

# 2-3) 앙상블 옵션
with st.expander("🎯 앙상블 모델 (여러 모델 평균)", expanded=False):
    st.markdown(
        "여러 모델의 예측을 합쳐서 최종 예측을 만듭니다. "
        "보통 단일 모델보다 더 안정적이고 정확한 결과가 나옵니다."
    )
    use_ensemble = st.checkbox("앙상블 모델 추가", value=False)
    if use_ensemble:
        ensemble_method = st.radio(
            "앙상블 방식",
            ["mean", "weighted"],
            format_func=lambda x: {
                "mean": "단순 평균",
                "weighted": "RMSE 역수 가중 (성능 좋은 모델에 더 큰 비중)",
            }[x],
            horizontal=True,
        )
    else:
        ensemble_method = "mean"


# =============================================================================
# 학습/검증 분할 + 예측 실행
# =============================================================================
if horizon >= len(ts):
    st.error(f"❌ 시평({horizon})이 데이터 길이({len(ts)})보다 크거나 같습니다. 시평을 줄이거나 데이터를 늘려주세요.")
    st.markdown("""
    ### 💡 해결 방법
    - 사이드바의 **시평을 줄이세요** (예: 12 → 6).
    - 또는 더 긴 시계열 데이터를 사용해주세요.
    """)
    st.stop()
if len(ts) - horizon < 4:
    st.warning(f"⚠️ 학습 데이터가 너무 적습니다 ({len(ts) - horizon}개). 결과 신뢰도가 낮을 수 있습니다.")

train, test = train_test_split_ts(ts, horizon)

st.plotly_chart(
    plot_train_test(train, test, title=f"학습 {len(train)} / 검증(시평) {len(test)} 분할"),
    use_container_width=True,
)


forecasts: dict[str, pd.Series] = {}
errors: dict[str, str] = {}

with st.spinner("🔮 모델 학습 및 예측 중..."):
    progress = st.progress(0.0)
    for i, k in enumerate(selected_models):
        info = MODEL_REGISTRY[k]
        try:
            fc = run_forecast(k, train, int(horizon), freq, seasonal_period)
            forecasts[info.name] = fc
        except Exception as e:
            errors[k] = str(e)
        progress.progress((i + 1) / len(selected_models))
    progress.empty()

if errors:
    with st.expander("⚠️ 일부 모델 실행 실패", expanded=False):
        for k, msg in errors.items():
            st.error(f"`{MODEL_REGISTRY[k].name}` — {msg}")

if not forecasts:
    st.error("모든 모델이 실패했습니다. 데이터나 라이브러리 설치를 확인해주세요.")
    st.stop()

# 2-2) ARIMA 수동 추가
if use_manual_arima:
    try:
        manual_fc = forecast_arima_manual(
            train, int(horizon), freq, seasonal_period,
            p=int(arima_p), d=int(arima_d), q=int(arima_q),
        )
        forecasts[f"ARIMA({arima_p},{arima_d},{arima_q}) 수동"] = manual_fc
    except Exception as e:
        st.warning(f"ARIMA 수동 차수 실행 실패: {e}")

# 2-3) 앙상블 추가
if use_ensemble and len(forecasts) >= 2:
    try:
        if ensemble_method == "mean":
            ens_fc = ensemble_simple_mean(forecasts)
            forecasts["🎯 앙상블 (단순 평균)"] = ens_fc
        else:
            # 가중 평균은 일단 모든 모델 RMSE를 미리 계산해야 함
            # 임시로 단순 평균 추가하고 평가지표 계산 후 뒤에서 가중평균 생성
            ens_fc = ensemble_simple_mean(forecasts)
            forecasts["🎯 앙상블 (단순 평균)"] = ens_fc
    except Exception as e:
        st.warning(f"앙상블 생성 실패: {e}")


# =============================================================================
# 예측 시각화
# =============================================================================
st.header("3. 예측 결과 비교")
st.plotly_chart(
    plot_forecasts(train, test, forecasts, title="모델별 예측 vs 실제값 (검증 구간)"),
    use_container_width=True,
)


# =============================================================================
# 평가지표 대시보드
# =============================================================================
st.header("4. 평가지표 대시보드")

metrics_rows = []
for name, fc in forecasts.items():
    # 인덱스를 test와 동일하게 정렬
    fc_aligned = pd.Series(fc.values[:len(test)], index=test.index)
    m = all_metrics(test, fc_aligned, y_train=train, seasonal_period=seasonal_period or 1)
    m["Model"] = name
    metrics_rows.append(m)

metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
metrics_df = metrics_df[["MAE", "MSE", "RMSE", "MAPE", "sMAPE", "R2", "MASE", "Bias"]]

# 가중 앙상블이 요청된 경우, RMSE 기반으로 다시 계산
if use_ensemble and ensemble_method == "weighted" and len(forecasts) >= 3:
    try:
        # 앙상블 자체는 평가에서 제외하고 가중 앙상블 새로 만듦
        base_forecasts = {k: v for k, v in forecasts.items() if not k.startswith("🎯")}
        rmse_dict = {k: metrics_df.loc[k, "RMSE"] if k in metrics_df.index else float("inf")
                     for k in base_forecasts}
        weighted_fc = ensemble_inverse_rmse(base_forecasts, rmse_dict)
        forecasts["🎯 앙상블 (RMSE 가중)"] = weighted_fc
        # 평가지표 추가
        fc_aligned = pd.Series(weighted_fc.values[:len(test)], index=test.index)
        m = all_metrics(test, fc_aligned, y_train=train, seasonal_period=seasonal_period or 1)
        m["Model"] = "🎯 앙상블 (RMSE 가중)"
        metrics_df.loc["🎯 앙상블 (RMSE 가중)"] = [
            m["MAE"], m["MSE"], m["RMSE"], m["MAPE"], m["sMAPE"],
            m["R2"], m["MASE"], m["Bias"],
        ]
    except Exception as e:
        st.warning(f"가중 앙상블 생성 실패: {e}")

# 4-1) 베스트 모델 강조
best_by_rmse = metrics_df["RMSE"].idxmin() if metrics_df["RMSE"].notna().any() else None
best_by_mape = metrics_df["MAPE"].idxmin() if metrics_df["MAPE"].notna().any() else None
best_by_r2 = metrics_df["R2"].idxmax() if metrics_df["R2"].notna().any() else None

cc1, cc2, cc3 = st.columns(3)
cc1.metric("🥇 RMSE 기준 최우수", best_by_rmse or "—",
           f"{metrics_df.loc[best_by_rmse, 'RMSE']:.4f}" if best_by_rmse else "")
cc2.metric("🥇 MAPE 기준 최우수", best_by_mape or "—",
           f"{metrics_df.loc[best_by_mape, 'MAPE']:.2f}%" if best_by_mape else "")
cc3.metric("🥇 R² 기준 최우수", best_by_r2 or "—",
           f"{metrics_df.loc[best_by_r2, 'R2']:.4f}" if best_by_r2 else "")

# 4-2) 지표 표 (스타일링)
def color_best(s: pd.Series) -> list[str]:
    """각 지표 컬럼에서 최고값을 강조."""
    if s.name == "R2":
        best = s.max()
    elif s.name == "Bias":
        best = s.iloc[(s.abs()).argmin()] if s.notna().any() else None
    else:
        best = s.min()
    return [
        "background-color: #d4edda; font-weight: 600" if v == best else ""
        for v in s
    ]

styled = metrics_df.style.apply(color_best, axis=0).format({
    "MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}",
    "MAPE": "{:.2f}%", "sMAPE": "{:.2f}%",
    "R2": "{:.4f}", "MASE": "{:.4f}", "Bias": "{:+.4f}",
})
st.dataframe(styled, use_container_width=True)

with st.expander("📚 지표 설명"):
    descs = metric_descriptions()
    for k, v in descs.items():
        st.markdown(f"- **{k}**: {v}")

# 4-3) 지표 비교 차트들
mc1, mc2 = st.columns(2)
with mc1:
    chosen_metric = st.selectbox(
        "비교할 지표 선택", ["RMSE", "MAE", "MAPE", "sMAPE", "R2", "MASE"],
        index=0,
    )
    st.plotly_chart(plot_metric_comparison(metrics_df, chosen_metric), use_container_width=True)
with mc2:
    st.plotly_chart(plot_metrics_radar(metrics_df), use_container_width=True)


# 4-4) 자동 진단 메시지 — "예측이 적절한가?"
st.subheader("🩺 자동 진단: 이 예측은 적절한가?")
diag_messages = []
if best_by_rmse:
    best_metrics = metrics_df.loc[best_by_rmse]
    # 1) MASE 기반: Naive 대비 우수성
    mase_v = best_metrics.get("MASE", float("nan"))
    if not pd.isna(mase_v):
        if mase_v < 0.5:
            diag_messages.append(("success", f"✅ **Naive 대비 매우 우수**합니다 (MASE={mase_v:.3f}). 이 모델을 사용해도 좋습니다."))
        elif mase_v < 1.0:
            diag_messages.append(("success", f"✅ **Naive보다 우수**합니다 (MASE={mase_v:.3f}). 모델이 의미 있는 정보를 학습했습니다."))
        elif mase_v < 1.5:
            diag_messages.append(("warning", f"⚠️ **Naive와 비슷한 수준**입니다 (MASE={mase_v:.3f}). 다른 모델도 시도해보세요."))
        else:
            diag_messages.append(("error", f"❌ **Naive보다 나쁨** (MASE={mase_v:.3f}). 데이터 특성과 맞지 않는 모델일 수 있습니다."))
    # 2) MAPE 기반: 절대 오차 수준
    mape_v = best_metrics.get("MAPE", float("nan"))
    if not pd.isna(mape_v):
        if mape_v < 10:
            diag_messages.append(("success", f"✅ **MAPE {mape_v:.1f}%** — 매우 정확한 예측 (10% 미만)."))
        elif mape_v < 20:
            diag_messages.append(("success", f"✅ **MAPE {mape_v:.1f}%** — 양호한 예측 (20% 미만)."))
        elif mape_v < 50:
            diag_messages.append(("warning", f"⚠️ **MAPE {mape_v:.1f}%** — 보통 수준. 시평을 줄이거나 다른 모델을 고려하세요."))
        else:
            diag_messages.append(("error", f"❌ **MAPE {mape_v:.1f}%** — 오차가 큽니다. 데이터 양/품질을 점검하세요."))
    # 3) R² 기반: 분산 설명력
    r2_v = best_metrics.get("R2", float("nan"))
    if not pd.isna(r2_v):
        if r2_v > 0.7:
            diag_messages.append(("success", f"✅ **R² {r2_v:.3f}** — 데이터 변동의 70% 이상을 설명합니다."))
        elif r2_v > 0.3:
            diag_messages.append(("warning", f"⚠️ **R² {r2_v:.3f}** — 일부 패턴은 잡지만 개선 여지 있음."))
        elif r2_v > 0:
            diag_messages.append(("warning", f"⚠️ **R² {r2_v:.3f}** — 단순 평균 대비 약간 우수한 수준."))
        else:
            diag_messages.append(("error", f"❌ **R² {r2_v:.3f}** (음수) — 단순 평균보다도 못함. 모델 재선택 필요."))
    # 4) Bias 기반: 편향
    bias_v = best_metrics.get("Bias", float("nan"))
    abs_bias_pct = abs(bias_v) / abs(summary["mean"]) * 100 if summary["mean"] != 0 else 0
    if not pd.isna(bias_v):
        if abs_bias_pct < 2:
            diag_messages.append(("success", f"✅ **편향 {bias_v:+.3f}** — 한쪽으로 치우치지 않음 (평균값의 {abs_bias_pct:.1f}%)."))
        elif abs_bias_pct < 5:
            diag_messages.append(("warning", f"⚠️ **편향 {bias_v:+.3f}** — 약한 {'과소예측' if bias_v > 0 else '과대예측'} 경향."))
        else:
            direction = "과소예측" if bias_v > 0 else "과대예측"
            diag_messages.append(("error", f"❌ **편향 {bias_v:+.3f}** — 강한 {direction} 경향 (평균값의 {abs_bias_pct:.1f}%)."))

st.markdown(f"**최우수 모델 ({best_by_rmse})**에 대한 종합 진단:")
for level, msg in diag_messages:
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.error(msg)

# 종합 의견
if diag_messages:
    success_cnt = sum(1 for l, _ in diag_messages if l == "success")
    warning_cnt = sum(1 for l, _ in diag_messages if l == "warning")
    error_cnt = sum(1 for l, _ in diag_messages if l == "error")
    if error_cnt >= 2:
        st.error(f"🚨 **종합: 이 예측은 신뢰하기 어렵습니다.** 데이터/모델 재검토 권장.")
    elif error_cnt == 1 or warning_cnt >= 2:
        st.warning(f"📋 **종합: 일부 우려가 있는 예측입니다.** 다른 모델과 비교해 보세요.")
    else:
        st.success(f"🎯 **종합: 이 예측은 신뢰할 만합니다.** 미래 예측에 사용해도 좋습니다.")


# =============================================================================
# 잔차 분석
# =============================================================================
st.header("5. 잔차 분석")
chosen_for_residual = st.selectbox(
    "잔차를 자세히 볼 모델",
    list(forecasts.keys()),
    index=list(forecasts.keys()).index(best_by_rmse) if best_by_rmse in forecasts else 0,
)
chosen_fc = pd.Series(forecasts[chosen_for_residual].values[:len(test)], index=test.index)
st.plotly_chart(plot_residuals(test, chosen_fc, chosen_for_residual), use_container_width=True)

# 5-2) Ljung-Box 검정 + 잔차 ACF
residuals = test - chosen_fc
res_c1, res_c2 = st.columns([1, 2])
with res_c1:
    st.markdown("**Ljung-Box 검정 (잔차 자기상관)**")
    lb_lags = min(10, max(2, len(residuals) // 3))
    lb_result = ljung_box_test(residuals, lags=lb_lags)
    st.metric("p-value",
              f"{lb_result['p_value']:.4f}" if not pd.isna(lb_result["p_value"]) else "—")
    if lb_result["passes"]:
        st.success(lb_result["interpretation"])
    else:
        st.warning(lb_result["interpretation"])
with res_c2:
    st.markdown("**잔차 ACF (자기상관이 남아있는지 확인)**")
    if len(residuals.dropna()) >= 4:
        nlags = min(20, len(residuals) - 1)
        r_acf, r_pacf, r_conf = compute_acf_pacf(residuals.dropna(), nlags=nlags)
        if len(r_acf) > 0:
            st.plotly_chart(
                plot_acf_pacf(r_acf, r_pacf, r_conf, title="잔차의 ACF / PACF"),
                use_container_width=True,
            )
        else:
            st.info("잔차 ACF 계산 불가.")
    else:
        st.info("잔차가 너무 적어 ACF 계산 불가.")

# 5-3) QQ plot + Shapiro 정규성 검정
qq_c1, qq_c2 = st.columns([2, 1])
with qq_c1:
    st.markdown("**잔차 QQ plot (정규성 시각 진단)**")
    qx, qy = qq_data(residuals)
    if len(qx) > 0:
        st.plotly_chart(plot_qq(qx, qy), use_container_width=True)
        st.caption("점들이 빨간 직선에 가까울수록 잔차가 정규분포에 가깝습니다 (신뢰구간 신뢰성 ↑).")
    else:
        st.info("QQ plot을 그릴 데이터가 부족합니다.")
with qq_c2:
    st.markdown("**Shapiro-Wilk 정규성 검정**")
    sw = shapiro_test(residuals)
    st.metric("p-value",
              f"{sw['p_value']:.4f}" if not pd.isna(sw["p_value"]) else "—")
    if sw["is_normal"]:
        st.success(sw["interpretation"])
    else:
        st.warning(sw["interpretation"])

# 5-4) 다중 horizon 평가 (단기/중기/장기)
st.markdown("---")
st.subheader("⏱️ 시평 구간별 성능 (단기 / 중기 / 장기)")
multi_results = evaluate_multi_horizon(test, chosen_fc)
mh_c1, mh_c2 = st.columns([2, 1])
with mh_c1:
    st.plotly_chart(
        plot_multi_horizon(multi_results, chosen_for_residual),
        use_container_width=True,
    )
with mh_c2:
    st.markdown("**구간별 RMSE / MAPE**")
    multi_df = pd.DataFrame({
        label: {"RMSE": m["RMSE"], "MAPE(%)": m["MAPE"]}
        for label, m in multi_results.items()
    }).T
    st.dataframe(multi_df.style.format({"RMSE": "{:.3f}", "MAPE(%)": "{:.2f}"}),
                 use_container_width=True)
    st.caption("📌 시평이 길수록 오차가 커지면 정상. 단기에서도 큰 오차는 모델 부적합 신호.")


# =============================================================================
# 미래 예측 (전체 데이터로 다시 학습 후 시평만큼 미래로)
# =============================================================================
st.header("6. 미래 예측")
st.caption("검증을 통과한 모델로 전체 데이터를 다시 학습하여 향후 시평만큼을 예측합니다.")

# best_by_rmse(사람 친화 이름)에서 키를 역매핑
name_to_key = {info.name: k for k, info in MODEL_REGISTRY.items()}

# 미래 예측에 쓸 모델 목록: forecasts.keys() 전체 (앙상블/수동ARIMA 포함)
future_options = list(forecasts.keys())
default_index = 0
if best_by_rmse and best_by_rmse in future_options:
    default_index = future_options.index(best_by_rmse)

future_model_label = st.selectbox(
    "미래 예측에 사용할 모델",
    future_options,
    index=default_index,
)

# 선택된 모델로 미래 예측 실행
def _is_ensemble(name: str) -> bool:
    return name.startswith("🎯")

def _is_manual_arima(name: str) -> bool:
    return name.startswith("ARIMA(") and "수동" in name

with st.spinner("🔮 전체 데이터로 재학습 후 미래 예측 중..."):
    future_lower = None
    future_upper = None
    if _is_ensemble(future_model_label):
        # 앙상블: 모든 베이스 모델을 ts 전체로 재학습 후 평균
        base_keys = [k for k in selected_models]
        base_fcs = {}
        for k in base_keys:
            try:
                base_fcs[MODEL_REGISTRY[k].name] = run_forecast(k, ts, int(horizon), freq, seasonal_period)
            except Exception:
                continue
        if "RMSE" in future_model_label and ensemble_method == "weighted":
            rmse_dict = {n: metrics_df.loc[n, "RMSE"] if n in metrics_df.index else float("inf")
                         for n in base_fcs.keys()}
            future_fc = ensemble_inverse_rmse(base_fcs, rmse_dict)
        else:
            future_fc = ensemble_simple_mean(base_fcs)
        # 앙상블의 신뢰구간: 베이스 모델들 분산으로 근사
        if enable_ci and len(base_fcs) >= 2:
            arr = np.array([fc.values for fc in base_fcs.values()])
            std_arr = arr.std(axis=0)
            future_lower = pd.Series(future_fc.values - 1.96 * std_arr, index=future_fc.index)
            future_upper = pd.Series(future_fc.values + 1.96 * std_arr, index=future_fc.index)
    elif _is_manual_arima(future_model_label):
        future_fc = forecast_arima_manual(
            ts, int(horizon), freq, seasonal_period,
            p=int(arima_p), d=int(arima_d), q=int(arima_q),
        )
        if enable_ci:
            # 수동 ARIMA에 대해 CI는 ARIMA 키로 처리
            ci_result = forecast_with_ci("arima", ts, int(horizon), freq, seasonal_period)
            future_lower = ci_result["lower"]
            future_upper = ci_result["upper"]
    else:
        # 일반 모델
        future_model_key = name_to_key.get(future_model_label)
        if future_model_key is None:
            st.error(f"모델을 찾을 수 없습니다: {future_model_label}")
            st.stop()
        if enable_ci:
            ci_result = forecast_with_ci(future_model_key, ts, int(horizon), freq, seasonal_period)
            future_fc = ci_result["mean"]
            future_lower = ci_result["lower"]
            future_upper = ci_result["upper"]
        else:
            future_fc = run_forecast(future_model_key, ts, int(horizon), freq, seasonal_period)

if enable_ci and future_lower is not None:
    st.plotly_chart(
        plot_future_forecast_with_ci(
            ts, future_fc, future_lower, future_upper,
            future_model_label, int(horizon),
        ),
        use_container_width=True,
    )
else:
    st.plotly_chart(
        plot_future_forecast(ts, future_fc, future_model_label, int(horizon)),
        use_container_width=True,
    )

# 6-2) 신뢰구간 적중률 (Coverage) — 검증 구간에서 측정
if enable_ci:
    with st.expander("🎯 신뢰구간 적중률 검증 (Coverage)", expanded=False):
        st.markdown(
            "검증 구간에서 95% 신뢰구간이 실제로 95%를 맞추는지 확인합니다. "
            "이상적으로는 **명목 95% ≈ 실제 95%**가 되어야 신뢰구간을 신뢰할 수 있습니다."
        )
        # 검증 구간에서 동일 모델의 CI를 계산
        try:
            if _is_ensemble(future_model_label) or _is_manual_arima(future_model_label):
                st.info("앙상블/수동 ARIMA의 정밀 적중률 계산은 생략합니다.")
            else:
                fmk = name_to_key.get(future_model_label)
                if fmk:
                    val_ci = forecast_with_ci(fmk, train, int(horizon), freq, seasonal_period)
                    val_lower = pd.Series(val_ci["lower"].values[:len(test)], index=test.index)
                    val_upper = pd.Series(val_ci["upper"].values[:len(test)], index=test.index)
                    cov = prediction_interval_coverage(test, val_lower, val_upper, nominal=0.95)
                    cov_c1, cov_c2 = st.columns([1, 2])
                    with cov_c1:
                        st.metric("실제 적중률",
                                  f"{cov['coverage']*100:.1f}%",
                                  delta=f"{(cov['coverage']-cov['nominal'])*100:+.1f}%p (vs 명목)")
                        st.metric("적중 / 전체",
                                  f"{cov['n_inside']} / {cov['n_total']}")
                    with cov_c2:
                        if abs(cov["coverage"] - cov["nominal"]) < 0.05:
                            st.success(cov["interpretation"])
                        else:
                            st.warning(cov["interpretation"])
        except Exception as e:
            st.info(f"적중률 계산 불가: {e}")

# 미래 예측 결과 다운로드 (평가지표 메타데이터 포함)
fc_df = pd.DataFrame({
    "시점": future_fc.index,
    "예측값": future_fc.values,
})
if enable_ci and future_lower is not None:
    fc_df["하한_95%"] = future_lower.values
    fc_df["상한_95%"] = future_upper.values

# 평가지표를 헤더 주석으로 추가
best_row = metrics_df.loc[future_model_label] if future_model_label in metrics_df.index else None
header_lines = [
    f"# 시계열 예측 결과",
    f"# 모델: {future_model_label}",
    f"# 시평(horizon): {horizon}",
    f"# 빈도(freq): {freq or 'N/A'}",
    f"# 계절주기(m): {seasonal_period or 'N/A'}",
    f"# 학습 데이터 기간: {summary['start']} ~ {summary['end']}",
    f"# 학습 데이터 관측치 수: {summary['n_observations']}",
]
if best_row is not None:
    header_lines.append(f"# --- 검증 구간 평가지표 ---")
    header_lines.append(f"# MAE: {best_row['MAE']:.4f}")
    header_lines.append(f"# RMSE: {best_row['RMSE']:.4f}")
    header_lines.append(f"# MAPE: {best_row['MAPE']:.2f}%")
    header_lines.append(f"# R2: {best_row['R2']:.4f}")
    header_lines.append(f"# MASE: {best_row['MASE']:.4f}")
csv_with_meta = "\n".join(header_lines) + "\n" + fc_df.to_csv(index=False)
csv_bytes = csv_with_meta.encode("utf-8-sig")

st.download_button(
    "📥 미래 예측 결과 CSV 다운로드 (평가지표 포함)",
    data=csv_bytes,
    file_name=f"forecast_{future_model_label}_h{horizon}.csv",
    mime="text/csv",
)


# =============================================================================
# 7. 롤링 교차검증 (선택)
# =============================================================================
if enable_cv:
    st.header("7. 롤링 교차검증 (Walk-forward CV)")
    st.caption(
        "여러 시점에서 학습/검증을 반복하여 모델의 안정성을 확인합니다. "
        "각 fold마다 학습 데이터가 점차 늘어나는 expanding window 방식입니다."
    )
    with st.spinner(f"🔄 {cv_splits} folds로 롤링 교차검증 실행 중..."):
        cv_results = rolling_cv(
            future_model_key, ts, int(horizon),
            n_splits=int(cv_splits),
            freq=freq, seasonal_period=seasonal_period,
        )

    if cv_results["fold"]:
        st.plotly_chart(
            plot_rolling_cv(cv_results, future_model_label),
            use_container_width=True,
        )
        # 안정성 진단
        rmse_arr = np.array(cv_results["rmse"])
        cv_mean = rmse_arr.mean()
        cv_std = rmse_arr.std()
        cv_cv = cv_std / cv_mean if cv_mean > 0 else 0
        st_c1, st_c2, st_c3 = st.columns(3)
        st_c1.metric("RMSE 평균", f"{cv_mean:.4f}")
        st_c2.metric("RMSE 표준편차", f"{cv_std:.4f}")
        st_c3.metric("변동 계수 (CV)", f"{cv_cv:.3f}")
        if cv_cv < 0.2:
            st.success("✅ **CV가 작음 (<20%)** — 모델이 안정적입니다. 다양한 시점에서 일관된 성능을 보입니다.")
        elif cv_cv < 0.5:
            st.warning("⚠️ **중간 정도의 변동성** — 시점에 따라 성능 편차가 있습니다.")
        else:
            st.error("❌ **변동성이 큼** — 모델이 시점에 민감합니다. 다른 모델을 시도해보세요.")
    else:
        st.info("CV를 실행하기에 데이터가 부족합니다. 데이터를 늘리거나 시평/fold 수를 줄이세요.")


# =============================================================================
# 8. 백테스팅 시뮬레이션
# =============================================================================
st.header("8. 백테스팅 시뮬레이션")
st.caption(
    "과거 여러 시점에 이 모델로 예측했다면 어떤 결과였을지를 시뮬레이션합니다. "
    "이 모델이 시간이 지나도 일관되게 잘 작동하는지 확인하는 도구입니다."
)
backtest_run = st.checkbox("백테스팅 실행", value=False, key="run_backtest")
if backtest_run:
    n_backtests = st.slider("백테스팅 횟수", min_value=2, max_value=8, value=4)
    with st.spinner(f"📊 {n_backtests}개 시점에서 백테스팅 중..."):
        # 데이터 끝에서 horizon만큼씩 뒤로 가면서 cutoff 잡기
        bt_results = []
        n_data = len(ts)
        min_train = max(2 * (seasonal_period or 1), 12)
        # 가능한 cutoff 범위
        cutoff_min = min_train + horizon
        cutoff_max = n_data
        if cutoff_min >= cutoff_max:
            st.warning("⚠️ 데이터가 부족해 백테스팅을 실행할 수 없습니다.")
        else:
            cutoffs = np.linspace(cutoff_min, cutoff_max, n_backtests, dtype=int)
            for cut in cutoffs:
                tr = ts.iloc[:cut - horizon]
                te = ts.iloc[cut - horizon:cut]
                if len(tr) < min_train or len(te) < horizon:
                    continue
                try:
                    if _is_ensemble(future_model_label):
                        base_keys = [k for k in selected_models]
                        base_fcs = {}
                        for k in base_keys:
                            try:
                                base_fcs[MODEL_REGISTRY[k].name] = run_forecast(
                                    k, tr, int(horizon), freq, seasonal_period)
                            except Exception:
                                continue
                        if base_fcs:
                            fc = ensemble_simple_mean(base_fcs)
                        else:
                            continue
                    elif _is_manual_arima(future_model_label):
                        fc = forecast_arima_manual(
                            tr, int(horizon), freq, seasonal_period,
                            p=int(arima_p), d=int(arima_d), q=int(arima_q),
                        )
                    else:
                        fc = run_forecast(name_to_key[future_model_label], tr,
                                          int(horizon), freq, seasonal_period)
                    bt_results.append({
                        "cutoff": tr.index[-1] if len(tr) > 0 else cut,
                        "forecast": pd.Series(fc.values[:len(te)], index=te.index),
                        "actual": te,
                    })
                except Exception:
                    continue

    if bt_results:
        st.plotly_chart(
            plot_backtest(ts, bt_results,
                          title=f"백테스팅 - {future_model_label} ({len(bt_results)} 회)"),
            use_container_width=True,
        )
        # 백테스트 평균 성능
        bt_rmses = []
        bt_mapes = []
        for bt in bt_results:
            from modules.metrics import rmse as _rmse, mape as _mape
            bt_rmses.append(_rmse(bt["actual"], bt["forecast"]))
            bt_mapes.append(_mape(bt["actual"], bt["forecast"]))
        bt_c1, bt_c2 = st.columns(2)
        bt_c1.metric("평균 RMSE (백테스트)", f"{np.mean(bt_rmses):.4f}",
                     delta=f"std {np.std(bt_rmses):.4f}")
        bt_c2.metric("평균 MAPE (백테스트)", f"{np.mean(bt_mapes):.2f}%",
                     delta=f"std {np.std(bt_mapes):.2f}%")
        if np.std(bt_rmses) / max(np.mean(bt_rmses), 1e-9) < 0.3:
            st.success("✅ 백테스트 결과가 일관적입니다. 모델이 시간에 걸쳐 안정적으로 동작합니다.")
        else:
            st.warning("⚠️ 시점에 따라 성능 변동이 큽니다. 트렌드 변화에 민감할 수 있습니다.")
    else:
        st.info("백테스팅 결과가 없습니다.")


# =============================================================================
# 푸터
# =============================================================================
st.divider()
st.caption(
    "📚 **모델 (12종)**: Naive 4종 · 평활법 5종 (SES, Holt, HW가법, HW곱셈, Theta) · "
    "ARIMA 3종 (수동, Auto, sktime)  |  "
    "📊 **지표 (8종)**: MAE · MSE · RMSE · MAPE · sMAPE · R² · MASE · Bias  |  "
    "🔬 **진단**: 품질점수 · 특성게이지 · 이상치 · 변동점 · ADF · ACF/PACF · "
    "Ljung-Box · Shapiro · QQ-plot · 차분 · 시계열분해 · Coverage  |  "
    "✨ **고급**: 95% 신뢰구간 · 롤링 CV · 백테스팅 · 앙상블 · 다중 horizon · "
    "변환(log/Box-Cox) · 다운샘플링 · 자동 추천  |  "
    "🛠️ Built with Streamlit + statsmodels + Plotly + scipy"
)
