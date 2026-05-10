"""
preprocessing.py
시계열의 고급 전처리 기능.
- 이상치 탐지 (IQR, Z-score)
- 변동점 탐지 (간단한 누적합 기반)
- 다운샘플링 (일→주/월 등)
- 로그 / Box-Cox 변환
- 데이터 품질 점수
- 시계열 특성 정량화 (추세/계절/노이즈 강도)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 1) 이상치 탐지
# =============================================================================
def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    IQR 방식 이상치 탐지.
    Returns: bool Series (True = 이상치)
    """
    clean = series.dropna()
    q1, q3 = np.percentile(clean.values, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Z-score 방식 이상치 탐지."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean) / std
    return z.abs() > threshold


def handle_outliers(
    series: pd.Series,
    method: str = "interpolate",
    detection: str = "iqr",
    iqr_mult: float = 1.5,
    z_threshold: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    이상치 처리.
    detection: 'iqr' or 'zscore'
    method: 'keep' (그대로) | 'interpolate' (보간) | 'winsorize' (클리핑) | 'remove' (제거)

    Returns: (처리된 Series, 이상치 마스크 bool Series)
    """
    if detection == "iqr":
        mask = detect_outliers_iqr(series, iqr_mult)
    else:
        mask = detect_outliers_zscore(series, z_threshold)

    if method == "keep" or mask.sum() == 0:
        return series, mask
    out = series.copy()
    if method == "interpolate":
        out[mask] = np.nan
        out = out.interpolate(method="linear", limit_direction="both")
    elif method == "winsorize":
        clean = series[~mask]
        if len(clean) > 0:
            lo = float(clean.quantile(0.05))
            hi = float(clean.quantile(0.95))
            out = out.clip(lower=lo, upper=hi)
    elif method == "remove":
        out = out[~mask]
    return out, mask


# =============================================================================
# 2) 변동점 탐지 (간단한 누적합 기반 — 외부 라이브러리 불필요)
# =============================================================================
def detect_change_points(series: pd.Series, n_points: int = 3) -> List:
    """
    CUSUM 기반 변동점 탐지.
    누적 편차의 절대값이 큰 지점을 변동점으로 판정.
    외부 라이브러리(ruptures) 없이 numpy만으로 구현.

    Returns: 변동점 인덱스 위치들의 리스트
    """
    clean = series.dropna()
    if len(clean) < 20:
        return []
    values = clean.values
    mean = values.mean()
    cusum = np.cumsum(values - mean)
    abs_cusum = np.abs(cusum)

    # 상위 n_points 위치를 찾되, 너무 가까운 점들은 제외
    min_distance = max(5, len(values) // 20)
    candidates = np.argsort(abs_cusum)[::-1]
    selected = []
    for idx in candidates:
        if all(abs(idx - s) > min_distance for s in selected):
            selected.append(int(idx))
            if len(selected) >= n_points:
                break

    # 양 끝에서 너무 가까운 것 제거
    selected = [s for s in selected if 5 < s < len(values) - 5]
    return [clean.index[s] for s in sorted(selected)]


# =============================================================================
# 3) 다운샘플링
# =============================================================================
def resample_series(
    series: pd.Series,
    target_freq: str,
    agg: str = "mean",
) -> pd.Series:
    """
    시계열을 목표 빈도로 다운샘플링.
    agg: 'mean', 'sum', 'median', 'last'
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    if agg == "mean":
        return series.resample(target_freq).mean().dropna()
    elif agg == "sum":
        return series.resample(target_freq).sum().dropna()
    elif agg == "median":
        return series.resample(target_freq).median().dropna()
    elif agg == "last":
        return series.resample(target_freq).last().dropna()
    return series.resample(target_freq).mean().dropna()


# =============================================================================
# 4) 변환 (Log / Box-Cox)
# =============================================================================
def apply_log_transform(series: pd.Series) -> Tuple[pd.Series, Dict]:
    """
    log 변환. 음수/0이 있으면 shift 후 log.
    Returns: (변환된 series, 역변환에 필요한 정보)
    """
    values = series.values
    shift = 0.0
    min_v = float(np.min(values))
    if min_v <= 0:
        shift = abs(min_v) + 1.0
    transformed = np.log(values + shift)
    return pd.Series(transformed, index=series.index, name=series.name), {
        "method": "log",
        "shift": shift,
    }


def inverse_log_transform(series: pd.Series, info: Dict) -> pd.Series:
    """log 역변환."""
    values = np.exp(series.values) - info.get("shift", 0.0)
    return pd.Series(values, index=series.index, name=series.name)


def apply_boxcox_transform(series: pd.Series) -> Tuple[pd.Series, Dict]:
    """
    Box-Cox 변환. lambda 자동 추정.
    양수 데이터에만 적용 가능 (음수는 shift).
    """
    try:
        from scipy.stats import boxcox
        values = series.values
        shift = 0.0
        min_v = float(np.min(values))
        if min_v <= 0:
            shift = abs(min_v) + 1.0
        transformed, lam = boxcox(values + shift)
        return pd.Series(transformed, index=series.index, name=series.name), {
            "method": "boxcox",
            "shift": shift,
            "lambda": float(lam),
        }
    except Exception:
        return apply_log_transform(series)


def inverse_boxcox_transform(series: pd.Series, info: Dict) -> pd.Series:
    """Box-Cox 역변환."""
    try:
        from scipy.special import inv_boxcox
        values = inv_boxcox(series.values, info["lambda"]) - info.get("shift", 0.0)
        return pd.Series(values, index=series.index, name=series.name)
    except Exception:
        return series


# =============================================================================
# 5) 시계열 특성 정량화
# =============================================================================
def compute_features(
    series: pd.Series,
    seasonal_period: Optional[int] = None,
) -> Dict[str, float]:
    """
    시계열의 주요 특성을 정량화.
    추세 강도, 계절성 강도, 변동성, 노이즈 비율 등.
    """
    values = series.dropna().values
    n = len(values)
    if n < 4:
        return {
            "n": n, "mean": float("nan"), "std": float("nan"),
            "cv": float("nan"), "trend_strength": 0.0,
            "seasonal_strength": 0.0, "noise_ratio": 1.0,
            "autocorr_lag1": float("nan"),
        }

    mean = float(np.mean(values))
    std = float(np.std(values))
    cv = std / abs(mean) if mean != 0 else float("inf")

    # 1차 자기상관 (정상성 간이 측정)
    if n >= 2:
        x1 = values[:-1]
        x2 = values[1:]
        if np.std(x1) > 0 and np.std(x2) > 0:
            autocorr = float(np.corrcoef(x1, x2)[0, 1])
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0

    # 추세 강도 / 계절성 강도 (statsmodels 분해 기반)
    trend_strength = 0.0
    seasonal_strength = 0.0
    noise_ratio = 1.0
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        period = seasonal_period if seasonal_period and seasonal_period > 1 else None
        if period is not None and n >= 2 * period:
            decomp = seasonal_decompose(
                pd.Series(values), model="additive", period=period,
                extrapolate_trend="freq",
            )
            var_resid = float(np.nanvar(decomp.resid))
            var_seas_resid = float(np.nanvar(decomp.seasonal + decomp.resid))
            var_trend_resid = float(np.nanvar(decomp.trend + decomp.resid))
            var_total = float(np.nanvar(values))
            if var_seas_resid > 0:
                seasonal_strength = max(0.0, min(1.0, 1.0 - var_resid / var_seas_resid))
            if var_trend_resid > 0:
                trend_strength = max(0.0, min(1.0, 1.0 - var_resid / var_trend_resid))
            if var_total > 0:
                noise_ratio = max(0.0, min(1.0, var_resid / var_total))
        else:
            # 계절성 없는 경우: 단순 추세 (선형 회귀)
            t = np.arange(n)
            slope, intercept = np.polyfit(t, values, 1)
            fitted = slope * t + intercept
            var_resid = float(np.var(values - fitted))
            var_total = float(np.var(values))
            if var_total > 0:
                trend_strength = max(0.0, min(1.0, 1.0 - var_resid / var_total))
                noise_ratio = max(0.0, min(1.0, var_resid / var_total))
    except Exception:
        pass

    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "cv": float(cv) if not np.isinf(cv) else float("nan"),
        "trend_strength": float(trend_strength),
        "seasonal_strength": float(seasonal_strength),
        "noise_ratio": float(noise_ratio),
        "autocorr_lag1": float(autocorr),
    }


# =============================================================================
# 6) 데이터 품질 점수
# =============================================================================
def data_quality_score(
    series: pd.Series,
    features: Dict[str, float],
    n_outliers: int = 0,
    n_missing: int = 0,
) -> Dict[str, float | str]:
    """
    데이터 품질을 0~100으로 점수화.
    - 길이 충분성
    - 결측 비율
    - 이상치 비율
    - 신호 대 노이즈 비율 (1 - noise_ratio)
    - 자기상관 (예측 가능성)
    """
    n = len(series)
    if n == 0:
        return {"score": 0.0, "grade": "F", "interpretation": "데이터 없음"}

    # 1) 길이 점수: 50개 이상이면 만점
    length_score = min(100.0, (n / 50.0) * 100.0)

    # 2) 결측 점수: 결측 비율의 역수
    missing_ratio = n_missing / n if n > 0 else 0
    missing_score = max(0.0, 100.0 - missing_ratio * 500.0)  # 5%마다 -25점

    # 3) 이상치 점수
    outlier_ratio = n_outliers / n if n > 0 else 0
    outlier_score = max(0.0, 100.0 - outlier_ratio * 500.0)

    # 4) 신호 대 노이즈 (낮을수록 좋음)
    snr_score = max(0.0, 100.0 - features.get("noise_ratio", 1.0) * 100.0)

    # 5) 예측 가능성 (자기상관 절대값)
    ac = abs(features.get("autocorr_lag1", 0.0))
    predictability_score = ac * 100.0

    # 가중 평균
    total = (
        length_score * 0.20
        + missing_score * 0.20
        + outlier_score * 0.15
        + snr_score * 0.20
        + predictability_score * 0.25
    )
    total = max(0.0, min(100.0, total))

    if total >= 85:
        grade = "A"
        interp = "✅ 우수: 예측에 매우 적합한 데이터입니다."
    elif total >= 70:
        grade = "B"
        interp = "✅ 양호: 예측이 잘 동작할 것으로 기대됩니다."
    elif total >= 55:
        grade = "C"
        interp = "⚠️ 보통: 예측은 가능하나 정확도가 제한적일 수 있습니다."
    elif total >= 40:
        grade = "D"
        interp = "⚠️ 미흡: 데이터 품질 개선 후 재시도를 권장합니다."
    else:
        grade = "F"
        interp = "❌ 부적합: 신뢰할 수 있는 예측이 어렵습니다."

    return {
        "score": float(total),
        "grade": grade,
        "interpretation": interp,
        "components": {
            "length": float(length_score),
            "missing": float(missing_score),
            "outliers": float(outlier_score),
            "signal_noise": float(snr_score),
            "predictability": float(predictability_score),
        },
    }


# =============================================================================
# 7) 모델 자동 추천 (데이터 특성 기반)
# =============================================================================
def recommend_models(
    features: Dict[str, float],
    seasonal_period: Optional[int],
    n: int,
) -> Tuple[List[str], List[str]]:
    """
    데이터 특성에 따라 추천 모델을 결정.
    Returns: (추천 모델 키 리스트, 그 이유 메시지 리스트)
    """
    recs = []
    reasons = []

    has_season = bool(seasonal_period and seasonal_period > 1 and n >= 2 * seasonal_period)
    seasonal_strong = features.get("seasonal_strength", 0.0) > 0.3
    trend_strong = features.get("trend_strength", 0.0) > 0.3
    noisy = features.get("noise_ratio", 1.0) > 0.5
    predictable = abs(features.get("autocorr_lag1", 0.0)) > 0.5

    # 베이스라인은 항상 포함
    recs.append("naive_last")
    reasons.append("Naive(마지막값): 모든 모델 비교를 위한 기본 베이스라인")

    if has_season and seasonal_strong:
        recs.append("seasonal_naive")
        reasons.append(f"Seasonal Naive: 계절성 강도 {features['seasonal_strength']:.2f}")

    if trend_strong and not seasonal_strong:
        recs.append("drift")
        reasons.append(f"Drift: 추세 강도 {features['trend_strength']:.2f}, 계절성 약함")

    if not seasonal_strong and not trend_strong:
        recs.append("ses")
        reasons.append("SES(단순 평활): 추세/계절성 모두 약함 → 안정적 평활 적합")

    if trend_strong and not has_season:
        recs.append("holt")
        reasons.append("Holt(이중 평활): 추세는 있고 계절성은 없음")

    if has_season and seasonal_strong:
        recs.append("holt_winters")
        reasons.append("Holt-Winters: 추세+계절성 동시 처리")

    if predictable and n >= 30:
        recs.append("auto_arima")
        reasons.append(f"Auto-ARIMA: 자기상관 {features['autocorr_lag1']:.2f} (예측 가능성 높음)")

    # theta는 일반적으로 안정적
    if n >= 20:
        recs.append("theta")
        reasons.append("Theta: M-competition 우승, 단순하지만 안정적인 성능")

    # 중복 제거하면서 순서 유지
    seen = set()
    out = []
    out_reasons = []
    for r, why in zip(recs, reasons):
        if r not in seen:
            seen.add(r)
            out.append(r)
            out_reasons.append(why)
    return out, out_reasons
