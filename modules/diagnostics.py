"""
diagnostics.py
시계열의 통계적 특성을 진단하는 모듈.

기능:
- ADF 정상성 검정 (Augmented Dickey-Fuller Test)
- ACF / PACF 자기상관 함수
- 시계열 분해 (trend / seasonal / residual)
- Ljung-Box 검정 (잔차 자기상관 검사)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# 1) ADF 정상성 검정
# -----------------------------------------------------------------------------
def adf_test(series: pd.Series) -> Dict[str, float | str | bool]:
    """
    Augmented Dickey-Fuller test로 정상성(stationarity) 검정.
    p-value < 0.05이면 "정상 시계열"로 판단 (ARIMA 적용 가능).
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        clean = series.dropna().values
        if len(clean) < 10:
            return {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "is_stationary": False,
                "interpretation": "데이터가 너무 적어 검정 불가",
                "n_lags": 0,
            }
        result = adfuller(clean, autolag="AIC")
        stat, p, lags, n_obs, crit, _ = result
        is_stat = bool(p < 0.05)
        if is_stat:
            interp = f"✅ 정상 시계열 (p={p:.4f} < 0.05). ARIMA 적용에 적합합니다."
        else:
            interp = f"⚠️ 비정상 시계열 (p={p:.4f} ≥ 0.05). 차분(d≥1)이 필요할 수 있습니다."
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "is_stationary": is_stat,
            "interpretation": interp,
            "n_lags": int(lags),
            "critical_1%": float(crit.get("1%", float("nan"))),
            "critical_5%": float(crit.get("5%", float("nan"))),
            "critical_10%": float(crit.get("10%", float("nan"))),
        }
    except Exception as e:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "is_stationary": False,
            "interpretation": f"검정 실패: {e}",
            "n_lags": 0,
        }


# -----------------------------------------------------------------------------
# 2) ACF / PACF
# -----------------------------------------------------------------------------
def compute_acf_pacf(series: pd.Series, nlags: int = 30) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns
    -------
    acf_vals, pacf_vals, conf_band
        conf_band = 95% 신뢰구간 폭 (대략 1.96/sqrt(N))
    """
    try:
        from statsmodels.tsa.stattools import acf, pacf
        clean = series.dropna().values
        n = len(clean)
        nlags_eff = min(nlags, max(1, n // 2 - 1))
        acf_vals = acf(clean, nlags=nlags_eff, fft=True)
        pacf_vals = pacf(clean, nlags=nlags_eff, method="ywm")
        conf = 1.96 / np.sqrt(n) if n > 0 else 0.0
        return acf_vals, pacf_vals, float(conf)
    except Exception:
        return np.array([]), np.array([]), 0.0


# -----------------------------------------------------------------------------
# 3) 시계열 분해
# -----------------------------------------------------------------------------
def decompose(
    series: pd.Series,
    seasonal_period: Optional[int],
    model: str = "additive",
) -> Optional[Dict[str, pd.Series]]:
    """
    statsmodels의 seasonal_decompose로 추세/계절/잔차 분해.

    Returns dict with keys: trend, seasonal, resid, observed
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        if not seasonal_period or seasonal_period <= 1:
            return None
        if len(series) < 2 * seasonal_period:
            return None
        # 음수가 있으면 multiplicative 불가
        if model == "multiplicative" and (series.values <= 0).any():
            model = "additive"
        result = seasonal_decompose(
            series, model=model, period=seasonal_period,
            extrapolate_trend="freq",
        )
        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
            "observed": result.observed,
        }
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 4) Ljung-Box 검정 (잔차의 자기상관 유무)
# -----------------------------------------------------------------------------
def ljung_box_test(residuals: pd.Series, lags: int = 10) -> Dict[str, float | str | bool]:
    """
    잔차에 자기상관이 남아있는지 검정.
    p-value > 0.05이면 "잔차가 백색잡음" → 모델이 정보를 잘 뽑아냄.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        clean = pd.Series(residuals).dropna()
        if len(clean) < lags + 5:
            return {
                "p_value": float("nan"),
                "passes": False,
                "interpretation": "잔차 데이터가 너무 적습니다.",
            }
        out = acorr_ljungbox(clean, lags=[lags], return_df=True)
        p = float(out["lb_pvalue"].iloc[0])
        passes = p > 0.05
        if passes:
            interp = f"✅ 잔차에 패턴 없음 (p={p:.4f} > 0.05). 모델이 정보를 잘 추출했습니다."
        else:
            interp = f"⚠️ 잔차에 자기상관 잔존 (p={p:.4f} ≤ 0.05). 더 복잡한 모델이 필요할 수 있습니다."
        return {
            "p_value": p,
            "passes": passes,
            "interpretation": interp,
        }
    except Exception as e:
        return {
            "p_value": float("nan"),
            "passes": False,
            "interpretation": f"검정 실패: {e}",
        }


# -----------------------------------------------------------------------------
# 5) QQ plot 데이터 (잔차 정규성)
# -----------------------------------------------------------------------------
def qq_data(residuals: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    QQ plot용 (이론 분위수, 표본 분위수) 반환.
    완벽한 정규분포면 직선 위에 점들이 놓임.
    """
    clean = pd.Series(residuals).dropna().values
    n = len(clean)
    if n < 4:
        return np.array([]), np.array([])
    sorted_vals = np.sort(clean)
    # 이론 분위수 (정규분포)
    try:
        from scipy.stats import norm
        # 위치 (Hazen plotting position)
        positions = (np.arange(1, n + 1) - 0.5) / n
        theoretical = norm.ppf(positions)
        return theoretical, sorted_vals
    except Exception:
        # scipy 없으면 표준 정규분포 근사
        positions = (np.arange(1, n + 1) - 0.5) / n
        # erfinv 근사 없이 단순 정렬값으로 폴백
        theoretical = np.sqrt(2) * np.array([
            _approx_inv_erf(2 * p - 1) for p in positions
        ])
        return theoretical, sorted_vals


def _approx_inv_erf(x: float) -> float:
    """간이 erf 역함수 근사 (Winitzki, 2008)."""
    if x == 0:
        return 0.0
    a = 0.147
    sign = 1.0 if x >= 0 else -1.0
    ln1 = np.log(1.0 - x * x)
    term1 = 2.0 / (np.pi * a) + ln1 / 2.0
    term2 = ln1 / a
    return sign * np.sqrt(np.sqrt(term1 * term1 - term2) - term1)


def shapiro_test(residuals: pd.Series) -> Dict[str, float | str | bool]:
    """
    Shapiro-Wilk 정규성 검정.
    p-value > 0.05이면 정규분포로 볼 수 있음.
    """
    try:
        from scipy.stats import shapiro
        clean = pd.Series(residuals).dropna().values
        if len(clean) < 4 or len(clean) > 5000:
            return {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "is_normal": False,
                "interpretation": "샘플 수가 적합하지 않습니다 (4~5000).",
            }
        stat, p = shapiro(clean)
        is_normal = bool(p > 0.05)
        if is_normal:
            interp = f"✅ 정규분포로 볼 수 있음 (p={p:.4f}). 신뢰구간 신뢰도 ↑"
        else:
            interp = f"⚠️ 정규분포 아님 (p={p:.4f}). 신뢰구간 해석에 주의."
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "is_normal": is_normal,
            "interpretation": interp,
        }
    except Exception as e:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "is_normal": False,
            "interpretation": f"검정 실패: {e}",
        }


# -----------------------------------------------------------------------------
# 6) 예측 구간 적중률 (Coverage)
# -----------------------------------------------------------------------------
def prediction_interval_coverage(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    nominal: float = 0.95,
) -> Dict[str, float | str]:
    """
    실제값이 예측 구간 [lower, upper]에 얼마나 들어갔는지 계산.
    nominal=0.95면 이상적으로 95%가 들어가야 함.
    """
    try:
        a = np.asarray(actual.values)
        l = np.asarray(lower.values)
        u = np.asarray(upper.values)
        n = len(a)
        if n == 0:
            return {"coverage": float("nan"), "nominal": nominal,
                    "interpretation": "데이터 없음"}
        inside = np.sum((a >= l) & (a <= u))
        coverage = inside / n
        diff = abs(coverage - nominal)
        if diff < 0.05:
            interp = f"✅ 구간 적중률 {coverage*100:.1f}% — 명목 {nominal*100:.0f}%와 매우 가깝습니다."
        elif diff < 0.10:
            interp = f"⚠️ 구간 적중률 {coverage*100:.1f}% — 명목과 약간 차이가 있습니다."
        else:
            if coverage < nominal:
                interp = f"❌ 구간 적중률 {coverage*100:.1f}% — 신뢰구간이 좁습니다 (과신)."
            else:
                interp = f"⚠️ 구간 적중률 {coverage*100:.1f}% — 신뢰구간이 너무 넓습니다."
        return {
            "coverage": float(coverage),
            "nominal": float(nominal),
            "n_inside": int(inside),
            "n_total": int(n),
            "interpretation": interp,
        }
    except Exception as e:
        return {"coverage": float("nan"), "nominal": nominal,
                "interpretation": f"계산 실패: {e}"}


# -----------------------------------------------------------------------------
# 7) 차분 시각화 데이터 (ARIMA 강의 연계)
# -----------------------------------------------------------------------------
def differencing_series(series: pd.Series, max_d: int = 2) -> Dict[int, pd.Series]:
    """
    원본, 1차 차분, 2차 차분을 반환.
    ARIMA의 d 파라미터 결정에 도움.
    """
    out = {0: series}
    cur = series.copy()
    for d in range(1, max_d + 1):
        cur = cur.diff().dropna()
        out[d] = cur
    return out
