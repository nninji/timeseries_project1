"""
models.py
시계열 예측 모델 모음 (강의자료 기반).

설계 원칙:
1) sktime 인터페이스를 1순위로 사용 (강의자료 05_sktime).
2) sktime/pmdarima가 없는 환경에서도 동작하도록 statsmodels 폴백 제공.
3) 모든 모델은 동일한 인터페이스: forecast(train: pd.Series, horizon: int) -> pd.Series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# 미래 인덱스 생성 헬퍼
# -----------------------------------------------------------------------------
def _future_index(train: pd.Series, horizon: int, freq: Optional[str] = None):
    if isinstance(train.index, pd.DatetimeIndex):
        f = freq or train.index.freqstr or pd.infer_freq(train.index)
        if f is None:
            # 마지막 두 시점 간격 사용
            delta = train.index[-1] - train.index[-2]
            return pd.date_range(start=train.index[-1] + delta, periods=horizon, freq=delta)
        return pd.date_range(start=train.index[-1], periods=horizon + 1, freq=f)[1:]
    else:
        start = int(train.index[-1]) + 1 if len(train) else 0
        return pd.RangeIndex(start=start, stop=start + horizon)


# -----------------------------------------------------------------------------
# 1) Naive 계열 (강의자료 07)
# -----------------------------------------------------------------------------
def forecast_naive_last(train: pd.Series, horizon: int, freq: Optional[str] = None,
                        seasonal_period: Optional[int] = None) -> pd.Series:
    """마지막 값 반복."""
    idx = _future_index(train, horizon, freq)
    return pd.Series(np.repeat(train.iloc[-1], horizon), index=idx, name="naive_last")


def forecast_naive_mean(train: pd.Series, horizon: int, freq: Optional[str] = None,
                        seasonal_period: Optional[int] = None) -> pd.Series:
    """전체 평균값 반복."""
    idx = _future_index(train, horizon, freq)
    return pd.Series(np.repeat(np.nanmean(train.values), horizon), index=idx, name="naive_mean")


def forecast_naive_seasonal(train: pd.Series, horizon: int, freq: Optional[str] = None,
                            seasonal_period: Optional[int] = None) -> pd.Series:
    """직전 계절 값 반복 (Seasonal Naive)."""
    m = seasonal_period or 1
    if m <= 1 or len(train) < m:
        return forecast_naive_last(train, horizon, freq, seasonal_period)
    idx = _future_index(train, horizon, freq)
    last_season = train.iloc[-m:].values
    reps = (horizon + m - 1) // m
    values = np.tile(last_season, reps)[:horizon]
    return pd.Series(values, index=idx, name="naive_seasonal")


def forecast_drift(train: pd.Series, horizon: int, freq: Optional[str] = None,
                   seasonal_period: Optional[int] = None) -> pd.Series:
    """첫 값과 마지막 값을 잇는 직선의 기울기로 외삽."""
    if len(train) < 2:
        return forecast_naive_last(train, horizon, freq, seasonal_period)
    idx = _future_index(train, horizon, freq)
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    base = train.iloc[-1]
    values = base + slope * np.arange(1, horizon + 1)
    return pd.Series(values, index=idx, name="drift")


# -----------------------------------------------------------------------------
# 2) 평활법 (강의자료 08)
# -----------------------------------------------------------------------------
def forecast_ses(train: pd.Series, horizon: int, freq: Optional[str] = None,
                 seasonal_period: Optional[int] = None) -> pd.Series:
    """Simple Exponential Smoothing."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    idx = _future_index(train, horizon, freq)
    model = SimpleExpSmoothing(train.values, initialization_method="estimated").fit(
        optimized=True
    )
    fc = model.forecast(horizon)
    return pd.Series(np.asarray(fc), index=idx, name="ses")


def forecast_holt(train: pd.Series, horizon: int, freq: Optional[str] = None,
                  seasonal_period: Optional[int] = None) -> pd.Series:
    """Holt (추세) 평활법."""
    from statsmodels.tsa.holtwinters import Holt
    idx = _future_index(train, horizon, freq)
    if len(train) < 4:
        return forecast_ses(train, horizon, freq, seasonal_period)
    try:
        model = Holt(train.values, initialization_method="estimated").fit(optimized=True)
        fc = model.forecast(horizon)
    except Exception:
        return forecast_ses(train, horizon, freq, seasonal_period)
    return pd.Series(np.asarray(fc), index=idx, name="holt")


def forecast_holt_winters(train: pd.Series, horizon: int, freq: Optional[str] = None,
                          seasonal_period: Optional[int] = None) -> pd.Series:
    """Holt-Winters (추세+계절성). 계절성이 없으면 Holt로 폴백."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    idx = _future_index(train, horizon, freq)
    m = seasonal_period or 1
    # 계절 패턴을 학습하려면 최소 두 주기가 필요
    if m <= 1 or len(train) < 2 * m:
        return forecast_holt(train, horizon, freq, seasonal_period)
    try:
        # 가법(additive) 우선. 음수가 있으면 곱셈은 불가.
        model = ExponentialSmoothing(
            train.values,
            trend="add",
            seasonal="add",
            seasonal_periods=m,
            initialization_method="estimated",
        ).fit(optimized=True)
        fc = model.forecast(horizon)
    except Exception:
        return forecast_holt(train, horizon, freq, seasonal_period)
    return pd.Series(np.asarray(fc), index=idx, name="holt_winters")


# -----------------------------------------------------------------------------
# 3) ARIMA (강의자료 09)
# -----------------------------------------------------------------------------
def forecast_arima(train: pd.Series, horizon: int, freq: Optional[str] = None,
                   seasonal_period: Optional[int] = None) -> pd.Series:
    """기본 ARIMA(1,1,1)."""
    from statsmodels.tsa.arima.model import ARIMA
    idx = _future_index(train, horizon, freq)
    try:
        model = ARIMA(train.values, order=(1, 1, 1)).fit()
        fc = model.forecast(steps=horizon)
    except Exception:
        return forecast_naive_last(train, horizon, freq, seasonal_period)
    return pd.Series(np.asarray(fc), index=idx, name="arima_1_1_1")


def forecast_auto_arima(train: pd.Series, horizon: int, freq: Optional[str] = None,
                        seasonal_period: Optional[int] = None) -> pd.Series:
    """
    Auto-ARIMA. pmdarima가 있으면 사용하고, 없으면 statsmodels로
    여러 (p,d,q)를 AIC로 그리드 서치.
    """
    idx = _future_index(train, horizon, freq)
    # pmdarima 시도
    try:
        import pmdarima as pm
        m = seasonal_period or 1
        seasonal = m > 1 and len(train) >= 2 * m
        model = pm.auto_arima(
            train.values,
            seasonal=seasonal,
            m=m if seasonal else 1,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
            max_p=3, max_q=3, max_P=2, max_Q=2,
        )
        fc = model.predict(n_periods=horizon)
        return pd.Series(np.asarray(fc), index=idx, name="auto_arima")
    except Exception:
        pass
    # statsmodels로 폴백 - 작은 그리드 서치
    from statsmodels.tsa.arima.model import ARIMA
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    m = ARIMA(train.values, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    try:
        model = ARIMA(train.values, order=best_order).fit()
        fc = model.forecast(steps=horizon)
        return pd.Series(np.asarray(fc), index=idx, name=f"auto_arima_{best_order}")
    except Exception:
        return forecast_naive_last(train, horizon, freq, seasonal_period)


# -----------------------------------------------------------------------------
# 3-2) ARIMA 수동 차수 (강의자료 09)
# -----------------------------------------------------------------------------
def forecast_arima_manual(
    train: pd.Series, horizon: int,
    freq: Optional[str] = None, seasonal_period: Optional[int] = None,
    p: int = 1, d: int = 1, q: int = 1,
) -> pd.Series:
    """사용자가 (p,d,q)를 직접 지정하는 ARIMA."""
    idx = _future_index(train, horizon, freq)
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train.values, order=(p, d, q)).fit()
        fc = model.forecast(steps=horizon)
    except Exception:
        return forecast_naive_last(train, horizon, freq, seasonal_period)
    return pd.Series(np.asarray(fc), index=idx, name=f"arima_manual_{p}_{d}_{q}")


# -----------------------------------------------------------------------------
# 3-3) Holt-Winters 곱셈형 (강의자료 08)
# -----------------------------------------------------------------------------
def forecast_holt_winters_mul(
    train: pd.Series, horizon: int,
    freq: Optional[str] = None, seasonal_period: Optional[int] = None,
) -> pd.Series:
    """Holt-Winters 곱셈형(multiplicative). 양수 데이터에만 적용 가능."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    idx = _future_index(train, horizon, freq)
    m = seasonal_period or 1
    if m <= 1 or len(train) < 2 * m:
        return forecast_holt(train, horizon, freq, seasonal_period)
    # 양수 체크
    if (train.values <= 0).any():
        return forecast_holt_winters(train, horizon, freq, seasonal_period)
    try:
        model = ExponentialSmoothing(
            train.values,
            trend="mul",
            seasonal="mul",
            seasonal_periods=m,
            initialization_method="estimated",
        ).fit(optimized=True)
        fc = model.forecast(horizon)
        return pd.Series(np.asarray(fc), index=idx, name="holt_winters_mul")
    except Exception:
        return forecast_holt_winters(train, horizon, freq, seasonal_period)


# -----------------------------------------------------------------------------
# 3-4) Theta 모델 (M-competition 우승)
# -----------------------------------------------------------------------------
def forecast_theta(
    train: pd.Series, horizon: int,
    freq: Optional[str] = None, seasonal_period: Optional[int] = None,
) -> pd.Series:
    """
    Theta 모델 (Assimakopoulos & Nikolopoulos, 2000).
    SES + 선형 추세의 평균. 단순하지만 매우 강력.
    """
    idx = _future_index(train, horizon, freq)
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel
        m = seasonal_period if seasonal_period and seasonal_period > 1 else None
        # 자동으로 계절성 디시즈널라이즈 후 예측 후 재시즈널라이즈
        if m and len(train) >= 2 * m:
            model = ThetaModel(train.values, period=m).fit()
        else:
            model = ThetaModel(train.values, period=None, deseasonalize=False).fit()
        fc = model.forecast(steps=horizon)
        return pd.Series(np.asarray(fc), index=idx, name="theta")
    except Exception:
        # 폴백: 수동 구현 (SES + drift의 평균)
        try:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            ses_fit = SimpleExpSmoothing(train.values, initialization_method="estimated").fit()
            ses_fc = np.asarray(ses_fit.forecast(horizon))
            # drift
            n = len(train)
            slope = (train.iloc[-1] - train.iloc[0]) / (n - 1) if n > 1 else 0
            drift_fc = train.iloc[-1] + slope * np.arange(1, horizon + 1)
            theta_fc = (ses_fc + drift_fc) / 2.0
            return pd.Series(theta_fc, index=idx, name="theta_manual")
        except Exception:
            return forecast_naive_last(train, horizon, freq, seasonal_period)


# -----------------------------------------------------------------------------
# 3-5) sktime 인터페이스 모델 (강의자료 05 매칭)
# -----------------------------------------------------------------------------
def forecast_sktime_naive(
    train: pd.Series, horizon: int,
    freq: Optional[str] = None, seasonal_period: Optional[int] = None,
) -> pd.Series:
    """
    sktime의 NaiveForecaster를 사용한 예측.
    sktime이 없으면 자체 Naive로 폴백.
    """
    idx = _future_index(train, horizon, freq)
    try:
        from sktime.forecasting.naive import NaiveForecaster
        m = seasonal_period if seasonal_period and seasonal_period > 1 else 1
        if m > 1 and len(train) >= 2 * m:
            forecaster = NaiveForecaster(strategy="last", sp=m)
        else:
            forecaster = NaiveForecaster(strategy="drift")
        # sktime은 정수 인덱스를 좋아함
        train_reset = train.reset_index(drop=True)
        forecaster.fit(train_reset)
        fc = forecaster.predict(fh=list(range(1, horizon + 1)))
        return pd.Series(np.asarray(fc.values), index=idx, name="sktime_naive")
    except Exception:
        return forecast_naive_seasonal(train, horizon, freq, seasonal_period)


# -----------------------------------------------------------------------------
# 4) 모델 레지스트리
# -----------------------------------------------------------------------------
@dataclass
class ModelInfo:
    key: str
    name: str
    fn: Callable
    description: str
    category: str
    needs_statsmodels: bool


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "naive_last": ModelInfo(
        "naive_last", "Naive (마지막값)", forecast_naive_last,
        "마지막 관측값을 그대로 미래에 적용. 모든 모델이 이보다 못하면 의미가 없는 베이스라인.",
        "Naive", False,
    ),
    "naive_mean": ModelInfo(
        "naive_mean", "Naive (평균값)", forecast_naive_mean,
        "전체 평균을 미래에 적용.",
        "Naive", False,
    ),
    "seasonal_naive": ModelInfo(
        "seasonal_naive", "Seasonal Naive", forecast_naive_seasonal,
        "한 주기 전 값을 미래로 가져옴. 계절성이 강한 데이터의 베이스라인.",
        "Naive", False,
    ),
    "drift": ModelInfo(
        "drift", "Drift", forecast_drift,
        "처음과 마지막 값을 잇는 추세선으로 외삽.",
        "Naive", False,
    ),
    "ses": ModelInfo(
        "ses", "단순 지수평활 (SES)", forecast_ses,
        "최근 값에 큰 가중치를 두는 평활. 추세/계절성이 없는 정상 시계열에 적합.",
        "평활법", True,
    ),
    "holt": ModelInfo(
        "holt", "Holt (이중 평활)", forecast_holt,
        "수준과 추세를 함께 추정. 추세가 있는 데이터에 적합.",
        "평활법", True,
    ),
    "holt_winters": ModelInfo(
        "holt_winters", "Holt-Winters (삼중 평활)", forecast_holt_winters,
        "수준+추세+계절성을 모두 추정. 계절성이 있는 데이터에 가장 적합한 평활법.",
        "평활법", True,
    ),
    "holt_winters_mul": ModelInfo(
        "holt_winters_mul", "Holt-Winters (곱셈형)", forecast_holt_winters_mul,
        "곱셈형 추세+계절성. 변동폭이 수준에 비례하는 데이터에 적합 (양수 데이터만).",
        "평활법", True,
    ),
    "theta": ModelInfo(
        "theta", "Theta 모델", forecast_theta,
        "M-competition 우승. SES와 추세선의 평균. 단순하지만 매우 강력.",
        "평활법", True,
    ),
    "arima": ModelInfo(
        "arima", "ARIMA(1,1,1)", forecast_arima,
        "고전적인 ARIMA의 기본 차수. 빠른 비교용.",
        "ARIMA", True,
    ),
    "auto_arima": ModelInfo(
        "auto_arima", "Auto-ARIMA", forecast_auto_arima,
        "AIC를 최소화하는 (p,d,q)를 자동 탐색. (pmdarima가 있으면 계절 SARIMA까지)",
        "ARIMA", True,
    ),
    "sktime_naive": ModelInfo(
        "sktime_naive", "sktime NaiveForecaster", forecast_sktime_naive,
        "sktime 라이브러리의 NaiveForecaster 사용 (강의 자료 05번 매칭).",
        "sktime", False,
    ),
}


def get_model_info(key: str) -> ModelInfo:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {key}")
    return MODEL_REGISTRY[key]


def get_default_models(seasonal_period: Optional[int]) -> List[str]:
    """데이터 특성에 맞는 추천 기본 모델."""
    base = ["naive_last", "drift", "ses", "holt", "auto_arima"]
    if seasonal_period and seasonal_period > 1:
        base.insert(2, "seasonal_naive")
        base.append("holt_winters")
    return base


def run_forecast(
    model_key: str,
    train: pd.Series,
    horizon: int,
    freq: Optional[str] = None,
    seasonal_period: Optional[int] = None,
) -> pd.Series:
    """모델 키로 예측 실행. 실패 시 Naive로 폴백."""
    info = get_model_info(model_key)
    try:
        return info.fn(train, horizon, freq, seasonal_period)
    except Exception as e:
        warnings.warn(f"Model {model_key} failed: {e}. Falling back to Naive.")
        return forecast_naive_last(train, horizon, freq, seasonal_period)


# -----------------------------------------------------------------------------
# 5) 신뢰구간 (Confidence Interval) 추정
# -----------------------------------------------------------------------------
def forecast_with_ci(
    model_key: str,
    train: pd.Series,
    horizon: int,
    freq: Optional[str] = None,
    seasonal_period: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, pd.Series]:
    """
    예측치 + 95% 신뢰구간 반환 (alpha=0.05 → 95% CI).

    statsmodels 기반 모델은 분포 기반 CI를 사용하고,
    Naive 계열은 학습 데이터 잔차의 표준편차로 근사 CI 계산.

    Returns dict: {'mean': Series, 'lower': Series, 'upper': Series}
    """
    info = get_model_info(model_key)
    idx = _future_index(train, horizon, freq)

    # 분포 기반 CI를 직접 제공하는 모델 (statsmodels)
    if model_key in ("ses", "holt", "holt_winters"):
        try:
            from statsmodels.tsa.holtwinters import (
                SimpleExpSmoothing, Holt, ExponentialSmoothing,
            )
            if model_key == "ses":
                fit = SimpleExpSmoothing(train.values, initialization_method="estimated").fit()
            elif model_key == "holt":
                fit = Holt(train.values, initialization_method="estimated").fit()
            else:
                m = seasonal_period or 1
                if m > 1 and len(train) >= 2 * m:
                    fit = ExponentialSmoothing(
                        train.values, trend="add", seasonal="add",
                        seasonal_periods=m, initialization_method="estimated",
                    ).fit()
                else:
                    fit = Holt(train.values, initialization_method="estimated").fit()
            mean_fc = np.asarray(fit.forecast(horizon))
            # 잔차 기반 표준편차로 CI 산정 (점진적 확장)
            sigma = float(np.std(fit.resid)) if hasattr(fit, "resid") else float(np.std(train.values))
            from scipy.stats import norm
            z = norm.ppf(1 - alpha / 2)
            # 시간이 길어질수록 불확실성 커짐: sigma * sqrt(t)
            t_arr = np.arange(1, horizon + 1)
            band = z * sigma * np.sqrt(t_arr)
            lower = mean_fc - band
            upper = mean_fc + band
            return {
                "mean": pd.Series(mean_fc, index=idx, name=info.name),
                "lower": pd.Series(lower, index=idx, name=f"{info.name}_lower"),
                "upper": pd.Series(upper, index=idx, name=f"{info.name}_upper"),
            }
        except Exception:
            pass

    if model_key in ("arima", "auto_arima"):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            order = (1, 1, 1)
            if model_key == "auto_arima":
                # auto_arima 결과를 다시 받아오기 (간단 버전)
                fc = forecast_auto_arima(train, horizon, freq, seasonal_period)
                # ARIMA 모델 객체 자체를 다시 만들어서 CI 추출
                # (간단화를 위해 best_order는 그리드 서치)
                best_aic = np.inf
                best_order = (1, 1, 1)
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                mm = ARIMA(train.values, order=(p, d, q)).fit()
                                if mm.aic < best_aic:
                                    best_aic = mm.aic
                                    best_order = (p, d, q)
                            except Exception:
                                continue
                order = best_order
            model = ARIMA(train.values, order=order).fit()
            forecast_obj = model.get_forecast(steps=horizon)
            mean_fc = np.asarray(forecast_obj.predicted_mean)
            ci = forecast_obj.conf_int(alpha=alpha)
            ci = np.asarray(ci)
            return {
                "mean": pd.Series(mean_fc, index=idx, name=info.name),
                "lower": pd.Series(ci[:, 0], index=idx, name=f"{info.name}_lower"),
                "upper": pd.Series(ci[:, 1], index=idx, name=f"{info.name}_upper"),
            }
        except Exception:
            pass

    # Naive 계열: 학습 데이터의 잔차 (직전 차이) 표준편차로 근사
    mean_fc = run_forecast(model_key, train, horizon, freq, seasonal_period)
    if len(train) >= 2:
        diffs = np.diff(train.values)
        sigma = float(np.std(diffs)) if len(diffs) > 0 else float(np.std(train.values))
    else:
        sigma = 0.0
    try:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
    except Exception:
        z = 1.96
    t_arr = np.arange(1, horizon + 1)
    band = z * sigma * np.sqrt(t_arr)
    lower = mean_fc.values - band
    upper = mean_fc.values + band
    return {
        "mean": mean_fc,
        "lower": pd.Series(lower, index=idx, name=f"{info.name}_lower"),
        "upper": pd.Series(upper, index=idx, name=f"{info.name}_upper"),
    }


# -----------------------------------------------------------------------------
# 6) 롤링 교차검증
# -----------------------------------------------------------------------------
def rolling_cv(
    model_key: str,
    series: pd.Series,
    horizon: int,
    n_splits: int = 5,
    freq: Optional[str] = None,
    seasonal_period: Optional[int] = None,
) -> Dict[str, list]:
    """
    Walk-forward (expanding window) 교차검증.
    각 fold마다 학습 → 예측 → 실제값과 비교 후 평가지표 계산.

    Returns dict with lists for each fold:
        {
            'fold': [1, 2, ...],
            'mae': [...], 'rmse': [...], 'mape': [...],
            'predictions': [Series, ...], 'actuals': [Series, ...],
        }
    """
    from .metrics import mae as _mae, rmse as _rmse, mape as _mape
    n = len(series)
    min_train = max(2 * (seasonal_period or 1), 10)
    if n < min_train + horizon:
        return {"fold": [], "mae": [], "rmse": [], "mape": [],
                "predictions": [], "actuals": []}

    # n_splits 개의 fold가 끝점이 등간격이 되도록 split point를 결정
    end_min = min_train + horizon
    end_max = n
    if n_splits == 1:
        cuts = [end_max]
    else:
        cuts = np.linspace(end_min, end_max, n_splits, dtype=int).tolist()

    results = {"fold": [], "mae": [], "rmse": [], "mape": [],
               "predictions": [], "actuals": []}
    for i, cut in enumerate(cuts):
        train_fold = series.iloc[:cut - horizon]
        test_fold = series.iloc[cut - horizon:cut]
        if len(train_fold) < min_train or len(test_fold) < horizon:
            continue
        try:
            fc = run_forecast(model_key, train_fold, horizon, freq, seasonal_period)
            fc_aligned = pd.Series(fc.values[:len(test_fold)], index=test_fold.index)
            results["fold"].append(i + 1)
            results["mae"].append(_mae(test_fold, fc_aligned))
            results["rmse"].append(_rmse(test_fold, fc_aligned))
            results["mape"].append(_mape(test_fold, fc_aligned))
            results["predictions"].append(fc_aligned)
            results["actuals"].append(test_fold)
        except Exception:
            continue
    return results


# -----------------------------------------------------------------------------
# 7) 앙상블 (단순 평균 / 가중 평균)
# -----------------------------------------------------------------------------
def ensemble_simple_mean(forecasts: Dict[str, pd.Series]) -> pd.Series:
    """여러 모델 예측의 단순 평균."""
    if not forecasts:
        return pd.Series(dtype=float)
    df = pd.DataFrame({k: v.values for k, v in forecasts.items()})
    mean_vals = df.mean(axis=1).values
    # 첫 번째 forecast의 인덱스 사용
    first = next(iter(forecasts.values()))
    return pd.Series(mean_vals, index=first.index, name="ensemble_mean")


def ensemble_weighted(
    forecasts: Dict[str, pd.Series],
    weights: Dict[str, float],
) -> pd.Series:
    """
    가중 평균 앙상블.
    weights는 {model_name: weight} 형태. 자동으로 정규화됨.
    """
    if not forecasts:
        return pd.Series(dtype=float)
    # 정규화
    total_w = sum(weights.get(k, 0.0) for k in forecasts.keys())
    if total_w <= 0:
        return ensemble_simple_mean(forecasts)
    weighted_sum = None
    first = next(iter(forecasts.values()))
    weighted_sum = np.zeros(len(first))
    for k, fc in forecasts.items():
        w = weights.get(k, 0.0) / total_w
        weighted_sum += np.asarray(fc.values) * w
    return pd.Series(weighted_sum, index=first.index, name="ensemble_weighted")


def ensemble_inverse_rmse(
    forecasts: Dict[str, pd.Series],
    rmse_dict: Dict[str, float],
) -> pd.Series:
    """
    검증 RMSE의 역수를 가중치로 사용 (RMSE가 낮을수록 큰 가중치).
    """
    weights = {}
    for k in forecasts.keys():
        r = rmse_dict.get(k, float("inf"))
        if r > 0 and not np.isinf(r) and not np.isnan(r):
            weights[k] = 1.0 / r
        else:
            weights[k] = 0.0
    return ensemble_weighted(forecasts, weights)


# -----------------------------------------------------------------------------
# 8) 다중 horizon 평가 (단기/중기/장기 분해)
# -----------------------------------------------------------------------------
def evaluate_multi_horizon(
    test: pd.Series,
    forecast: pd.Series,
    breakpoints: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    예측 시평을 짧게/중간/길게 나누어 각 구간의 성능을 따로 평가.
    예: horizon=12면 1-4 / 5-8 / 9-12로 분할.
    """
    from .metrics import all_metrics
    n = len(test)
    if breakpoints is None:
        # 기본: 1/3, 2/3 지점에서 분할
        b1 = max(1, n // 3)
        b2 = max(b1 + 1, 2 * n // 3)
        breakpoints = [b1, b2]
    breakpoints = sorted(set(breakpoints))
    breakpoints = [b for b in breakpoints if 0 < b < n]

    splits = []
    prev = 0
    for b in breakpoints:
        splits.append((prev, b))
        prev = b
    splits.append((prev, n))

    labels = ["단기", "중기", "장기"]
    if len(splits) > 3:
        labels = [f"구간{i+1}" for i in range(len(splits))]
    elif len(splits) == 2:
        labels = ["단기", "장기"]
    elif len(splits) == 1:
        labels = ["전체"]

    results = {}
    for (start, end), label in zip(splits, labels):
        t_slice = test.iloc[start:end]
        f_slice = forecast.iloc[start:end]
        if len(t_slice) > 0:
            results[f"{label} (t={start+1}~{end})"] = all_metrics(t_slice, f_slice)
    return results
