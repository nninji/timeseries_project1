"""
metrics.py
시계열 예측 평가 지표 (강의자료 07_시계열예측개요 기반).

지원 지표:
- MAE  : Mean Absolute Error
- MSE  : Mean Squared Error
- RMSE : Root Mean Squared Error
- MAPE : Mean Absolute Percentage Error (%)
- sMAPE: Symmetric MAPE (%)
- R^2  : Coefficient of determination
- MASE : Mean Absolute Scaled Error (계절성 고려)
- Bias : 잔차의 평균 (예측이 한쪽으로 치우쳤는지)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _align(y_true, y_pred) -> tuple:
    """예측값과 실제값을 numpy 배열로 정렬."""
    yt = np.asarray(pd.Series(y_true).values, dtype=float)
    yp = np.asarray(pd.Series(y_pred).values, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def mse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (%). 0인 실제값은 제외."""
    yt, yp = _align(y_true, y_pred)
    nz = np.abs(yt) > 1e-12
    if nz.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100.0)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE (%). 분모 = (|true|+|pred|)/2."""
    yt, yp = _align(y_true, y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    nz = denom > 1e-12
    if nz.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(yt[nz] - yp[nz]) / denom[nz]) * 100.0)


def r2_score(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) < 2:
        return float("nan")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mase(y_true, y_pred, y_train=None, m: int = 1) -> float:
    """
    Mean Absolute Scaled Error.
    계절성(m) 단위 Naive 예측 대비 얼마나 잘하는지를 보여주는 지표.
    1보다 작으면 Naive보다 우수.
    """
    yt, yp = _align(y_true, y_pred)
    if y_train is None or len(y_train) <= m:
        # train 없을 때는 test 자체로 대체 (수업에서는 일반적인 정의 사용)
        ref = yt
    else:
        ref = np.asarray(pd.Series(y_train).dropna().values, dtype=float)
    if len(ref) <= m:
        return float("nan")
    naive_diff = np.abs(ref[m:] - ref[:-m])
    scale = np.mean(naive_diff)
    if scale < 1e-12:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)) / scale)


def bias(y_true, y_pred) -> float:
    """예측 잔차의 평균 (양수=과소예측, 음수=과대예측)."""
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(yt - yp))


def all_metrics(
    y_true,
    y_pred,
    y_train=None,
    seasonal_period: int = 1,
) -> Dict[str, float]:
    """모든 지표를 한 번에 계산."""
    return {
        "MAE":   mae(y_true, y_pred),
        "MSE":   mse(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2":    r2_score(y_true, y_pred),
        "MASE":  mase(y_true, y_pred, y_train=y_train, m=max(1, seasonal_period)),
        "Bias":  bias(y_true, y_pred),
    }


def metric_descriptions() -> Dict[str, str]:
    """대시보드 툴팁용 지표 설명."""
    return {
        "MAE":   "평균 절대 오차. 작을수록 좋음. 단위가 원시 데이터와 동일.",
        "MSE":   "평균 제곱 오차. 큰 오차에 민감. 작을수록 좋음.",
        "RMSE":  "MSE의 제곱근. 단위가 원시 데이터와 동일하며 가장 직관적.",
        "MAPE":  "평균 절대 백분율 오차(%). 스케일에 무관하지만 0 근처 값에 약함.",
        "sMAPE": "대칭형 MAPE(%). MAPE의 단점을 보완. 0~200% 범위.",
        "R2":    "결정계수. 1에 가까울수록 좋음 (음수 가능). 분산 설명력.",
        "MASE":  "Naive 예측 대비 상대 성능. 1 미만이면 Naive보다 우수.",
        "Bias":  "잔차 평균. 0에 가까울수록 좋음. 양수=과소예측, 음수=과대예측.",
    }
