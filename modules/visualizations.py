"""
visualizations.py
Plotly 기반 인터랙티브 시각화 함수 모음.
대시보드의 모든 차트를 여기서 생성한다.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 색상 팔레트 (강의자료의 깔끔한 톤에 맞춰 선정)
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


# -----------------------------------------------------------------------------
# 1) 원시 시계열 + 통계 (EDA)
# -----------------------------------------------------------------------------
def plot_raw_series(ts: pd.Series, title: str = "원시 시계열") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        mode="lines",
        name=ts.name or "value",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="시점: %{x}<br>값: %{y:.4f}<extra></extra>",
    ))
    # 이동평균 (smoothing 보조선)
    if len(ts) >= 7:
        window = max(3, min(len(ts) // 10, 30))
        ma = ts.rolling(window=window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=ma.index, y=ma.values,
            mode="lines",
            name=f"이동평균({window})",
            line=dict(color="#ff7f0e", width=1.5, dash="dot"),
            opacity=0.8,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="시점",
        yaxis_title="값",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# -----------------------------------------------------------------------------
# 2) 학습/검증 분할 시각화
# -----------------------------------------------------------------------------
def plot_train_test(train: pd.Series, test: pd.Series, title: str = "학습/검증 분할") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index, y=train.values, mode="lines",
        name=f"학습 ({len(train)})", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=test.index, y=test.values, mode="lines+markers",
        name=f"검증 ({len(test)})", line=dict(color="#d62728", width=2),
        marker=dict(size=6),
    ))
    # 분할 시점에 수직선
    if len(train) > 0:
        split_x = train.index[-1]
        fig.add_vline(x=split_x, line_width=1.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="시점", yaxis_title="값",
        template="plotly_white",
        height=380,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# 3) 모델별 예측 곡선 비교
# -----------------------------------------------------------------------------
def plot_forecasts(
    train: pd.Series,
    test: Optional[pd.Series],
    forecasts: Dict[str, pd.Series],
    title: str = "모델별 예측 비교",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index, y=train.values, mode="lines",
        name="학습 데이터", line=dict(color="#7f7f7f", width=1.5),
        opacity=0.85,
    ))
    if test is not None and len(test) > 0:
        fig.add_trace(go.Scatter(
            x=test.index, y=test.values, mode="lines+markers",
            name="실제값 (검증)", line=dict(color="black", width=2.5),
            marker=dict(size=7, symbol="circle"),
        ))

    for i, (model_name, fc) in enumerate(forecasts.items()):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.values, mode="lines+markers",
            name=model_name,
            line=dict(color=color, width=2, dash="solid"),
            marker=dict(size=5),
            hovertemplate=f"{model_name}<br>시점: %{{x}}<br>예측: %{{y:.4f}}<extra></extra>",
        ))

    if len(train) > 0:
        fig.add_vline(x=train.index[-1], line_width=1.5, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="시점", yaxis_title="값",
        template="plotly_white",
        height=480,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return fig


# -----------------------------------------------------------------------------
# 4) 평가지표 비교 차트
# -----------------------------------------------------------------------------
def plot_metric_comparison(metrics_df: pd.DataFrame, metric: str = "RMSE") -> go.Figure:
    """모델별 단일 지표 비교 막대 그래프."""
    if metric not in metrics_df.columns:
        return go.Figure()
    df = metrics_df[[metric]].dropna().sort_values(metric)
    if metric == "R2":
        df = df.sort_values(metric, ascending=False)  # R2는 큰 게 좋음
    best_idx = df.index[0]
    colors = [
        "#2ca02c" if i == best_idx else PALETTE[j % len(PALETTE)]
        for j, i in enumerate(df.index)
    ]
    fig = go.Figure(go.Bar(
        x=df.index, y=df[metric].values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in df[metric].values],
        textposition="outside",
        hovertemplate="%{x}<br>" + metric + ": %{y:.4f}<extra></extra>",
    ))
    direction = "낮을수록 좋음" if metric != "R2" else "높을수록 좋음"
    fig.update_layout(
        title=f"{metric} 비교 ({direction})",
        xaxis_title="모델", yaxis_title=metric,
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def plot_metrics_radar(metrics_df: pd.DataFrame) -> go.Figure:
    """
    여러 지표를 한 번에 비교하는 레이더 차트.
    각 지표별로 0~1로 정규화 (작은 게 좋은 지표는 뒤집어서).
    """
    metrics_to_show = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE"]
    available = [m for m in metrics_to_show if m in metrics_df.columns]
    if len(available) < 3:
        return go.Figure()

    df = metrics_df[available].copy()
    # 정규화: 각 지표의 최대값으로 나눠 0~1, 그 후 1-x (작은 게 좋게)
    df_norm = df.copy()
    for col in df_norm.columns:
        col_max = df_norm[col].max()
        if col_max > 0:
            df_norm[col] = 1.0 - (df_norm[col] / col_max)
        else:
            df_norm[col] = 1.0

    fig = go.Figure()
    for i, model in enumerate(df_norm.index):
        values = df_norm.loc[model].tolist()
        values.append(values[0])  # 닫힘
        cats = available + [available[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=cats,
            fill="toself",
            name=model,
            line=dict(color=PALETTE[i % len(PALETTE)]),
            opacity=0.55,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="모델 종합 성능 (외곽일수록 우수)",
        height=460,
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=True,
    )
    return fig


# -----------------------------------------------------------------------------
# 5) 잔차 분석
# -----------------------------------------------------------------------------
def plot_residuals(test: pd.Series, forecast: pd.Series, model_name: str) -> go.Figure:
    """잔차 시계열, 잔차 히스토그램, 산점도(실제 vs 예측)를 한 그림에."""
    res = test.values - forecast.values
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "잔차 시계열", "잔차 분포",
            "실제 vs 예측", "잔차 누적합",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    # 1) 잔차 시계열
    fig.add_trace(go.Scatter(
        x=test.index, y=res, mode="lines+markers",
        name="잔차", line=dict(color="#d62728"),
        showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # 2) 잔차 히스토그램
    fig.add_trace(go.Histogram(
        x=res, nbinsx=max(10, len(res) // 3),
        marker_color="#1f77b4",
        showlegend=False,
        name="히스토그램",
    ), row=1, col=2)

    # 3) 실제 vs 예측 산점도 + y=x
    fig.add_trace(go.Scatter(
        x=test.values, y=forecast.values,
        mode="markers",
        marker=dict(color="#2ca02c", size=8),
        showlegend=False,
        name="actual vs pred",
    ), row=2, col=1)
    lo = min(np.min(test.values), np.min(forecast.values))
    hi = max(np.max(test.values), np.max(forecast.values))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(dash="dash", color="gray"),
        showlegend=False,
    ), row=2, col=1)

    # 4) 잔차 누적합 (편향 시각화)
    fig.add_trace(go.Scatter(
        x=test.index, y=np.cumsum(res),
        mode="lines+markers",
        line=dict(color="#9467bd"),
        showlegend=False, name="잔차 누적합",
    ), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

    fig.update_layout(
        title=f"잔차 분석 - {model_name}",
        height=600, template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    fig.update_xaxes(title_text="시점", row=1, col=1)
    fig.update_yaxes(title_text="잔차", row=1, col=1)
    fig.update_xaxes(title_text="잔차 값", row=1, col=2)
    fig.update_yaxes(title_text="빈도", row=1, col=2)
    fig.update_xaxes(title_text="실제값", row=2, col=1)
    fig.update_yaxes(title_text="예측값", row=2, col=1)
    fig.update_xaxes(title_text="시점", row=2, col=2)
    fig.update_yaxes(title_text="누적 잔차", row=2, col=2)
    return fig


# -----------------------------------------------------------------------------
# 6) 미래 예측 (학습+테스트 전체로 학습하여 미래 시평까지 예측)
# -----------------------------------------------------------------------------
def plot_future_forecast(
    full: pd.Series,
    forecast: pd.Series,
    model_name: str,
    horizon: int,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=full.index, y=full.values, mode="lines",
        name="과거 관측값", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
        mode="lines+markers",
        name=f"미래 예측 ({horizon}기간)",
        line=dict(color="#d62728", width=2.5),
        marker=dict(size=7, symbol="diamond"),
    ))
    if len(full) > 0:
        fig.add_vline(x=full.index[-1], line_width=1.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"미래 예측 - {model_name}",
        xaxis_title="시점", yaxis_title="값",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# 7) 신뢰구간 포함 미래 예측
# -----------------------------------------------------------------------------
def plot_future_forecast_with_ci(
    full: pd.Series,
    mean: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    model_name: str,
    horizon: int,
    ci_label: str = "95% 신뢰구간",
) -> go.Figure:
    fig = go.Figure()
    # 과거 데이터
    fig.add_trace(go.Scatter(
        x=full.index, y=full.values, mode="lines",
        name="과거 관측값", line=dict(color="#1f77b4", width=2),
    ))
    # CI 밴드 (upper → lower 순으로 닫힘)
    fig.add_trace(go.Scatter(
        x=list(upper.index) + list(lower.index[::-1]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself",
        fillcolor="rgba(214, 39, 40, 0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        name=ci_label,
        hoverinfo="skip",
    ))
    # 평균 예측선
    fig.add_trace(go.Scatter(
        x=mean.index, y=mean.values,
        mode="lines+markers",
        name=f"미래 예측 (점추정)",
        line=dict(color="#d62728", width=2.5),
        marker=dict(size=7, symbol="diamond"),
    ))
    if len(full) > 0:
        fig.add_vline(x=full.index[-1], line_width=1.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"미래 예측 + {ci_label} - {model_name}",
        xaxis_title="시점", yaxis_title="값",
        template="plotly_white",
        height=460,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# 8) 시계열 분해 (Trend / Seasonal / Resid)
# -----------------------------------------------------------------------------
def plot_decomposition(decomp: dict, title: str = "시계열 분해 (Decomposition)") -> go.Figure:
    """
    decomp: {'observed', 'trend', 'seasonal', 'resid'} 형태의 dict
    """
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("관측값 (Observed)", "추세 (Trend)", "계절성 (Seasonal)", "잔차 (Residual)"),
        vertical_spacing=0.06,
    )
    fig.add_trace(go.Scatter(
        x=decomp["observed"].index, y=decomp["observed"].values,
        mode="lines", line=dict(color="#1f77b4", width=2),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=decomp["trend"].index, y=decomp["trend"].values,
        mode="lines", line=dict(color="#ff7f0e", width=2),
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=decomp["seasonal"].index, y=decomp["seasonal"].values,
        mode="lines", line=dict(color="#2ca02c", width=1.5),
        showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=decomp["resid"].index, y=decomp["resid"].values,
        mode="lines+markers",
        line=dict(color="#d62728", width=1),
        marker=dict(size=3),
        showlegend=False,
    ), row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    fig.update_layout(
        title=title,
        height=620,
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# 9) ACF / PACF
# -----------------------------------------------------------------------------
def plot_acf_pacf(
    acf_vals: np.ndarray,
    pacf_vals: np.ndarray,
    conf_band: float,
    title: str = "자기상관 함수 (ACF / PACF)",
) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ACF (자기상관)", "PACF (편자기상관)"),
    )
    if len(acf_vals) > 0:
        lags_a = list(range(len(acf_vals)))
        fig.add_trace(go.Bar(
            x=lags_a, y=acf_vals,
            marker_color="#1f77b4",
            showlegend=False,
            name="ACF",
        ), row=1, col=1)
        # 신뢰밴드
        fig.add_hline(y=conf_band, line_dash="dot", line_color="red", row=1, col=1)
        fig.add_hline(y=-conf_band, line_dash="dot", line_color="red", row=1, col=1)
    if len(pacf_vals) > 0:
        lags_p = list(range(len(pacf_vals)))
        fig.add_trace(go.Bar(
            x=lags_p, y=pacf_vals,
            marker_color="#ff7f0e",
            showlegend=False,
            name="PACF",
        ), row=1, col=2)
        fig.add_hline(y=conf_band, line_dash="dot", line_color="red", row=1, col=2)
        fig.add_hline(y=-conf_band, line_dash="dot", line_color="red", row=1, col=2)

    fig.update_layout(
        title=title,
        height=380,
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="상관계수", row=1, col=1)
    fig.update_yaxes(title_text="상관계수", row=1, col=2)
    return fig


# -----------------------------------------------------------------------------
# 10) 롤링 교차검증 결과
# -----------------------------------------------------------------------------
def plot_rolling_cv(cv_results: dict, model_name: str) -> go.Figure:
    """
    각 fold별 RMSE/MAE/MAPE를 막대그래프로 비교.
    """
    folds = cv_results.get("fold", [])
    if not folds:
        return go.Figure()
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE per fold", "RMSE per fold", "MAPE per fold (%)"),
    )
    fig.add_trace(go.Bar(
        x=folds, y=cv_results["mae"], marker_color="#1f77b4",
        text=[f"{v:.2f}" for v in cv_results["mae"]],
        textposition="outside", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=folds, y=cv_results["rmse"], marker_color="#ff7f0e",
        text=[f"{v:.2f}" for v in cv_results["rmse"]],
        textposition="outside", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=folds, y=cv_results["mape"], marker_color="#2ca02c",
        text=[f"{v:.1f}" for v in cv_results["mape"]],
        textposition="outside", showlegend=False,
    ), row=1, col=3)

    # 평균선 추가
    if cv_results["rmse"]:
        avg_rmse = np.mean(cv_results["rmse"])
        fig.add_hline(y=avg_rmse, line_dash="dash", line_color="gray",
                      row=1, col=2)
        # annotation은 별도로 (plotly 호환성을 위해)
        fig.add_annotation(
            xref="x2", yref="y2",
            x=folds[-1] if folds else 1, y=avg_rmse,
            text=f"평균 {avg_rmse:.2f}",
            showarrow=False,
            font=dict(color="gray", size=11),
            yshift=10,
        )

    fig.update_layout(
        title=f"롤링 교차검증 결과 - {model_name} ({len(folds)} folds)",
        height=400,
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    fig.update_xaxes(title_text="Fold", row=1, col=1)
    fig.update_xaxes(title_text="Fold", row=1, col=2)
    fig.update_xaxes(title_text="Fold", row=1, col=3)
    return fig


# -----------------------------------------------------------------------------
# 11) 이상치 표시 (원시 시계열 + 빨간 점)
# -----------------------------------------------------------------------------
def plot_with_outliers(
    ts: pd.Series, outlier_mask: pd.Series,
    title: str = "이상치 탐지 결과",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values, mode="lines",
        name="시계열", line=dict(color="#1f77b4", width=2),
    ))
    if outlier_mask is not None and outlier_mask.sum() > 0:
        out_idx = ts.index[outlier_mask]
        out_vals = ts.values[outlier_mask.values]
        fig.add_trace(go.Scatter(
            x=out_idx, y=out_vals, mode="markers",
            name=f"이상치 ({int(outlier_mask.sum())}개)",
            marker=dict(color="red", size=11, symbol="x", line=dict(width=2)),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="시점", yaxis_title="값",
        template="plotly_white", height=380,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# 12) 변동점(Change point) 표시
# -----------------------------------------------------------------------------
def plot_change_points(
    ts: pd.Series, change_points: list,
    title: str = "변동점(Change Point) 탐지",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values, mode="lines",
        name="시계열", line=dict(color="#1f77b4", width=2),
    ))
    # plotly의 add_vline + annotation_text 조합이 datetime 인덱스에서
    # TypeError를 일으키므로, 줄과 텍스트를 분리해서 그린다.
    if len(ts) > 0:
        y_max = float(np.nanmax(ts.values))
        y_min = float(np.nanmin(ts.values))
        y_top = y_max + (y_max - y_min) * 0.05
    else:
        y_top = 0.0
    for i, cp in enumerate(change_points):
        # 줄만 (annotation 인자 없이)
        fig.add_vline(
            x=cp, line_width=2, line_dash="dash", line_color="orange",
        )
        # 텍스트는 별도 annotation
        fig.add_annotation(
            x=cp, y=y_top,
            text=f"CP{i+1}",
            showarrow=False,
            font=dict(color="orange", size=12),
            yanchor="bottom",
        )
    fig.update_layout(
        title=title + (f" ({len(change_points)}개 발견)" if change_points else " (없음)"),
        xaxis_title="시점", yaxis_title="값",
        template="plotly_white", height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# 13) QQ plot
# -----------------------------------------------------------------------------
def plot_qq(theoretical: np.ndarray, sample: np.ndarray, title: str = "잔차 QQ plot") -> go.Figure:
    fig = go.Figure()
    if len(theoretical) == 0:
        return fig
    # 정규성 정도를 확인하기 위해 표본을 표준화
    s_mean = float(np.mean(sample))
    s_std = float(np.std(sample))
    if s_std > 0:
        sample_std = (sample - s_mean) / s_std
    else:
        sample_std = sample
    # 점들
    fig.add_trace(go.Scatter(
        x=theoretical, y=sample_std,
        mode="markers",
        marker=dict(color="#1f77b4", size=7),
        name="표본 분위수",
        showlegend=False,
    ))
    # y=x 기준선
    lo = float(min(theoretical.min(), sample_std.min()))
    hi = float(max(theoretical.max(), sample_std.max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(dash="dash", color="red"),
        name="이상적 직선",
        showlegend=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="이론 분위수 (정규분포)",
        yaxis_title="표본 분위수 (표준화)",
        template="plotly_white", height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# 14) 차분 시각화 (ARIMA d 결정)
# -----------------------------------------------------------------------------
def plot_differencing(diffs: dict, title: str = "차분 단계별 시계열") -> go.Figure:
    n_d = len(diffs)
    fig = make_subplots(
        rows=n_d, cols=1, shared_xaxes=False,
        subplot_titles=tuple(
            f"차분 d={d}" + (" (원본)" if d == 0 else "")
            for d in sorted(diffs.keys())
        ),
        vertical_spacing=0.10,
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, d in enumerate(sorted(diffs.keys())):
        s = diffs[d]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=colors[i % len(colors)], width=1.5),
            showlegend=False,
        ), row=i + 1, col=1)
        if d > 0:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=i + 1, col=1)
    fig.update_layout(
        title=title,
        height=200 * n_d,
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# 15) 다중 horizon 평가
# -----------------------------------------------------------------------------
def plot_multi_horizon(multi_results: dict, model_name: str) -> go.Figure:
    """단기/중기/장기 RMSE/MAPE 비교 막대그래프."""
    if not multi_results:
        return go.Figure()
    labels = list(multi_results.keys())
    rmse_vals = [multi_results[l].get("RMSE", float("nan")) for l in labels]
    mape_vals = [multi_results[l].get("MAPE", float("nan")) for l in labels]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("RMSE per 구간", "MAPE per 구간 (%)"),
    )
    fig.add_trace(go.Bar(
        x=labels, y=rmse_vals, marker_color="#1f77b4",
        text=[f"{v:.2f}" for v in rmse_vals],
        textposition="outside", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=labels, y=mape_vals, marker_color="#ff7f0e",
        text=[f"{v:.2f}" for v in mape_vals],
        textposition="outside", showlegend=False,
    ), row=1, col=2)
    fig.update_layout(
        title=f"시평 구간별 성능 - {model_name}",
        height=380, template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# 16) 데이터 특성 카드 (게이지 차트)
# -----------------------------------------------------------------------------
def plot_feature_gauges(features: dict) -> go.Figure:
    """추세 강도, 계절성 강도, 노이즈 비율, 자기상관을 게이지로."""
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}, {"type": "indicator"},
                {"type": "indicator"}, {"type": "indicator"}]],
    )
    items = [
        ("추세 강도", features.get("trend_strength", 0.0), "#1f77b4"),
        ("계절성 강도", features.get("seasonal_strength", 0.0), "#2ca02c"),
        ("노이즈 비율", features.get("noise_ratio", 0.0), "#d62728"),
        ("자기상관(lag1)", abs(features.get("autocorr_lag1", 0.0)), "#9467bd"),
    ]
    for i, (label, val, color) in enumerate(items):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=float(val),
            title={"text": label, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 0.3], "color": "#f0f0f0"},
                    {"range": [0.3, 0.7], "color": "#dcdcdc"},
                    {"range": [0.7, 1.0], "color": "#c0c0c0"},
                ],
            },
            number={"valueformat": ".2f"},
        ), row=1, col=i + 1)
    fig.update_layout(
        height=240, template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# -----------------------------------------------------------------------------
# 17) 백테스팅 시각화 (과거 시점에서 했다면 어땠을지)
# -----------------------------------------------------------------------------
def plot_backtest(
    full: pd.Series,
    backtest_results: list,
    title: str = "백테스팅: 과거 시점 예측 vs 실제",
) -> go.Figure:
    """
    backtest_results: list of dict
        [{'cutoff': index, 'forecast': Series, 'actual': Series}, ...]
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=full.index, y=full.values, mode="lines",
        name="실제 데이터",
        line=dict(color="#7f7f7f", width=2),
    ))
    palette = PALETTE
    for i, bt in enumerate(backtest_results):
        fc = bt["forecast"]
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.values, mode="lines+markers",
            name=f"예측 #{i+1} (시점 {bt['cutoff']}부터)",
            line=dict(color=palette[i % len(palette)], width=1.8, dash="dot"),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="시점", yaxis_title="값",
        template="plotly_white", height=460,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
