"""
data_handler.py
시계열 CSV 파일의 자동 인식, 검증, 전처리를 담당하는 모듈.

지원하는 CSV 형태:
1) 두 컬럼: [날짜, 값]  ─ 가장 일반적
2) 한 컬럼: [값]        ─ 인덱스가 자동 생성됨
3) 인덱스에 날짜, 첫 컬럼에 값
4) 여러 컬럼 중 사용자가 선택 (단변량으로 처리)
"""
from __future__ import annotations

import io
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# CSV 로딩
# -----------------------------------------------------------------------------
def read_csv_robust(file_buffer) -> pd.DataFrame:
    """다양한 인코딩/구분자에 대응하여 CSV를 견고하게 읽어들인다."""
    raw_bytes = file_buffer.read() if hasattr(file_buffer, "read") else file_buffer
    if isinstance(raw_bytes, str):
        raw_bytes = raw_bytes.encode("utf-8")

    # 인코딩 후보 순회
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]
    text = None
    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("CSV 파일의 인코딩을 인식할 수 없습니다.")

    # 구분자 자동 추론 (sniff)
    sample = text[:4096]
    sep_candidates = [",", ";", "\t", "|"]
    counts = {s: sample.count(s) for s in sep_candidates}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","

    df = pd.read_csv(io.StringIO(text), sep=sep)
    # 모든 컬럼명을 문자열로 정규화
    df.columns = [str(c).strip() for c in df.columns]
    return df


# -----------------------------------------------------------------------------
# 컬럼 자동 감지
# -----------------------------------------------------------------------------
def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """날짜로 해석 가능한 컬럼 중 가장 가능성이 높은 것을 반환."""
    best_col = None
    best_ratio = 0.0
    for col in df.columns:
        s = df[col]
        # 이미 날짜형이면 즉시 반환
        if pd.api.types.is_datetime64_any_dtype(s):
            return col
        # 숫자형은 보통 값 컬럼이므로 패스
        if pd.api.types.is_numeric_dtype(s):
            continue
        # 문자열을 날짜로 파싱 시도
        try:
            parsed = pd.to_datetime(s, errors="coerce")
            ratio = parsed.notna().mean()
            if ratio > 0.9 and ratio > best_ratio:
                best_ratio = ratio
                best_col = col
        except Exception:
            continue
    return best_col


def detect_value_column(df: pd.DataFrame, date_col: Optional[str]) -> Optional[str]:
    """수치형 컬럼 중 단변량 값으로 사용할 컬럼을 추천."""
    numeric_cols = [
        c for c in df.columns
        if c != date_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        # 수치 변환 시도
        for c in df.columns:
            if c == date_col:
                continue
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().mean() > 0.9:
                return c
        return None
    # 결측이 가장 적은 컬럼 우선
    return min(numeric_cols, key=lambda c: df[c].isna().sum())


# -----------------------------------------------------------------------------
# 시계열 변환
# -----------------------------------------------------------------------------
def to_timeseries(
    df: pd.DataFrame,
    date_col: Optional[str],
    value_col: str,
) -> pd.Series:
    """선택된 컬럼들로 pandas Series (DatetimeIndex) 생성."""
    if date_col is not None:
        idx = pd.to_datetime(df[date_col], errors="coerce")
        values = pd.to_numeric(df[value_col], errors="coerce")
        ts = pd.Series(values.values, index=idx, name=value_col)
        ts = ts[ts.index.notna()]  # 잘못 파싱된 날짜 제거
        ts = ts.sort_index()
    else:
        # 날짜 컬럼이 없으면 RangeIndex 그대로 사용 (PeriodIndex로 대체)
        values = pd.to_numeric(df[value_col], errors="coerce")
        ts = pd.Series(values.values, name=value_col)

    # 중복 인덱스가 있으면 평균으로 집계
    if ts.index.has_duplicates:
        ts = ts.groupby(ts.index).mean()

    return ts


def infer_frequency(ts: pd.Series) -> Tuple[Optional[str], Optional[int]]:
    """
    시계열의 빈도(freq)와 계절성 주기(seasonal period)를 추론.

    Returns
    -------
    (freq_str, seasonal_period)
        freq_str: pandas offset alias (예: 'D', 'M', 'MS', 'Q', 'Y', 'H')
        seasonal_period: 일반적인 계절성 주기 (예: 7, 12, 4, 24)
    """
    if not isinstance(ts.index, pd.DatetimeIndex) or len(ts) < 3:
        return None, None

    # pandas의 자동 추론을 먼저 시도
    inferred = pd.infer_freq(ts.index)
    if inferred is None:
        # 인덱스 차이의 중간값으로 추정
        diffs = ts.index.to_series().diff().dropna()
        if len(diffs) == 0:
            return None, None
        median_diff = diffs.median()
        days = median_diff.days + median_diff.seconds / 86400.0
        if days < 1.5 / 24:
            inferred = "h"
        elif 0.9 <= days <= 1.1:
            inferred = "D"
        elif 6 <= days <= 8:
            inferred = "W"
        elif 28 <= days <= 31:
            inferred = "MS"
        elif 89 <= days <= 92:
            inferred = "QS"
        elif 360 <= days <= 370:
            inferred = "YS"
        else:
            inferred = None

    # 계절성 주기 매핑
    seasonal_map = {
        "h": 24, "H": 24,
        "D": 7, "B": 5,
        "W": 52, "W-SUN": 52, "W-MON": 52,
        "M": 12, "MS": 12, "ME": 12,
        "Q": 4, "QS": 4, "QE": 4,
        "Y": 1, "YS": 1, "YE": 1, "A": 1, "AS": 1,
    }
    seasonal_period = None
    if inferred is not None:
        # alias의 첫 글자 또는 prefix를 매칭
        for key, value in seasonal_map.items():
            if inferred.startswith(key):
                seasonal_period = value
                break

    return inferred, seasonal_period


def fill_missing(ts: pd.Series, method: str = "interpolate") -> pd.Series:
    """결측치를 처리. 시계열에서 일반적으로 권장되는 방법들 제공."""
    if ts.isna().sum() == 0:
        return ts
    if method == "interpolate":
        # 선형 보간 후 양쪽 끝 채우기
        out = ts.interpolate(method="linear", limit_direction="both")
    elif method == "ffill":
        out = ts.ffill().bfill()
    elif method == "mean":
        out = ts.fillna(ts.mean())
    elif method == "drop":
        out = ts.dropna()
    else:
        out = ts.interpolate(method="linear", limit_direction="both")
    return out


def regularize_index(ts: pd.Series, freq: Optional[str]) -> pd.Series:
    """불규칙한 시점을 균일한 freq로 재샘플링 (필요시)."""
    if freq is None or not isinstance(ts.index, pd.DatetimeIndex):
        return ts
    try:
        full_idx = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)
        ts2 = ts.reindex(full_idx)
        # 새로 생긴 결측은 보간으로 채움
        if ts2.isna().any():
            ts2 = ts2.interpolate(method="linear", limit_direction="both")
        return ts2
    except Exception:
        return ts


# -----------------------------------------------------------------------------
# 한 번에 처리하는 헬퍼
# -----------------------------------------------------------------------------
def auto_process(
    df: pd.DataFrame,
    date_col_override: Optional[str] = None,
    value_col_override: Optional[str] = None,
    fill_method: str = "interpolate",
    regularize: bool = True,
) -> dict:
    """
    DataFrame → 분석 가능한 시계열로 자동 변환.

    Returns dict with keys:
        ts (pd.Series), date_col, value_col, freq, seasonal_period,
        n_observations, n_missing, summary (dict)
    """
    date_col = date_col_override or detect_date_column(df)
    value_col = value_col_override or detect_value_column(df, date_col)

    if value_col is None:
        raise ValueError("수치형 값 컬럼을 찾을 수 없습니다. CSV 파일을 확인해주세요.")

    ts = to_timeseries(df, date_col, value_col)
    n_missing_before = int(ts.isna().sum())

    freq, seasonal_period = infer_frequency(ts)

    if regularize and freq is not None:
        ts = regularize_index(ts, freq)

    ts = fill_missing(ts, fill_method)

    summary = {
        "start": str(ts.index.min()) if isinstance(ts.index, pd.DatetimeIndex) else "0",
        "end": str(ts.index.max()) if isinstance(ts.index, pd.DatetimeIndex) else str(len(ts) - 1),
        "n_observations": int(len(ts)),
        "n_missing_before": n_missing_before,
        "n_missing_after": int(ts.isna().sum()),
        "min": float(np.nanmin(ts.values)) if len(ts) else np.nan,
        "max": float(np.nanmax(ts.values)) if len(ts) else np.nan,
        "mean": float(np.nanmean(ts.values)) if len(ts) else np.nan,
        "std": float(np.nanstd(ts.values)) if len(ts) else np.nan,
    }

    return {
        "ts": ts,
        "date_col": date_col,
        "value_col": value_col,
        "freq": freq,
        "seasonal_period": seasonal_period,
        "summary": summary,
    }


def train_test_split_ts(ts: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series]:
    """시계열의 끝 horizon 만큼을 테스트로, 나머지를 학습으로 분리."""
    if horizon >= len(ts):
        raise ValueError(f"시평({horizon})이 데이터 길이({len(ts)})보다 크거나 같습니다.")
    if horizon <= 0:
        raise ValueError("시평은 1 이상이어야 합니다.")
    train = ts.iloc[:-horizon]
    test = ts.iloc[-horizon:]
    return train, test


def make_future_index(last_index, periods: int, freq: Optional[str]):
    """예측을 위한 미래 인덱스 생성."""
    if isinstance(last_index, pd.Timestamp) and freq is not None:
        return pd.date_range(start=last_index, periods=periods + 1, freq=freq)[1:]
    elif isinstance(last_index, pd.Timestamp):
        # freq 추정 실패 시 일 단위로
        return pd.date_range(start=last_index, periods=periods + 1, freq="D")[1:]
    else:
        # 정수 인덱스
        start = int(last_index) + 1 if last_index is not None else 0
        return pd.RangeIndex(start=start, stop=start + periods)
