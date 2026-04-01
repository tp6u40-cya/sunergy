# routers/train_utils.py
# 共用常數、optional imports、工具函式
from fastapi import HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from models import AfterData

TW_TIMEZONE = timezone(timedelta(hours=8))

# ───────────────────────────────────────
# Optional imports for ML libraries
# ───────────────────────────────────────
HAS_SKLEARN = False
HAS_XGBOOST = False
HAS_OPTUNA = False

try:
    from sklearn.model_selection import train_test_split, ParameterGrid
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.svm import SVR as SKSVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


# ───────────────────────────────────────
# 目錄工具
# ───────────────────────────────────────
def _processed_data_dir() -> Path:
    base = Path(__file__).resolve().parent.parent / "processed_data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _models_dir(data_id: int) -> Path:
    p = Path(__file__).resolve().parent.parent / "uploads" / "models" / str(data_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _list_artifacts(data_id: int):
    d = _models_dir(data_id)
    items = []
    for p in sorted(d.glob("*")):
        if p.suffix in {".joblib", ".json", ".pt"}:
            meta = p.with_suffix(".meta.json")
            items.append({
                "artifact": p.name,
                "meta": meta.name if meta.exists() else None,
                "size": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
    return items


# ───────────────────────────────────────
# 資料讀取 / 驗證
# ───────────────────────────────────────
def _load_cleaned_csv(entry: AfterData) -> pd.DataFrame:
    """Load cleaned CSV from processed_data/ via AfterData.file_path"""
    if not entry.file_path:
        raise HTTPException(status_code=400, detail="此筆清洗紀錄沒有關聯的 CSV 檔案路徑")
    csv_path = Path(entry.file_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"清洗後的 CSV 不存在: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"讀取 CSV 失敗: {e}")


def _to_native(obj):
    try:
        import numpy as _np
    except Exception:
        _np = None
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if _np is not None and isinstance(obj, (_np.integer,)):
        return int(obj)
    if _np is not None and isinstance(obj, (_np.floating,)):
        return float(obj)
    return obj


def _ensure_time_features(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    """derive basic time features if a time column is available"""
    if time_col and time_col in df.columns:
        s = pd.to_datetime(df[time_col], errors="coerce")
        df = df.copy()
        df['hour'] = s.dt.hour
        df['dayofweek'] = s.dt.dayofweek
        df['month'] = s.dt.month
        df['hour_sin'] = np.sin(2 * np.pi * (df['hour'].fillna(0) / 24))
        df['hour_cos'] = np.cos(2 * np.pi * (df['hour'].fillna(0) / 24))
    return df


def _validate_clean_data(df: pd.DataFrame, gi_col: str, tm_col: str, target_col: str):
    missing_cols = [c for c in [gi_col, tm_col, target_col] if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"missing required column(s): {','.join(missing_cols)}")

    for c in [gi_col, tm_col, target_col]:
        try:
            df[c].astype(float)
        except Exception:
            raise HTTPException(status_code=400, detail=f"column '{c}' cannot be parsed as float")
        if pd.isna(df[c]).any():
            raise HTTPException(status_code=400, detail=f"column '{c}' has NaN but data should be cleaned")
