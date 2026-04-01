# routers/train.py
# 訓練 + 模型管理端點
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import delete
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import func

from database import get_db
from models import AfterData, TrainedModel, Site, SiteData
from schemas import TrainRequest
from routers.train_utils import (
    TW_TIMEZONE,
    HAS_SKLEARN, HAS_XGBOOST, HAS_OPTUNA,
    _processed_data_dir, _models_dir, _list_artifacts,
    _load_cleaned_csv, _to_native,
    _ensure_time_features, _validate_clean_data,
)

# re-import conditional libraries (guarded by HAS_* flags from train_utils)
if HAS_SKLEARN:
    from sklearn.model_selection import train_test_split, ParameterGrid
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.svm import SVR as SKSVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
if HAS_XGBOOST:
    import xgboost as xgb
if HAS_OPTUNA:
    import optuna


router = APIRouter(prefix="/train", tags=["Train"])


# ═══════════════════════════════════════
#  資訊 / 除錯端點
# ═══════════════════════════════════════
@router.get("/debug")
def debug_modules():
    """Debug endpoint to check module availability"""
    return {
        "HAS_SKLEARN": HAS_SKLEARN,
        "HAS_XGBOOST": HAS_XGBOOST,
        "HAS_OPTUNA": HAS_OPTUNA,
    }


@router.get("/info")
def train_info(data_id: int, db: Session = Depends(get_db)):
    # 先找清洗資料
    entry = db.query(AfterData).filter(AfterData.after_id == data_id).first()

    if entry:
        return {
            "type": "cleaned",
            "file_name": entry.after_name
        }

    # 沒有清洗 → 找原始資料
    row = db.query(SiteData).filter(SiteData.upload_id == data_id).first()

    if row:
        return {
            "type": "raw",
            "file_name": row.data_name
        }

    raise HTTPException(status_code=404, detail="找不到資料")
    return {
        "data_id": data_id,
        "cleaned_file": entry.file_path,
        "after_name": entry.after_name,
        "before_rows": entry.before_rows,
        "after_rows": entry.after_rows,
    }


@router.get("/models")
def list_models(data_id: int):
    return {"data_id": data_id, "artifacts": _list_artifacts(data_id)}


# ═══════════════════════════════════════
#  模型管理端點
# ═══════════════════════════════════════
@router.delete("/trained-models/{model_id}")
def delete_trained_model(
    model_id: int,
    user_id: int = Query(...),
    db: Session = Depends(get_db)
):
    row = (
        db.query(TrainedModel, Site)
        .join(Site, TrainedModel.site_id == Site.site_id)
        .filter(
            TrainedModel.model_id == model_id,
            Site.user_id == user_id
        )
        .first()
    )

    if not row:
        raise HTTPException(status_code=404, detail="找不到此模型，或你沒有權限刪除")

    model, site = row

    # 刪除模型檔案
    if model.file_path:
        base_dir = Path(__file__).resolve().parent.parent
        artifact_path = base_dir / model.file_path
        if artifact_path.exists():
            artifact_path.unlink()

        meta_path = artifact_path.with_suffix(".meta.json")
        if meta_path.exists():
            meta_path.unlink()

    db.delete(model)
    db.commit()

    return {
        "message": f"模型 {model_id} 已刪除"
    }


@router.get("/trained-models")
def list_trained_models(
    user_id: int = Query(...),
    db: Session = Depends(get_db)
):
    rows = (
        db.query(TrainedModel, Site, AfterData)
        .join(Site, TrainedModel.site_id == Site.site_id)
        .outerjoin(AfterData, TrainedModel.data_id == AfterData.after_id)
        .filter(Site.user_id == user_id)
        .order_by(TrainedModel.trained_at.desc())
        .all()
    )

    out = []
    for model, site, after in rows:

        if after:
            file_name = after.after_name
        else:
            # 🔥 再查一次 SiteData（只查一筆，不會重複）
            site_data = db.query(SiteData).filter(
                SiteData.upload_id == model.data_id
            ).first()

            file_name = site_data.data_name if site_data else "未知檔案"
    
        out.append({
            "model_id": model.model_id,
            "model_type": model.model_type,
            "parameters": model.parameters,
            "file_path": model.file_path,
            "trained_at": model.trained_at.isoformat() if model.trained_at else None,

            "site_name": site.site_name if site else None,
            "location": site.location if site else None,

            "file_name": file_name
        })

    return out

@router.post("/trained-models/batch-delete")
def batch_delete_models(data: dict, db: Session = Depends(get_db)):
    model_ids = data.get("model_ids", [])
    user_id = data.get("user_id")

    if not model_ids:
        raise HTTPException(status_code=400, detail="沒有選擇模型")

    # 🔥 只刪自己的模型（超重要）
    models = (
        db.query(TrainedModel)
        .join(Site, TrainedModel.site_id == Site.site_id)
        .filter(
            TrainedModel.model_id.in_(model_ids),
            Site.user_id == user_id
        )
        .all()
    )

    if not models:
        raise HTTPException(status_code=404, detail="找不到可刪除的模型")

    # 刪檔案（跟你單筆一樣）
    for model in models:
        if model.file_path:
            base_dir = Path(__file__).resolve().parent.parent
            artifact_path = base_dir / model.file_path
            if artifact_path.exists():
                artifact_path.unlink()

            meta_path = artifact_path.with_suffix(".meta.json")
            if meta_path.exists():
                meta_path.unlink()

        db.delete(model)

    db.commit()

    return {"message": f"成功刪除 {len(models)} 筆"}

# ═══════════════════════════════════════
#  單模型訓練（內部函式）
# ═══════════════════════════════════════
def _train_single_model(model_id: str, X_train, y_train, X_test, y_test, strategy: str, param_spec: Dict[str, Any], device_pref: str = "auto"):
    # Dependency checks per model
    if model_id in ("SVR", "RandomForest") and not HAS_SKLEARN:
        raise HTTPException(status_code=500, detail="scikit-learn not available on server")
    if model_id == "XGBoost" and not HAS_XGBOOST:
        raise HTTPException(status_code=500, detail="xgboost not available on server")
    if model_id == "LSTM":
        raise HTTPException(status_code=400, detail="LSTM model has been removed from this system")

    best = None
    tried = []

    def evaluate(m):
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        eps = 1e-6
        wmape = float(np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test) + eps)))
        return {"r2": r2, "rmse": rmse, "mae": mae, "wmape": wmape}

    # build candidate param grid
    def build_grid(spec: Dict[str, Any]):
        def _make_range(start, end, step):
            def _is_intlike(x):
                try:
                    return float(x).is_integer()
                except Exception:
                    return isinstance(x, int)
            is_int_grid = _is_intlike(start) and _is_intlike(end) and _is_intlike(step)
            arr = np.arange(float(start), float(end) + (float(step) or 1.0), float(step))
            if is_int_grid:
                return [int(round(x)) for x in arr]
            return [float(np.round(x, 6)) for x in arr]

        grid = {}
        for k, v in (spec or {}).items():
            if isinstance(k, str) and k.startswith('_'):
                continue
            if isinstance(v, dict) and {"start", "end", "step"}.issubset(v.keys()):
                start, end, step = v.get("start"), v.get("end"), v.get("step", 1)
                grid[k] = _make_range(start, end, step)
            elif isinstance(v, dict) and {"values"}.issubset(v.keys()):
                grid[k] = list(v["values"])
            else:
                grid[k] = [v]
        return list(ParameterGrid(grid)) if grid else [{}]

    # Bayesian Optimization (Optuna-based) for non-LSTM models
    if strategy == "bayes":
        if not HAS_OPTUNA:
            raise HTTPException(status_code=500, detail="bayesian optimization requires optuna on server")

        def suggest_from_spec(trial, spec: Dict[str, Any]):
            params = {}
            for k, v in (spec or {}).items():
                if isinstance(v, dict) and {"values"}.issubset(v.keys()):
                    params[k] = trial.suggest_categorical(k, list(v["values"]))
                elif isinstance(v, dict) and {"start", "end"}.issubset(v.keys()):
                    start = float(v["start"])
                    end = float(v["end"])
                    step = float(v.get("step", 0))
                    is_int = float(start).is_integer() and float(end).is_integer() and float(step or 0).is_integer()
                    if is_int:
                        if step and step > 0:
                            params[k] = trial.suggest_int(k, int(start), int(end), step=int(step))
                        else:
                            params[k] = trial.suggest_int(k, int(start), int(end))
                    else:
                        if step and step > 0:
                            params[k] = trial.suggest_float(k, start, end, step=step)
                        else:
                            params[k] = trial.suggest_float(k, start, end)
                else:
                    params[k] = v
            return params

        def build_model(mid: str, p: Dict[str, Any]):
            if mid == "SVR":
                C = float(p.get("C", 100.0))
                kernel = p.get("kernel", "rbf")
                gamma = p.get("gamma", "scale")
                epsilon = float(p.get("epsilon", 0.1))
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("svr", SKSVR(C=C, kernel=kernel, gamma=gamma, epsilon=epsilon, cache_size=500)),
                ])
            if mid == "RandomForest":
                n_estimators = int(p.get("n_estimators", 200))
                max_depth = p.get("max_depth", None)
                return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            if mid == "XGBoost":
                xgb_kwargs = {
                    'n_estimators': int(p.get('n_estimators', 300)),
                    'learning_rate': float(p.get('learning_rate', 0.1)),
                    'subsample': float(p.get('subsample', 0.8)),
                    'colsample_bytree': float(p.get('colsample_bytree', 0.8)),
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist',
                }
                md = p.get('max_depth', 6)
                if md is not None and md != '' and md != 'None':
                    xgb_kwargs['max_depth'] = int(md)
                if 'min_child_weight' in p:
                    xgb_kwargs['min_child_weight'] = float(p.get('min_child_weight'))
                if 'reg_lambda' in p:
                    xgb_kwargs['reg_lambda'] = float(p.get('reg_lambda'))
                if 'reg_alpha' in p:
                    xgb_kwargs['reg_alpha'] = float(p.get('reg_alpha'))
                return xgb.XGBRegressor(**xgb_kwargs)
            raise ValueError(f"unsupported model '{mid}' for bayes")

        def objective(trial):
            p = suggest_from_spec(trial, param_spec or {})
            model = build_model(model_id, p)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            eps = 1e-6
            wmape = float(np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test)) + eps))
            return wmape

        trials = int((param_spec or {}).get('_trials', 30))
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best_p = study.best_params

        # evaluate metrics for best params
        model = build_model(model_id, best_p)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        eps = 1e-6
        wmape = float(np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test)) + eps))
        return {"best": {"params": best_p, "r2": r2, "rmse": rmse, "mae": mae, "wmape": wmape}, "trials": []}

    candidates = [{}]
    if strategy == "grid":
        candidates = build_grid(param_spec or {})
        try:
            max_n = int((param_spec or {}).get('_max_combinations', 0) or 0)
        except Exception:
            max_n = 0
        if max_n > 0 and len(candidates) > max_n:
            idx = np.linspace(0, len(candidates) - 1, num=max_n)
            idx = [int(round(i)) for i in idx]
            seen = set()
            selected = []
            for i in idx:
                if i not in seen:
                    selected.append(i)
                    seen.add(i)
            candidates = [candidates[i] for i in selected]

    for params in candidates:
        if model_id == "SVR":
            C = float(params.get("C", 100.0))
            kernel = params.get("kernel", "rbf")
            gamma = params.get("gamma", "scale")
            epsilon = float(params.get("epsilon", 0.1))
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SKSVR(C=C, kernel=kernel, gamma=gamma, epsilon=epsilon, cache_size=500)),
            ])
        elif model_id == "RandomForest":
            n_estimators = int(params.get("n_estimators", 200))
            max_depth = params.get("max_depth", None)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif model_id == "XGBoost":
            xgb_kwargs = {
                'n_estimators': int(params.get('n_estimators', 300)),
                'learning_rate': float(params.get('learning_rate', 0.1)),
                'subsample': float(params.get('subsample', 0.8)),
                'colsample_bytree': float(params.get('colsample_bytree', 0.8)),
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
            }
            md = params.get('max_depth', 6)
            if md is not None and md != '' and md != 'None':
                xgb_kwargs['max_depth'] = int(md)
            if 'min_child_weight' in params:
                xgb_kwargs['min_child_weight'] = float(params.get('min_child_weight'))
            if 'reg_lambda' in params:
                xgb_kwargs['reg_lambda'] = float(params.get('reg_lambda'))
            if 'reg_alpha' in params:
                xgb_kwargs['reg_alpha'] = float(params.get('reg_alpha'))
            model = xgb.XGBRegressor(**xgb_kwargs)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_id}")

        # For SVR, RandomForest, XGBoost: use generic evaluate function
        metrics = evaluate(model)
        tried.append({"params": params, **metrics})
        if (best is None) or (metrics["wmape"] < best["wmape"]):
            best = {"params": params, **metrics}

    return {"best": best, "trials": tried}


# ═══════════════════════════════════════
#  POST /train/run  — 執行模型訓練
# ═══════════════════════════════════════
@router.post("/run")
def run_training(payload: TrainRequest, db: Session = Depends(get_db)):
    """payload.data_id = after_data.after_id"""
    entry = db.query(AfterData).filter(AfterData.after_id == payload.data_id).first()

    data_source = "cleaned"
    cleaned_path = None
    site_id = None

    # 有清洗資料
    if entry:
        df = _load_cleaned_csv(entry)
        cleaned_path = entry.file_path
        site_id = entry.site_id

    # 沒有清洗資料 → 從 site_data
    else:
        rows = (
            db.query(SiteData)
            .filter(SiteData.upload_id == payload.data_id)
            .all()
        )

        if not rows:
            raise HTTPException(status_code=404, detail="找不到原始資料")

        data_source = "raw"
        site_id = rows[0].site_id

        df = pd.DataFrame([{
            "GI": float(r.gi),
            "TM": float(r.tm),
            "EAC": float(r.eac),
            "the_date": r.the_date,
            "the_hour": r.the_hour
        } for r in rows])
    target_col = payload.target or 'EAC'
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"target column '{payload.target}' not found")

    # enforce default features GI + TM
    features: List[str] = payload.features or ['GI','TM']
    missing = [c for c in ['GI','TM'] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"required feature(s) missing: {','.join(missing)}")

    _validate_clean_data(df, gi_col='GI', tm_col='TM', target_col=target_col)

    # time features & strict time sorting on df level
    time_col = payload.time_col
    if time_col is None:
        for cand in ['timestamp','datetime','time','recordtime','thedate','Date','Time']:
            if cand in df.columns:
                time_col = cand
                break
    if time_col and time_col in df.columns:
        parsed = pd.to_datetime(df[time_col], errors='coerce')
        df = df.loc[parsed.notna()].copy()
        df[time_col] = parsed[parsed.notna()]
        df = df.sort_values(time_col).reset_index(drop=True)
        df = _ensure_time_features(df, time_col)

    # build design matrix
    feature_cols = features
    num_df = df[feature_cols].select_dtypes(include=[np.number])
    if num_df.isna().any().any():
        raise HTTPException(status_code=400, detail="numeric features contain NaN; cleaned data should not have missing values in features")
    if num_df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no numeric features after preprocessing; ensure GI/TM exist and are numeric")
    X = num_df.values
    y = df[target_col].astype(float).values

    if len(y) < 10:
        raise HTTPException(status_code=400, detail="not enough rows to train")

    # time-based split or random split
    if payload.split_method == 'time' and len(y) > 10:
        test_size = max(0.05, min(0.5, 1.0 - float(payload.split_ratio)))
        n_test = max(1, int(len(y) * test_size))
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]
    else:
        test_size = max(0.05, min(0.5, 1.0 - float(payload.split_ratio)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    results: Dict[str, Any] = {}

    for m in payload.models:
        spec = (payload.params or {}).get(m, {})
        try:
            res = _train_single_model(m, X_train, y_train, X_test, y_test, payload.strategy, spec, payload.device)
            best = res.get("best") or {}
            results[m] = {
                "id": m,
                "status": "ok" if best else "failed",
                "r2": round(best.get("r2", 0.0), 3),
                "rmse": round(best.get("rmse", 0.0), 3),
                "mae": round(best.get("mae", 0.0), 3),
                "wmape": round(best.get("wmape", 0.0), 4),
                "best_params": best.get("params", {}),
            }
        except HTTPException as e:
            results[m] = {
                "id": m,
                "status": "error",
                "error": e.detail if hasattr(e, 'detail') else str(e),
            }
        except Exception as e:
            results[m] = {"id": m, "status": "error", "error": str(e)}

    # Optional artifact saving
    if payload.save_model:
        import json as _json
        from datetime import datetime as _dt
        import joblib as _joblib
        saved = []
        save_errors = []
        timestamp = _dt.utcnow().strftime('%Y%m%d_%H%M%S%f')
        models_dir = _models_dir(payload.data_id)
        for mid, res in results.items():
            if res.get('status') != 'ok':
                continue
            try:
                if mid == 'XGBoost' and HAS_XGBOOST:
                    best_params = res.get('best_params', {}) or {}
                    xgb_kwargs = {
                        'n_estimators': int(best_params.get('n_estimators', 300)),
                        'learning_rate': float(best_params.get('learning_rate', 0.1)),
                        'subsample': float(best_params.get('subsample', 0.8)),
                        'colsample_bytree': float(best_params.get('colsample_bytree', 0.8)),
                        'random_state': 42,
                        'n_jobs': -1,
                        'tree_method': 'hist',
                    }
                    md = best_params.get('max_depth', 6)
                    if md is not None and md != '' and md != 'None':
                        xgb_kwargs['max_depth'] = int(md)
                    if 'min_child_weight' in best_params:
                        xgb_kwargs['min_child_weight'] = float(best_params.get('min_child_weight'))
                    if 'reg_lambda' in best_params:
                        xgb_kwargs['reg_lambda'] = float(best_params.get('reg_lambda'))
                    if 'reg_alpha' in best_params:
                        xgb_kwargs['reg_alpha'] = float(best_params.get('reg_alpha'))
                    model = xgb.XGBRegressor(**xgb_kwargs)
                    model.fit(X_train, y_train)
                    try:
                        path = models_dir / f"{timestamp}_{mid}.json"
                        model.save_model(str(path))
                    except Exception:
                        path = models_dir / f"{timestamp}_{mid}.joblib"
                        _joblib.dump(model, path)
                else:
                    if mid == 'RandomForest':
                        n_estimators = int(res.get('best_params', {}).get('n_estimators', 200))
                        max_depth = res.get('best_params', {}).get('max_depth', None)
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth if max_depth is not None else None, random_state=42, n_jobs=-1)
                        model.fit(X_train, y_train)
                        path = models_dir / f"{timestamp}_{mid}.joblib"
                        _joblib.dump(model, path)
                    elif mid == 'SVR':
                        C = float(res.get('best_params', {}).get('C', 100.0))
                        epsilon = float(res.get('best_params', {}).get('epsilon', 0.1))
                        model = Pipeline([('scaler', StandardScaler()), ('svr', SKSVR(C=C, kernel='rbf', gamma='scale', epsilon=epsilon, cache_size=500))])
                        model.fit(X_train, y_train)
                        path = models_dir / f"{timestamp}_{mid}.joblib"
                        _joblib.dump(model, path)
                    else:
                        continue
                # write sidecar metadata
                meta_path = models_dir / f"{timestamp}_{mid}.meta.json"
                sidecar = {
                    'model_id': mid,
                    'artifact': str(path.name),
                    'feature_cols_used': feature_cols,
                    'target': target_col,
                    'time_col': time_col,
                    'split_method': payload.split_method,
                    'trained_at': timestamp,
                }
                with open(meta_path, 'w', encoding='utf-8') as fh:
                    _json.dump(sidecar, fh, ensure_ascii=False, indent=2)
                saved.append({'model_id': mid, 'artifact': str(path.name), 'meta': meta_path.name})
            except Exception as e:
                try:
                    save_errors.append(f"save_failed:{mid}:{str(e)}")
                except Exception:
                    save_errors.append(f"save_failed:{mid}")
                continue
        # insert trained models into DB
        if site_id is None:
            raise HTTPException(status_code=400, detail="site_id 為空，無法建立 trained_model")

        for art in saved:
            try:
                tm = TrainedModel(
                    site_id=site_id,
                    data_id=payload.data_id,
                    model_type=art.get('model_id'),
                    parameters=results.get(art.get('model_id'), {}).get('best_params', {}),
                    file_path=str(Path("uploads") / "models" / str(payload.data_id) / art.get('artifact')),
                    trained_at=datetime.now(TW_TIMEZONE).replace(tzinfo=None),
                )
                db.add(tm)
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail=f"建立 TrainedModel 失敗: {str(e)}")

        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"寫入 trained_model 失敗: {str(e)}")

        db.commit()

    warnings = []
    if 'save_errors' in locals() and save_errors:
        warnings.extend(save_errors)

    return _to_native({
        "data_id": payload.data_id,
        "data_source": data_source,
        "cleaned_file": cleaned_path,
        "split": {
            "train": float(payload.split_ratio),
            "test": 1.0 - float(payload.split_ratio),
        },
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_cols_used": feature_cols,
        "results": results,
        "warnings": warnings,
    })

@router.get("/dashboard-stats/{user_id}")
def get_dashboard_stats(user_id: int, db: Session = Depends(get_db)):
    """
    計算該使用者的減碳效益統計資料
    """
    # 1. 取得該使用者所有模型關聯的總發電量 (EAC)
    # 邏輯：透過 Site 關聯 TrainedModel，再透過 TrainedModel 關聯 SiteData
    # 這裡計算的是所有上傳資料的總和
    total_eac = (
        db.query(func.sum(SiteData.eac))
        .join(Site, SiteData.site_id == Site.site_id)
        .filter(Site.user_id == user_id)
        .scalar()
    ) or 0

    # 2. 定義電力排碳係數 (根據最新公告)
    carbon_factor = 0.494 
    
    # 3. 計算總減碳量
    total_carbon_reduction = round(total_eac * carbon_factor, 2)

    return {
        "user_id": user_id,
        "total_kwh": round(total_eac, 2),
        "carbon_factor": carbon_factor,
        "total_carbon_reduction": total_carbon_reduction
    }