"""
Yield Prediction Model (Hummingbird PyTorch Edition)
======================
입력 피처:
  - Step 1 ~ Step 6 : 각 공정 단계에 배정된 기계 번호 (범주형 → OneHotEncoding)
  - Wafer quality   : 웨이퍼 초기 품질 (수치형)

타겟:
  - Yield : 최종 웨이퍼 수율

※ 모델 학습은 Scikit-learn으로 진행 후, 최고 성능 모델을 Hummingbird를 통해 
   순수 PyTorch 모델로 변환(Compile)하여 저장합니다.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Hummingbird Import
from hummingbird.ml import convert

warnings.filterwarnings("ignore")

def analysis(q_id = 1, paths_id = 2, csv_path = None, out_dir = None):
    # ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
    if csv_path is None:
        csv_path = f"quality_data/Q_{q_id}/historical_paths_{paths_id}.csv"
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df):,}")

    STEP_COLS  = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"]
    WAFER_COLS = ["Wafer quality"]
    TARGET     = "Yield"

    # 향후 PyTorch에서 F.one_hot()의 num_classes를 지정하기 위해 기계 개수 파악 (CSV는 1-based)
    machine_cnt_list = [int(df[col].max()) for col in STEP_COLS]

    X = df[STEP_COLS + WAFER_COLS]
    y = df[TARGET]

    # ── 2. 50:50 Train / Test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # ── 3. 전처리 ─────────────────────────────────────────────────────────────────
    preprocessor = ColumnTransformer([
        ("steps_ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), STEP_COLS),
        ("wafer_num", "passthrough", WAFER_COLS),
    ])

    # ── 4. 모델 정의 ──────────────────────────────────────────────────────────────
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
        ),
    }

    # ── 5. 학습 & 평가 ────────────────────────────────────────────────────────────
    results = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        results[name] = {"pipeline": pipe, "mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}

        print(f"\n[{name}]")
        print(f"  MAE  : {mae:.6f}")
        print(f"  RMSE : {rmse:.6f}")
        print(f"  R²   : {r2:.6f}")

    # ── 6. 최적 모델 선택 (RMSE 기준) ─────────────────────────────────────────────
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best      = results[best_name]
    print(f"\n★ Best model : {best_name}")

    # ── 7. Feature Importance (트리 기반 모델) ────────────────────────────────────
    best_model = best["pipeline"].named_steps["model"]

    if hasattr(best_model, "feature_importances_"):
        prep          = best["pipeline"].named_steps["prep"]
        ohe_features  = prep.named_transformers_["steps_ohe"].get_feature_names_out(STEP_COLS).tolist()
        feature_names = ohe_features + WAFER_COLS
        importances   = best_model.feature_importances_

        step_importance = {}
        for col in STEP_COLS:
            idx = [i for i, n in enumerate(feature_names) if n.startswith(col)]
            step_importance[col] = float(importances[idx].sum())
        step_importance["Wafer quality"] = float(importances[len(ohe_features)])

        print("\nFeature Importance (Step별 합산):")
        for k, v in sorted(step_importance.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:.4f} ({v*100:.1f}%)")

    # ── 8. JSON 메타데이터 결과 저장 ───────────────────────────────────────────────
    output = {
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "results": {
            k: {"mae": v["mae"], "rmse": v["rmse"], "r2": v["r2"]}
            for k, v in results.items()
        },
        "best_model":      best_name,
        "step_importance": step_importance if hasattr(best_model, "feature_importances_") else {},
    }

    if out_dir is None:
        out_dir = f"quality_results/Q_{q_id}/paths_{paths_id}"
    os.makedirs(out_dir, exist_ok=True)

    results_path = f"{out_dir}/model_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── 9. Best 모델 변환(Hummingbird) 및 저장 ────────────────────────────────────
    
    # [1] Pandas 파이프라인(비교용/백업) 저장
    model_path = f"{out_dir}/best_pipeline.pkl"
    joblib.dump(best["pipeline"], model_path)

    # [2] 순수 머신러닝 예측기만 추출하여 PyTorch로 변환 및 저장
    print("\nConverting Best Scikit-learn model to PyTorch via Hummingbird...")
    hb_model = convert(best_model, 'pytorch')
    hb_model_path = f"{out_dir}/best_hb_model"
    hb_model.save(hb_model_path)

    # [3] 시뮬레이터를 위한 통계 및 파라미터 저장
    wafer_q = df[WAFER_COLS[0]].to_numpy(dtype=np.float64)
    wafer_stats = {
        "model_name":       best_name,
        "machine_cnt_list": machine_cnt_list,  # RL Wrapper에서 OneHot 생성을 위해 추가
        "step_cols":        STEP_COLS,
        "wafer_quality_col": WAFER_COLS[0],
        "wafer_quality": {
            "min":    float(wafer_q.min()),
            "max":    float(wafer_q.max()),
            "mean":   float(wafer_q.mean()),
            "std":    float(wafer_q.std()),
            "samples": wafer_q.tolist(),
        },
    }
    wafer_path = f"{out_dir}/wafer_quality.json"
    with open(wafer_path, "w") as f:
        json.dump(wafer_stats, f)

    print(f"Saved PyTorch(Hummingbird) model → {hb_model_path}")
    print(f"Saved wafer quality stat         → {wafer_path}")

if __name__ == "__main__":
    for i in range(1, 11):
        analysis(q_id=1, paths_id=i)