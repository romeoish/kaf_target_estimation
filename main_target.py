import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Type
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

##### ===================== KS-2 연신&방사 feature를 활용한 target 값 예측 =====================

targets = {
    'Denier': ['S/P 속도', '방사 속도', 'DS-1 연신비'],
    'Elongation': ['S/P 속도', '방사 속도', 'DS-1 연신비', '원료'],
    'Tenacity': ['S/P 속도', '방사 속도', 'DS-1 연신비', '원료'],
    'Cohesion': ['CR Box 압력', 'CR Roll 압력', 'HAC_온도_상', 'HAC_온도_하', 'Steam 압력',
                 'CAN 수', 'Cutter 속도', 'DS-1 연신비', '원료'],
    'Total Finish': ['S/P 속도', '방사 속도', 'DS-1 연신비', 'CAN 수', 'DS-2 속도', 'Spray농도', '분사량']
}


class FeatureInput_KS_2_Y(BaseModel):
    # ---------------- 공통 Feature ----------------
    SP_속도: float = Field(..., alias="S/P 속도")
    방사_속도: float = Field(..., alias="방사 속도")
    DS_1_연신비: float = Field(..., alias="DS-1 연신비")
    원료: str = Field(..., alias="원료")

    # ---------------- Cohesion 전용 ----------------
    CR_Box_압력: float = Field(..., alias="CR Box 압력")
    CR_Roll_압력: float = Field(..., alias="CR Roll 압력")
    HAC_온도_상: float = Field(..., alias="HAC_온도_상")
    HAC_온도_하: float = Field(..., alias="HAC_온도_하")
    Steam_압력: float = Field(..., alias="Steam 압력")
    Cutter_속도: float = Field(..., alias="Cutter 속도")

    # ---------------- Total Finish 전용 ----------------
    CAN_수: float = Field(..., alias="CAN 수")
    DS_2_속도: float = Field(..., alias="DS-2 속도")
    Spray_농도: float = Field(..., alias="Spray농도")
    분사량: float = Field(..., alias="분사량")

    class Config:
        allow_population_by_field_name = True

def predict_KS_2_Y(features: FeatureInput_KS_2_Y):
    input_data = pd.DataFrame([features.dict(by_alias=True)])

    model_dir = "model"
    results = {}

    for target, feature_cols in targets.items():
        model_path = os.path.join(model_dir, f"{target}_xgb_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"🔴 모델 파일이 없습니다: {model_path}")

        model = joblib.load(model_path)

        # 🎯 예측에 필요한 feature만 추출
        input_features = input_data[feature_cols].copy()

# ❗ 범주형 처리: 학습된 LabelEncoder 사용
        if "원료" in input_features.columns:
            le_path = os.path.join(model_dir, f"label_encoder_{target}.joblib")
            if not os.path.exists(le_path):
                raise FileNotFoundError(f"🔴 LabelEncoder 파일이 없습니다: {le_path}")

            le = joblib.load(le_path)
            input_features["원료"] = le.transform(input_features["원료"].astype(str))

        y_pred = model.predict(input_features)[0]
        results[target] = round(float(y_pred), 4)

    return {"prediction": results}


# ✅ CORS 설정 포함 FastAPI 앱 정의
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]
app = FastAPI(middleware=middleware)


# ✅ API 엔드포인트
@app.post("/predict_KS_2_Y")
async def predict_target(features: FeatureInput_KS_2_Y):
    try:
        result = predict_KS_2_Y(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 오류: {str(e)}")

# 로컬 서버 구동
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)