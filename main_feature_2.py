import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

# 📁 모델 디렉토리
model_dir = 'model'

# 🔧 최적화 대상 연속 변수
variables = [
    'S/P 속도', '방사 속도', 'DS-1 연신비',
    'CR Box 압력', 'CR Roll 압력', 'HAC_온도_상', 'HAC_온도_하',
    'Steam 압력', 'CAN 수', 'Cutter 속도', 'DS-2 속도',
    'Spray농도', '분사량'
]

# 🎯 각 target별 사용 feature 정의
targets = {
    'Denier': ['S/P 속도', '방사 속도', 'DS-1 연신비'],
    'Elongation': ['S/P 속도', '방사 속도', 'DS-1 연신비', '원료'],
    'Tenacity': ['S/P 속도', '방사 속도', 'DS-1 연신비', '원료'],
    'Cohesion': ['CR Box 압력', 'CR Roll 압력', 'HAC_온도_상', 'HAC_온도_하', 'Steam 압력',
                 'CAN 수', 'Cutter 속도', 'DS-1 연신비', '원료'],
    'TotalFinish': ['S/P 속도', '방사 속도', 'DS-1 연신비', 'CAN 수', 'DS-2 속도', 'Spray농도', '분사량']
}

# ✅ 가중치 설정 (값 스케일 보정용)
weights = {
    'Denier': 100,
    'Elongation': 1,
    'Tenacity': 30,
    'Cohesion': 1,
    'TotalFinish': 300
}

# ✅ 입력 스키마 정의
class TargetInput(BaseModel):
    Denier: float
    Elongation: float
    Tenacity: float
    Cohesion: float
    TotalFinish: float
    원료: str
    S_P_속도: float | None = None
    방사_속도: float | None = None
    CAN_수: float | None = None


# ✅ 역방향 최적화 함수
def inverse_modeling(target_values: dict, raw_material: str, overrides: dict = None):
    overrides = overrides or {}

    x0 = [24.8065, 1413.2207, 1.4670,
          1.7207, 2.6636, 83.1885, 82.7519,
          50.6121, 18.3847, 121.7472,
          161.8047, 7.3908, 45.7331]

    bounds = [(19.6, 27.0), (310.0, 1560.0), (0.25, 2.2),
              (0.55, 2.85), (0.4, 3.2), (1.0, 128.0), (10.0, 126.0),
              (0.0, 75.0), (5.0, 30.0), (106.0, 128.4),
              (50.0, 181.2), (0.23, 10.52), (34.0, 70.0)]

    # 🔒 고정할 변수 인덱스
    fixed_vars = {}
    for key, value in overrides.items():
        if key in variables:
            idx = variables.index(key)
            fixed_vars[idx] = value
            x0[idx] = value
            bounds[idx] = (value, value)  # 고정

    def objective(x):
        input_dict = dict(zip(variables, x))
        input_dict.update(overrides)  # 강제 덮어쓰기
        input_dict['원료'] = raw_material
        input_df_base = pd.DataFrame([input_dict])

        loss = 0
        for prop, target_val in target_values.items():
            try:
                model_path = os.path.join(model_dir, f"{prop}_xgb_model.pkl")
                model = joblib.load(model_path)
                features = targets[prop]
                input_df = input_df_base.copy()

                if '원료' in features:
                    le_path = os.path.join(model_dir, f"label_encoder_{prop}.joblib")
                    le = joblib.load(le_path)
                    input_df['원료'] = le.transform([raw_material])

                pred = model.predict(input_df[features])[0]
                weighted_loss = weights[prop] * ((pred - target_val) ** 2)
                loss += weighted_loss

            except Exception as e:
                print(f"⚠ 예측 실패: {prop}, error: {e}")
                loss += 1000
        return loss

    result = minimize(objective, x0, method='Powell', bounds=bounds)
    optimal_conditions = dict(zip(variables, result.x))
    optimal_conditions.update(overrides)  # 사용자 입력값 반영
    optimal_conditions['원료'] = raw_material
    return {"prediction": optimal_conditions}



# ✅ FastAPI 구성
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
@app.post("/KS-2")
async def inverse_predict(input_data: TargetInput):
    try:
        target_dict = {
            'Denier': input_data.Denier,
            'Elongation': input_data.Elongation,
            'Tenacity': input_data.Tenacity,
            'Cohesion': input_data.Cohesion,
            'TotalFinish': input_data.TotalFinish,
        }

        # 사용자가 입력한 값만 overrides로 넘김
        overrides = {}
        if input_data.S_P_속도 is not None:
            overrides['S/P 속도'] = input_data.S_P_속도
        if input_data.방사_속도 is not None:
            overrides['방사 속도'] = input_data.방사_속도
        if input_data.CAN_수 is not None:
            overrides['CAN 수'] = input_data.CAN_수

        result = inverse_modeling(target_dict, input_data.원료, overrides)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"역방향 예측 실패: {str(e)}")
    
# ✅ 실행 명령어
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
