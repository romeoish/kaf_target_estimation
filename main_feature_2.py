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

# ✅ 변수 정의
all_variables = [
    'S/P 속도', '방사 속도', 'DS-1 연신비',
    'CR Box 압력', 'CR Roll 압력', 'HAC_온도_상', 'HAC_온도_하',
    'Steam 압력', 'CAN 수', 'Cutter 속도', 'DS-2 속도',
    'Spray농도', '분사량'
]

fixed_variable_names = ['S/P 속도', '방사 속도', 'CAN 수']
opt_variables = [v for v in all_variables if v not in fixed_variable_names]

# 🎯 각 target별 사용 feature 정의
targets = {
    'Denier': ['S/P 속도', '방사 속도', 'DS-1 연신비'],
    'Elongation': ['S/P 속도', '방사 속도', 'DS-1 연신비', '원료'],
    'Tenacity': ['S/P 속도', '방사 속도', 'DS-1 연신비', '원료'],
    'Cohesion': ['CR Box 압력', 'CR Roll 압력', 'HAC_온도_상', 'HAC_온도_하', 'Steam 압력',
                 'CAN 수', 'Cutter 속도', 'DS-1 연신비', '원료'],
    'TotalFinish': ['S/P 속도', '방사 속도', 'DS-1 연신비', 'CAN 수', 'DS-2 속도', 'Spray농도', '분사량']
}

# ✅ 가중치 설정
weights = {
    'Denier': 100,
    'Elongation': 1,
    'Tenacity': 30,
    'Cohesion': 1,
    'TotalFinish': 300
}

# ✅ 초기값과 bounds (opt_variables 순서에 맞춰 구성)
x0_map = {
    'DS-1 연신비': 1.4670,
    'CR Box 압력': 1.7207,
    'CR Roll 압력': 2.6636,
    'HAC_온도_상': 83.1885,
    'HAC_온도_하': 82.7519,
    'Steam 압력': 50.6121,
    'Cutter 속도': 121.7472,
    'DS-2 속도': 161.8047,
    'Spray농도': 7.3908,
    '분사량': 45.7331
}

bounds_map = {
    'DS-1 연신비': (0.25, 2.2),
    'CR Box 압력': (0.55, 2.85),
    'CR Roll 압력': (0.4, 3.2),
    'HAC_온도_상': (1.0, 128.0),
    'HAC_온도_하': (10.0, 126.0),
    'Steam 압력': (0.0, 75.0),
    'Cutter 속도': (106.0, 128.4),
    'DS-2 속도': (50.0, 181.2),
    'Spray농도': (0.23, 10.52),
    '분사량': (34.0, 70.0)
}

x0 = [x0_map[var] for var in opt_variables]
bounds = [bounds_map[var] for var in opt_variables]

# ✅ 입력 스키마 정의
class TargetInput(BaseModel):
    Denier: float
    Elongation: float
    Tenacity: float
    Cohesion: float
    TotalFinish: float
    원료: str
    S_P_속도: float
    방사_속도: float
    CAN_수: float

# ✅ 역방향 최적화 함수
def inverse_modeling(target_values: dict, raw_material: str, fixed_inputs: dict):
    def objective(x):
        input_dict = dict(zip(opt_variables, x))
        input_dict.update(fixed_inputs)
        input_dict['원료'] = raw_material  # 원료: 문자열 상태로 유지

        total_loss = 0

        for prop, target_val in target_values.items():
            try:
                model_path = os.path.join(model_dir, f"{prop}_xgb_model.pkl")
                model = joblib.load(model_path)
                features = targets[prop]

                input_df = pd.DataFrame([input_dict])

                if '원료' in features:
                    le_path = os.path.join(model_dir, f"label_encoder_{prop}.joblib")
                    le = joblib.load(le_path)
                    input_df['원료'] = le.transform([raw_material])

                pred = model.predict(input_df[features])[0]
                loss = weights[prop] * ((pred - target_val) ** 2)
                total_loss += loss
            except Exception as e:
                print(f"⚠ 예측 실패 ({prop}): {e}")
                total_loss += 1000

        return total_loss

    result = minimize(objective, x0, method='Powell', bounds=bounds)
    optimal = dict(zip(opt_variables, result.x))
    optimal.update(fixed_inputs)
    optimal['원료'] = raw_material
    return {"prediction": optimal}

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
        fixed_inputs = {
            'S/P 속도': input_data.S_P_속도,
            '방사 속도': input_data.방사_속도,
            'CAN 수': input_data.CAN_수,
        }
        result = inverse_modeling(target_dict, input_data.원료, fixed_inputs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"역방향 예측 실패: {str(e)}")

# ✅ 실행 명령어
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
