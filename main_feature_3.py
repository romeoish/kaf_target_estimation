import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# --- 0. 설정 및 상수 정의 ---

MODEL_DIR = 'model_reverse' 

# ⭐️ 역방향 모델의 입력 Feature 정의 (API의 입력값이 될 것)
inverse_features = [
    'Denier', 'Elongation', 'Tenacity', 'Cohesion', 'Total Finish',  # 모델 학습 시 사용된 실제 컬럼명
    'S/P 속도', '방사 속도', '원료','CAN 수'  
]

# ⭐️ 역방향 모델의 출력 Target 정의 (API의 출력값이 될 것)
inverse_targets = [
    'DS-1 연신비', 'CR Box 압력', 'CR Roll 압력', 'HAC_온도_상', 'HAC_온도_하',
    'Steam 압력', 'Cutter 속도', 'DS-2 속도', 'Spray농도', '분사량'
]

# --- 1. 저장된 역방향 모델 및 LabelEncoder 로드 ---
loaded_inverse_models: Dict[str, Any] = {}
loaded_inverse_label_encoder_원료: Any = None 

print(f"Loading inverse models and LabelEncoders from: {MODEL_DIR}")

for target_name in inverse_targets:
    # 모델 파일명은 공백을 언더스코어로 바꾼 형태로 저장되어야 합니다.
    model_filename = os.path.join(MODEL_DIR, f'inverse_model_{target_name.replace(" ", "_")}.pkl')
    if os.path.exists(model_filename):
        try:
            loaded_inverse_models[target_name] = joblib.load(model_filename)
            print(f"✅ Loaded inverse model: {model_filename}")
        except Exception as e:
            print(f"❌ Failed to load inverse model {model_filename}: {e}")
    else:
        print(f"❌ Inverse model file not found: {model_filename}")


le_filename = os.path.join(MODEL_DIR, 'label_encoder_inverse_원료.joblib')
if os.path.exists(le_filename):
    try:
        loaded_inverse_label_encoder_원료 = joblib.load(le_filename)
        print(f"✅ Loaded '원료' LabelEncoder: {le_filename}")
    except Exception as e:
        print(f"❌ Failed to load '원료' LabelEncoder {le_filename}: {e}")
else:
    print(f"❌ '원료' LabelEncoder file not found: {le_filename}")
    print(f"⚠️ WARNING: '원료' LabelEncoder is crucial for inverse prediction if '원료' is an input feature. Please ensure it exists.")


# --- 2. FastAPI 애플리케이션 정의 ---

app = FastAPI(
    title="역방향 예측 API (간결 버전)",
    description="물성값과 고정 조건으로 나머지 조건값을 직접 예측하는 간결한 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Pydantic 모델 정의 ---

class InverseInput(BaseModel):
    Denier: float = Field(..., description="원하는 Denier 값")
    Elongation: float = Field(..., description="원하는 Elongation 값")
    Tenacity: float = Field(..., description="원하는 Tenacity 값")
    Cohesion: float = Field(..., description="원하는 Cohesion 값")
    Total_Finish: float = Field(..., alias="TotalFinish", description="원하는 Total Finish 값") 

    SP_속도: float = Field(..., alias="S/P 속도", description="고정 S/P 속도")
    방사_속도: float = Field(..., alias="방사 속도", description="고정 방사 속도")
    원료: str = Field(..., description="고정 원료 종류 (예: '원료_0', '원료_1')")
    CAN_수: float = Field(..., alias="CAN 수", description="고정 CAN 수")

    class Config:
        populate_by_name = True # Pydantic V2
        json_schema_extra = { 
            "example": {
                "Denier": 75.0,
                "Elongation": 30.0,
                "Tenacity": 35.0,
                "Cohesion": 15.0,
                "TotalFinish": 5.0, 
                "S/P 속도": 25.0,
                "방사 속도": 1000.0,
                "원료": "원료_0",
                "CAN 수": 15.0
            }
        }

class InverseOutput(BaseModel):
    # Output 모델의 필드명은 파이썬 변수명 규칙을 따르고, alias는 JSON 키에 맞춥니다.
    # HAC_온도_상/하와 분사량은 alias가 없어도 파이썬 변수명과 JSON 키가 동일할 수 있으나,
    # 명확성을 위해 일관된 패턴을 따르는 것이 좋습니다.
    DS_1_연신비: float = Field(..., alias="DS-1 연신비", description="예측된 DS-1 연신비")
    CR_Box_압력: float = Field(..., alias="CR Box 압력", description="예측된 CR Box 압력")
    CR_Roll_압력: float = Field(..., alias="CR Roll 압력", description="예측된 CR Roll 압력")
    HAC_온도_상: float = Field(..., alias="HAC_온도_상", description="예측된 HAC_온도_상") # alias 명시
    HAC_온도_하: float = Field(..., alias="HAC_온도_하", description="예측된 HAC_온도_하") # alias 명시
    Steam_압력: float = Field(..., alias="Steam 압력", description="예측된 Steam 압력")
    Cutter_속도: float = Field(..., alias="Cutter 속도", description="예측된 Cutter 속도")
    DS_2_속도: float = Field(..., alias="DS-2 속도", description="예측된 DS-2 속도")
    Spray_농도: float = Field(..., alias="Spray농도", description="예측된 Spray농도")
    분사량: float = Field(..., description="예측된 분사량") # 분사량은 alias가 없어도 됨

    class Config:
        populate_by_name = True 


# --- 4. 예측 함수 정의 ---
def predict_inverse_conditions_simplified(inputs: InverseInput) -> Dict[str, float]:
    
    input_data_dict_internal = inputs.model_dump(by_alias=False) 
    
    encoded_원료 = None
    if '원료' in inverse_features:
        if loaded_inverse_label_encoder_원료:
            try:
                encoded_원료 = loaded_inverse_label_encoder_원료.transform([input_data_dict_internal['원료']])[0]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"입력된 '원료' 값('{input_data_dict_internal['원료']}')은 학습된 LabelEncoder에 없습니다.")
        else:
            raise HTTPException(status_code=500, detail="'원료' LabelEncoder가 로드되지 않았습니다. 모델 로드 단계를 확인하세요.")

    # 예측에 사용할 입력 데이터프레임 생성
    input_df_data = {}
    for feature_col in inverse_features: 
        if feature_col == '원료':
            input_df_data[feature_col] = encoded_원료
        elif feature_col == 'Total Finish': 
            input_df_data[feature_col] = input_data_dict_internal['Total_Finish'] 
        elif feature_col == 'S/P 속도':
            input_df_data[feature_col] = input_data_dict_internal['SP_속도']
        elif feature_col == '방사 속도':
            input_df_data[feature_col] = input_data_dict_internal['방사_속도']
        elif feature_col == 'CAN 수':
            input_df_data[feature_col] = input_data_dict_internal['CAN_수']
        else: # Denier, Elongation, Tenacity, Cohesion (Pydantic 변수명과 모델 컬럼명 동일)
            input_df_data[feature_col] = input_data_dict_internal[feature_col]

    input_df = pd.DataFrame([input_df_data])

    results = {}
    for target_name in inverse_targets:
        if target_name not in loaded_inverse_models:
            # 이 분기문은 로드 로그를 보니 실행되지 않을 것입니다.
            # 하지만 안전을 위해 남겨둡니다.
            raise HTTPException(status_code=500, detail=f"'{target_name}'에 대한 역방향 모델이 로드되지 않았습니다.")
        
        model = loaded_inverse_models[target_name]
        try:
            y_pred = model.predict(input_df)[0]
            
            # Pydantic Output 모델의 alias에 맞게 키를 변환 (더 명확하게 정의)
            # InverseOutput BaseModel의 Field alias에 정의된 이름을 사용해야 합니다.
            # 즉, 파이썬 변수명을 키로 사용하고, 해당 필드의 alias 값을 가져와서 results 딕셔너리의 키로 사용합니다.
            
            # InverseOutput BaseModel의 필드와 해당 alias를 매핑합니다.
            output_model_field_to_alias_map = {
                'DS_1_연신비': 'DS-1 연신비',
                'CR_Box_압력': 'CR Box 압력',
                'CR_Roll_압력': 'CR Roll 압력',
                'HAC_온도_상': 'HAC_온도_상',
                'HAC_온도_하': 'HAC_온도_하',
                'Steam_압력': 'Steam 압력',
                'Cutter_속도': 'Cutter 속도',
                'DS_2_속도': 'DS-2 속도',
                'Spray_농도': 'Spray농도',
                '분사량': '분사량' # 이 필드는 alias가 없으므로 변수명 그대로
            }
            
            # target_name (예: 'DS-1 연신비')을 파이썬 변수명 (예: 'DS_1_연신비')으로 변환
            # 이전에 `output_key = target_name.replace(" ", "_").replace("-", "_")` 했던 것과 유사
            internal_field_name = target_name.replace(" ", "_").replace("-", "_")
            
            # 그리고 실제 results 딕셔너리에 추가할 때는 InverseOutput의 파이썬 필드명을 사용해야 함.
            # 왜냐하면 InverseOutput(**result) 시 Pydantic이 파이썬 필드명을 찾기 때문.
            # alias는 Pydantic이 JSON 직렬화/역직렬화 시에만 사용함.
            results[internal_field_name] = round(float(y_pred), 4)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"'{target_name}' 모델 예측 중 오류 발생: {str(e)}")

    return results

# --- 5. API 엔드포인트 정의 ---

@app.post("/KS-2", response_model=InverseOutput, summary="물성값 및 고정 조건 기반 역방향 조건 예측")
async def direct_inverse_predict(inputs: InverseInput):
    try:
        result = predict_inverse_conditions_simplified(inputs)
        # InverseOutput 생성 시, results 딕셔너리의 키는 InverseOutput 모델의 파이썬 필드명과 일치해야 합니다.
        return InverseOutput(**result)
    except HTTPException as e:
        raise e 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 알 수 없는 오류 발생: {str(e)}")

# --- 6. API 실행 (로컬 서버 구동) ---
if __name__ == "__main__":
    print(f"\nStarting FastAPI server on http://0.0.0.0:8001")
    print(f"Access interactive API documentation (Swagger UI) at http://0.0.0.0:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)