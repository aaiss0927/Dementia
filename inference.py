import pandas as pd
import numpy as np
import joblib
import os

# 모델 및 스케일러 파일 경로
MODEL_FILE = 'final_model.joblib'
SCALER_FILE = 'final_scaler.joblib'

def load_model_and_scaler():
    """
    저장된 Logistic Regression 모델과 StandardScaler를 불러옵니다.
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print(f"오류: 모델 파일 ({MODEL_FILE}) 또는 스케일러 파일 ({SCALER_FILE})을 찾을 수 없습니다.")
        print("train.py를 먼저 실행하여 모델을 저장하세요.")
        return None, None
        
    print(f"--- 모델 및 스케일러 로드 완료 ---")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

def run_inference(new_sample_data, model, scaler):
    """
    로드된 모델을 사용하여 새로운 샘플에 대한 추론을 실행합니다.
    """
    if model is None or scaler is None:
        return "모델이 로드되지 않았습니다."

    # 1. 입력 데이터 준비 (컬럼 순서 유지 및 DataFrame 변환)
    # 추론 시에도 학습 시 사용한 컬럼 순서와 이름, 형태를 유지해야 함
    
    # 학습 데이터의 피처 목록 (Q01 ~ Q19, Q13_1~3 등 총 30개 피처)
    # 이 목록은 train_all.csv의 피처 목록과 일치해야 합니다.
    # 안전을 위해 샘플 데이터의 컬럼을 하드 코딩하여 순서를 맞춥니다.
    feature_columns = ['DIAG_SEQ', 'MMSE_KIND', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08', 'Q09', 'Q10', 
                       'Q11_1', 'Q11_2', 'Q11_3', 'Q12_1', 'Q12_2', 'Q12_3', 'Q12_4', 'Q12_5', 'Q13_1', 'Q13_2', 
                       'Q13_3', 'Q14_1', 'Q14_2', 'Q15', 'Q16_1', 'Q16_2', 'Q16_3', 'Q17', 'Q18', 'Q19']

    # 딕셔너리 또는 리스트를 DataFrame으로 변환
    if isinstance(new_sample_data, dict):
         df_sample = pd.DataFrame([new_sample_data], columns=feature_columns)
    elif isinstance(new_sample_data, list):
         df_sample = pd.DataFrame(new_sample_data, columns=feature_columns)
    else:
        return "유효하지 않은 입력 데이터 형식입니다."
    
    
    # 2. 스케일링 적용
    X_sample_scaled = scaler.transform(df_sample)
    
    # 3. 예측 및 확률 추론
    prediction = model.predict(X_sample_scaled)[0]
    probabilities = model.predict_proba(X_sample_scaled)[0]
    
    # 클래스 이름 (학습 시 CN/Dem 순서로 인코딩되었음)
    classes = model.classes_ 
    
    result = {
        "진단_예측": prediction,
        "확률": dict(zip(classes, probabilities)),
        "Dem_확률": probabilities[np.where(classes == 'Dem')[0][0]] # Dem 클래스의 확률만 별도 출력
    }
    
    return result

if __name__ == '__main__':
    model, scaler = load_model_and_scaler()
    
    if model and scaler:
        # 새로운 가상의 환자 데이터 (학습에 사용된 32개 피처를 모두 포함해야 함)
        # 예시: 정상에 가까운 환자 데이터 (대부분 2점)
        # 주의: 이 예시는 학습 데이터의 피처 컬럼을 임의로 맞춘 것이므로 실제 추론 시에는 정확한 컬럼 이름/순서를 확인해야 합니다.
        sample_data = {
            'DIAG_SEQ': 1, 'MMSE_KIND': 2, 'Q01': 1, 'Q02': 1, 'Q03': 1, 'Q04': 1, 'Q05': 1, 'Q06': 1, 
            'Q07': 1, 'Q08': 1, 'Q09': 1, 'Q10': 1, 'Q11_1': 1, 'Q11_2': 1, 'Q11_3': 1, 'Q12_1': 1, 
            'Q12_2': 1, 'Q12_3': 1, 'Q12_4': 1, 'Q12_5': 1, 'Q13_1': 1, 'Q13_2': 1, 'Q13_3': 1, 
            'Q14_1': 1, 'Q14_2': 1, 'Q15': 1, 'Q16_1': 1, 'Q16_2': 1, 'Q16_3': 1, 'Q17': 1, 'Q18': 1, 'Q19': 1
        }
        
        # 실제 치매에 가까운 환자 데이터 (점수가 낮음)
        # sample_data = {
        #     'DIAG_SEQ': 1, 'MMSE_KIND': 2, 'Q01': 0, 'Q02': 0, 'Q03': 0, 'Q04': 0, 'Q05': 0, 'Q06': 0, 
        #     'Q07': 0, 'Q08': 0, 'Q09': 0, 'Q10': 0, 'Q11_1': 0, 'Q11_2': 0, 'Q11_3': 0, 'Q12_1': 0, 
        #     'Q12_2': 0, 'Q12_3': 0, 'Q12_4': 0, 'Q12_5': 0, 'Q13_1': 0, 'Q13_2': 0, 'Q13_3': 0, 
        #     'Q14_1': 0, 'Q14_2': 0, 'Q15': 0, 'Q16_1': 0, 'Q16_2': 0, 'Q16_3': 0, 'Q17': 0, 'Q18': 0, 'Q19': 0
        # }
        
        inference_result = run_inference(sample_data, model, scaler)
        print("\n--- 추론 결과 ---")
        print(inference_result)