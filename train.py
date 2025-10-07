import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut # Leave-One-Out Cross-Validation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import os
from sklearn.base import clone
import joblib 

# 사용할 파일 경로 상수 정의
TRAIN_FILE_PATH = 'train_all.csv'
TEST_FILE_PATH = 'test_all.csv'
MODEL_FILE = 'final_model.joblib'
SCALER_FILE = 'final_scaler.joblib'
random_state = 96

def load_data(path, target_column='DIAG_NM'):
    """
    데이터를 로드하고 피처(X)와 타겟(y)을 분리합니다.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        print(f"오류: 파일 경로를 찾을 수 없습니다: {e}")
        return None, None
    
    # 불필요한 컬럼 삭제 (Data Leakage 방지 및 ID/TOTAL 삭제)
    # MMSE_NUM은 data_splitter.py에서 제외되었지만 안전하게 TOTAL과 함께 한 번 더 제외
    features_to_drop = ['SAMPLE_EMAIL', target_column, 'TOTAL', 'MMSE_NUM']
    
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    y = df[target_column]
    
    return X, y

def train_and_cross_validate(X_train, y_train, class_names):
    """
    Logistic Regression 모델을 LOOCV로 학습하고 평균 정확도를 계산합니다.
    """
    print(f"\n--- Leave-One-Out Cross-Validation ({len(X_train)}-Fold) 시작 ---")
    
    # LOOCV 객체 생성 (폴드 수 = 전체 학습 샘플 수)
    loo = LeaveOneOut() 
    
    cv_accuracies = []
    dem_w = 1.1
    
    # 🔥 모델 초기화: Dem(치매)에 수동으로 가중치 부여 🔥
    base_model = LogisticRegression(
        # multi_class='multinomial', 
        solver='lbfgs', 
        max_iter=100, 
        random_state=random_state, 
        class_weight={'CN': 1.0, 'Dem': dem_w} # CN은 0, Dem은 1로 인코딩되어 있다고 가정 (Dem에 높은 가중치)
    )
    scaler = StandardScaler()
    
    # LOOCV 루프
    # LOOCV는 폴드 수가 너무 많아 진행 상황을 전부 출력하지 않습니다.
    for fold, (train_index, val_index) in enumerate(loo.split(X_train, y_train)):
        
        # 훈련/검증 데이터 분할 (검증 데이터는 항상 1개)
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # 데이터 스케일링
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = scaler.transform(X_fold_val)
        
        # 모델 학습 및 예측
        model = clone(base_model)
        model.fit(X_fold_train_scaled, y_fold_train)
        y_pred = model.predict(X_fold_val_scaled)
        
        # 정확도 기록
        accuracy = accuracy_score(y_fold_val, y_pred)
        cv_accuracies.append(accuracy)

    print(f"\n============= LOOCV 평균 평가 결과 (가중치: {dem_w}, 시드: {random_state}) =============")
    avg_accuracy = np.mean(cv_accuracies)
    print(f"LOOCV 평균 정확도 (CV Mean Accuracy): {avg_accuracy:.4f}")
    print("==============================================================")
    
    # LOOCV는 모델 선택보다 성능 추정에 중점을 두므로, 
    # 전체 훈련 데이터로 최종 모델을 학습하여 반환합니다.
    X_train_final_scaled = scaler.fit_transform(X_train)
    final_model = clone(base_model)
    final_model.fit(X_train_final_scaled, y_train)

    # 🔥 모델과 스케일러 저장 🔥
    joblib.dump(final_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"✅ 모델 및 스케일러 저장 완료: {MODEL_FILE}, {SCALER_FILE}")
    
    return final_model, scaler, class_names


def evaluate_final_test(final_model, scaler, X_test, y_test, class_names):
    """
    최종 훈련된 모델을 독립적인 테스트 데이터셋으로 평가합니다.
    """
    print("\n--- 최종 모델 테스트 데이터 평가 시작 ---")
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = final_model.predict(X_test_scaled)

    print("\n============= 최종 테스트 평가 결과 =============")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"최종 테스트 정확도 (Test Accuracy): {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print("\n분류 리포트:")
    print(report)
    print("==============================================")


if __name__ == '__main__':
    # Step 1: 데이터 로드
    X_train_raw, y_train = load_data(TRAIN_FILE_PATH)
    X_test_raw, y_test = load_data(TEST_FILE_PATH)
    
    if X_train_raw is not None and X_test_raw is not None:
        
        # 레이블 인코딩
        le = LabelEncoder()
        # y_train과 y_test는 이미 CN/Dem만 존재
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        class_names = le.classes_
        print(f"레이블 매핑: {dict(zip(le.transform(class_names), class_names))}")

        # Step 2: LOOCV 수행
        final_model, final_scaler, _ = train_and_cross_validate(X_train_raw, y_train, class_names)

        # Step 3: 최종 테스트 데이터 평가
        evaluate_final_test(final_model, final_scaler, X_test_raw, y_test, class_names)