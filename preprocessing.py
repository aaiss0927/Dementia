# data_splitter.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 사용할 파일 경로 상수 정의
TRAIN_FILE = './train_mmse.csv'
VAL_FILE = './val_mmse.csv'
OUTPUT_TRAIN_FILE = 'train_all.csv'
OUTPUT_TEST_FILE = 'test_all.csv'
random_state = 96

def prepare_and_split_data(train_path, val_path, test_size=0.2, random_state=random_state):
    """
    1. Train/Val 파일을 합치고
    2. MCI 행을 제외하며
    3. 불필요 컬럼을 삭제하고
    4. 클래스 균형을 맞춰 훈련/테스트 데이터로 분할 후 저장합니다.
    """
    print("--- 데이터 병합 및 초기 로드 ---")
    try:
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_combined = pd.concat([df_train, df_val], ignore_index=True)
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다: {e}")
        return

    original_rows = len(df_combined)

    q_columns = df_combined.columns[df_combined.columns.str.startswith('Q')]
    df_combined.loc[:, q_columns] = df_combined.loc[:, q_columns] - 1
    
    # 1. MCI 행 제외
    # df_filtered = df_combined[df_combined['DIAG_NM'] != 'MCI'].copy()
    
    # removed_rows = original_rows - len(df_filtered)
    # print(f"✅ 'MCI' 행 {removed_rows}개 제외됨. (남은 행: {len(df_filtered)})")
    df_combined.loc[df_combined['DIAG_NM'] == 'MCI', 'DIAG_NM'] = 'Dem'
    df_filtered = df_combined.copy()
    
    # 2. 불필요한 컬럼 및 누수 컬럼 제거
    # MMSE_NUM과 TOTAL을 제거하여 데이터 누수 경로를 차단합니다.
    columns_to_drop = ['DOCTOR_NM', 'Q12_TOTAL', 'MMSE_NUM', 'TOTAL']
    
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_filtered.columns]
    
    if existing_cols_to_drop:
        df_final = df_filtered.drop(columns=existing_cols_to_drop, axis=1)
        print(f"✅ {existing_cols_to_drop} 컬럼 삭제됨. (데이터 누수 방지)")
    else:
        df_final = df_filtered
        print("삭제할 컬럼이 파일에 없습니다.")

    

    print(f"CN/Dem 최종 데이터셋 크기: {len(df_final)}")
    print("CN/Dem 클래스 분포:")
    print(df_final['DIAG_NM'].value_counts())
    
    # 3. 클래스 균형(stratify)을 맞춘 훈련/테스트 분할
    print(f"\n--- 데이터 분할 (Train: {1.0 - test_size:.0%}, Test: {test_size:.0%}) ---")
    
    # stratify=df_final['DIAG_NM']를 사용하여 CN과 Dem의 비율을 테스트셋에서도 유지합니다.
    df_train_set, df_test_set = train_test_split(
        df_final, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True, # 셔플은 기본값으로 수행
        stratify=df_final['DIAG_NM']
    )
    
    # 4. 결과 저장
    df_train_set.to_csv(OUTPUT_TRAIN_FILE, index=False)
    df_test_set.to_csv(OUTPUT_TEST_FILE, index=False)
    
    print(f"✅ 학습 데이터 저장 완료: {OUTPUT_TRAIN_FILE} (행: {len(df_train_set)})")
    print(f"✅ 테스트 데이터 저장 완료: {OUTPUT_TEST_FILE} (행: {len(df_test_set)})")
    print("분할된 파일로 모델을 다시 학습/평가해 주세요.")


if __name__ == '__main__':
    # ⚠️ 경고: 기존 train_mmse.csv와 val_mmse.csv 파일명을 사용합니다.
    # 만약 전처리된 파일명을 사용하고 싶다면, TRAIN_FILE과 VAL_FILE을 적절히 수정해야 합니다.
    prepare_and_split_data(TRAIN_FILE, VAL_FILE)