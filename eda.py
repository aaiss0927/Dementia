# eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 파일 경로 상수 정의
TRAIN_FILE = './train_mmse.csv'
VAL_FILE = './val_mmse.csv'

def load_and_combine_data(train_path, val_path):
    """
    훈련 및 검증 파일을 로드하고 병합하여 원본 전체 데이터셋을 반환합니다.
    """
    try:
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_combined = pd.concat([df_train, df_val], ignore_index=True)
        print(f"✅ 원본 데이터 로드 및 병합 완료. 총 샘플 수: {len(df_combined)}")
        return df_combined
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다: {e}")
        return None

def run_exploratory_data_analysis(df):
    """
    탐색적 데이터 분석(EDA)을 수행하고 주요 시각화 결과를 출력합니다.
    """
    if df is None:
        return

    print("\n" + "="*50)
    print("1. 데이터 기본 정보")
    print("="*50)
    print(df.info())
    print("\n상위 5개 행:")
    print(df.head())
    
    print("\n" + "="*50)
    print("2. 진단 클래스 (DIAG_NM) 분포 및 불균형 분석")
    print("="*50)
    class_counts = df['DIAG_NM'].value_counts()
    print(class_counts)
    
    # 클래스 분포 시각화
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution of Diagnostic Classes (CN, MCI, Dem)')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.show()

    # 3. MCI/Dem 통합 결정의 근거 분석 (MMSE_NUM 활용)
    print("\n" + "="*50)
    print("3. 주요 진단 지표 (MMSE_NUM)와 클래스 관계 분석")
    print("="*50)
    
    # MMSE_NUM 결측치 제거 후 분석
    df_plot = df.dropna(subset=['MMSE_NUM', 'DIAG_NM'])

    plt.figure(figsize=(10, 6))
    # 박스 플롯: 각 클래스별 MMSE 점수 분포 확인
    sns.boxplot(x='DIAG_NM', y='MMSE_NUM', data=df_plot, order=['CN', 'MCI', 'Dem'])
    
    # CNA - Box Plot (CN/MCI/Dem 별 MMSE_NUM 분포) 
    
    plt.title('MMSE Score Distribution by Diagnosis (Justification for MCI -> Dem)')
    plt.xlabel('Diagnosis')
    plt.ylabel('MMSE Score')
    plt.grid(axis='y', linestyle='--')
    plt.show()


if __name__ == '__main__':
    # Step 1: 데이터 로드 및 병합
    df_combined = load_and_combine_data(TRAIN_FILE, VAL_FILE)
    
    # Step 2: EDA 실행
    if df_combined is not None:
        run_exploratory_data_analysis(df_combined)