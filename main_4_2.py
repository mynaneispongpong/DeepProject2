import os
import glob
import numpy as np
import pandas as pd

# 1. 데이터셋 경로 설정 (클론한 폴더 이름에 맞게 수정)
dataset_dir = 'sEMG-Hand-Gesture-Elbow-Angle-Dataset'

# 2. 데이터 파일 검색 (예: csv, npy, mat 등 데이터 포맷에 맞게 확장자 변경 가능)
# 여기서는 일반적인 sEMG 구조인 csv 파일을 기준으로 탐색합니다.
search_pattern = os.path.join(dataset_dir, '**', '*.csv')
files = sorted(glob.glob(search_pattern, recursive=True))

print(f'총 발견된 파일 수: {len(files)}')

summary_data = []

if len(files) == 0:
    print("경고: 지정된 경로에서 파일을 찾지 못했습니다. 폴더 경로와 데이터 포맷(확장자)을 확인해주세요.")
else:
    for file_path in files:
        try:
            # 데이터 파일 읽기 (CSV 기준)
            df = pd.read_csv(file_path)
            data_array = df.values

            # 배열 모양(Shape) 및 채널 수 추출
            array_shape = data_array.shape  # 예: (타임스텝, 채널 수)
            num_channels = array_shape[1] if len(array_shape) > 1 else 1

            # 파일 이름 및 경로를 기반으로 클래스(Gesture) 정보 추출
            # 예시: 파일명이나 상위 폴더명에 'class', 'gesture', 'label' 등이 포함되어 있다고 가정
            file_name = os.path.basename(file_path)
            parent_folder = os.path.basename(os.path.dirname(file_path))

            # 클래스 구분 로직 (프로젝트 구조에 맞게 수정 필요)
            class_label = parent_folder if parent_folder != 'data' else file_name.split('_')[0]

            summary_data.append({
                '파일명': file_name,
                '클래스': class_label,
                '배열 모양 (Shape)': array_shape,
                '채널 수': num_channels
            })
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({file_path}): {e}")

    # 3. 데이터 요약 표 변환 및 출력
    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # 클래스별 시행(Trial) 수 집계
        class_counts = summary_df['클래스'].value_counts().reset_index()
        class_counts.columns = ['클래스', '시행 수 (Trial Count)']

        print("\n" + "=" * 40)
        print(" 1. 클래스별 시행 수 요약 표 ")
        print("=" * 40)
        print(class_counts.to_string(index=False))

        print("\n" + "=" * 40)
        print(" 2. 데이터 배열 모양 및 채널 수 상세 요약 표 ")
        print("=" * 40)
        # 가독성을 위해 상위 10개만 먼저 출력하거나 전체 출력
        print(summary_df[['파일명', '클래스', '배열 모양 (Shape)', '채널 수']].head(20).to_string(index=False))

        if len(summary_df) > 20:
            print(f"\n... (이하 생략, 총 {len(summary_df)}개 파일)")