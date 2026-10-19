import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 1. 데이터셋 경로 설정
dataset_dir = 'sEMG-Hand-Gesture-Elbow-Angle-Dataset'

# 하위 폴더의 모든 CSV 파일 검색
search_pattern = os.path.join(dataset_dir, '**', '*.csv')
files = sorted(glob.glob(search_pattern, recursive=True))

print(f'총 발견된 파일 수: {len(files)}')


# 2. 클래스 레이블 추출 함수 (데이터셋의 파일 구조에 맞게 수정 필요)
def label_of(path):
    # 예시: 파일 이름이나 상위 디렉토리 구조에서 클래스 정보를 파싱
    file_name = os.path.basename(path)
    parent_folder = os.path.basename(os.path.dirname(path))

    # 폴더명 또는 파일명 패턴에 따른 분류 로직
    return parent_folder if parent_folder != dataset_dir else file_name.split('_')[0]


if len(files) > 0:
    # 클래스별 개수 확인
    labels = [label_of(f) for f in files]
    print("\n=== [클래스별 개수 (Counter)] ===")
    print(Counter(labels))

    # 3. 첫 번째 파일 로드 및 배열 모양/채널 수 확인
    # CSV 파일인 경우 pandas 또는 np.loadtxt 사용
    df = pd.read_csv(files[0])
    x = df.values

    print('\n=== [데이터 배열 정보] ==0')
    print(f'예시 파일: {files[0]}')
    print('배열 모양 (Shape):', x.shape)  # 예: (5000, 5) 등
    print('채널 수:', x.shape[1] if len(x.shape) > 1 else 1)

    # 4. 5채널 파형 그림 및 동작 구간 표시 시각화
    if len(x.shape) > 1 and x.shape[1] >= 1:
        num_channels = x.shape[1]
        plt.figure(figsize=(12, 2 * num_channels))

        # 동작 구간 설정 (예시: 전체 샘플의 20% ~ 60 구간을 동작 구간으로 가정)
        action_start = int(len(x) * 0.2)
        action_end = int(len(x) * 0.6)

        for i in range(num_channels):
            plt.subplot(num_channels, 1, i + 1)
            plt.plot(x[:, i], color='teal', linewidth=0.8)

            # 동작 구간 음영 표시 (Action Period)
            plt.axvspan(action_start, action_end, color='orange', alpha=0.3,
                        label='Action Period (동작 구간)' if i == 0 else "")

            plt.ylabel(f'Ch {i + 1}')
            plt.grid(True, linestyle='--', alpha=0.5)

            if i == 0:
                plt.title('sEMG 5-Channel Waveform & Action Region')
                plt.legend(loc='upper right')
            if i == num_channels - 1:
                plt.xlabel('Sample Index')

        plt.tight_layout()
        plt.show()
    else:
        print("데이터 차원이 1차원이거나 시각화하기에 적절하지 않습니다.")
else:
    print(f"경고: '{dataset_dir}' 경로 내에서 CSV 파일을 찾을 수 없습니다. git clone 여부와 폴더 경로를 확인해주세요.")