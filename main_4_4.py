import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 경로 설정 및 파일 검색
dataset_dir = 'sEMG-Hand-Gesture-Elbow-Angle-Dataset'
search_pattern = os.path.join(dataset_dir, '**', '*.csv')
files = sorted(glob.glob(search_pattern, recursive=True))

print(f'총 발견된 파일 수: {len(files)}')


# 2. 동일 동작에 대해 30°와 60° 파일을 찾아주는 함수 (파일명 규칙에 맞게 조정)
def find_angle_files(files, gesture_keyword):
    file_30 = next((f for f in files if '30' in os.path.basename(f) and gesture_keyword in f), None)
    file_60 = next((f for f in files if '60' in os.path.basename(f) and gesture_keyword in f), None)
    return file_30, file_60


# 테스트할 동작 키워드 설정 (예: 'fist' 또는 데이터셋 내 실제 동작 이름)
sample_gesture = 'fist'
f_30, f_60 = find_angle_files(files, sample_gesture)

if f_30 and f_60:
    print(f'30° 파일: {f_30}')
    print(f'60° 파일: {f_60}')

    # 데이터 로드 (이미지 소스 코드 참고)[cite: 4]
    x_30 = np.loadtxt(f_30, delimiter=',', skiprows=1)
    x_60 = np.loadtxt(f_60, delimiter=',', skiprows=1)

    t_30 = np.arange(len(x_30)) / 1000.0
    t_60 = np.arange(len(x_60)) / 1000.0
    names = ['ECR', 'ED', 'FDS', 'Biceps', 'Triceps']

    # 3. 5채널 비교 플롯 생성 (좌측: 30°, 우측: 60°)
    fig, ax = plt.subplots(5, 2, figsize=(14, 9), sharex=True, sharey='row')

    for ch in range(5):
        # --- 30° 신호 플롯 (왼쪽) ---
        ax[ch, 0].plot(t_30, x_30[:, ch], lw=0.5, color='royalblue')
        ax[ch, 0].set_ylabel(names[ch], fontsize=9)
        ax[ch, 0].axvline(2.0, color='r', ls='--')  # 동작 시작 구간 표시[cite: 4]
        ax[ch, 0].axvline(3.0, color='r', ls='--')  # 동작 종료 구간 표시[cite: 4]
        if ch == 0:
            ax[ch, 0].set_title(f'Gesture: {sample_gesture} (30°)')

        # --- 60° 신호 플롯 (오른쪽) ---
        ax[ch, 1].plot(t_60, x_60[:, ch], lw=0.5, color='darkorange')
        ax[ch, 1].axvline(2.0, color='r', ls='--')
        ax[ch, 1].axvline(3.0, color='r', ls='--')
        if ch == 0:
            ax[ch, 1].set_title(f'Gesture: {sample_gesture} (60°)')

    ax[4, 0].set_xlabel('time (s)')
    ax[4, 1].set_xlabel('time (s)')

    plt.tight_layout()
    plt.savefig('comparison_30_60.png', dpi=120)
    plt.show()
else:
    print("지정한 조건의 30° 및 60° 파일을 찾지 못했습니다. 파일명 규칙이나 동작 키워드를 확인해주세요.")