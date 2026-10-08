import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정 (Mac)
mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

# 출력 폴더 생성
import os
os.makedirs('ppt_images', exist_ok=True)

# ============================
# 그래프 1: 베이스라인 vs XGBoost MSE 비교
# ============================
fig, ax = plt.subplots(figsize=(10, 6))

models = ['전체 평균', 'Naive\n(직전값)', '30분 전\n값', '60분\n이동평균', 'XGBoost\n(본 모델)']
mse_values = [3100.29, 2067.73, 1736.87, 1317.06, 715.71]
colors = ['#cccccc', '#cccccc', '#cccccc', '#888888', '#2E86AB']

bars = ax.bar(models, mse_values, color=colors, edgecolor='black', linewidth=1.2)
ax.set_ylabel('MSE (Mean Squared Error)', fontsize=12, fontweight='bold')
ax.set_title('30분 후 살균온도 예측 — 모델별 MSE 비교', fontsize=14, fontweight='bold', pad=15)

# 값 표시
for bar, val in zip(bars, mse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 개선율 화살표
ax.annotate('', xy=(4, 715), xytext=(3, 1317),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(3.5, 1050, '-45.7%', color='red', fontsize=13, fontweight='bold', ha='center')

ax.set_ylim(0, 3500)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('ppt_images/01_mse_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 01_mse_comparison.png 저장")

# ============================
# 그래프 2: Feature Importance Top 10
# ============================
features = [
    '최근 10분 평균',
    '1분 전 값',
    'B살균기 1분 전',
    '최근 60분 평균',
    '2분 전 값',
    '5분 전 값',
    '30분 전 값',
    '최근 30분 평균',
    '120분 전 값',
    '최근 30분 표준편차'
]
importance = [0.841, 0.099, 0.019, 0.009, 0.006, 0.003, 0.003, 0.002, 0.002, 0.002]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(features))[::-1]
colors_imp = ['#2E86AB' if i == 0 else '#A8C5DA' for i in range(len(features))]

bars = ax.barh(y_pos, importance, color=colors_imp, edgecolor='black', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=11)
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title('XGBoost Feature Importance — 무엇이 30분 후 온도를 결정하는가', 
             fontsize=14, fontweight='bold', pad=15)

for bar, val in zip(bars, importance):
    ax.text(val + 0.015, bar.get_y() + bar.get_height()/2,
            f'{val*100:.1f}%', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0, 1.0)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('ppt_images/02_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 02_feature_importance.png 저장")

# ============================
# 그래프 3: 예측 vs 실제 시계열
# ============================
pred_df = pd.read_csv('xgb_predictions.csv', parse_dates=['STD_DT'], index_col='STD_DT')

# 검증셋에서 가장 변동 큰 구간 1000개 골라서 시각화
sample = pred_df.iloc[5000:6000]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sample.index, sample['target_30min'].values / 10, 
        label='실제값', color='#333333', linewidth=1.5, alpha=0.9)
ax.plot(sample.index, sample['prediction'].values / 10, 
        label='XGBoost 예측', color='#E63946', linewidth=1.5, alpha=0.8)

ax.set_xlabel('시간', fontsize=12, fontweight='bold')
ax.set_ylabel('살균온도 (℃)', fontsize=12, fontweight='bold')
ax.set_title('30분 후 살균온도 예측 — 실제 vs 예측 (검증셋 일부)', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('ppt_images/03_prediction_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 03_prediction_vs_actual.png 저장")

# ============================
# 그래프 4: 데이터 정제 전후 비교
# ============================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 원본 (이상치 포함)
df_raw = pd.read_csv('pasteurizer.csv')
axes[0].hist(df_raw['MIXB_PASTEUR_TEMP'].dropna(), bins=100, 
             color='#E63946', edgecolor='black', alpha=0.7)
axes[0].set_title('정제 전 — 이상치 포함\n(최댓값 4,279만)', 
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('MIXB_PASTEUR_TEMP', fontsize=11)
axes[0].set_ylabel('빈도', fontsize=11)
axes[0].set_yscale('log')

# 정제 후
df_clean = df_raw[df_raw['MIXB_PASTEUR_TEMP'] < 1000]
axes[1].hist(df_clean['MIXB_PASTEUR_TEMP'].dropna(), bins=100, 
             color='#2E86AB', edgecolor='black', alpha=0.7)
axes[1].set_title('정제 후 — 100개 이상치 제거\n(0~793 정상범위)', 
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('MIXB_PASTEUR_TEMP', fontsize=11)
axes[1].set_ylabel('빈도', fontsize=11)

plt.suptitle('데이터 정제 — 시스템 오류값 100개 제거', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ppt_images/04_data_cleaning.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 04_data_cleaning.png 저장")

print("\n" + "=" * 50)
print("PPT 이미지 4장 생성 완료 → ppt_images/ 폴더 확인")
print("=" * 50)