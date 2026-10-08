import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# 데이터 로드
train = pd.read_csv('train.csv', parse_dates=['STD_DT'], index_col='STD_DT')
val = pd.read_csv('val.csv', parse_dates=['STD_DT'], index_col='STD_DT')

# 타깃: MIXA_PASTEUR_TEMP (1분 후 예측)
TARGET = 'MIXA_PASTEUR_TEMP'

# 검증셋의 실제값
y_true = val[TARGET].values

results = {}

# ============================
# 베이스라인 1: Naive (직전값 그대로)
# ============================
# 검증셋의 직전값 = train 마지막 + val 직전값들
combined = pd.concat([train[TARGET], val[TARGET]])
y_pred_naive = combined.shift(1).loc[val.index].values
mse_naive = mean_squared_error(y_true, y_pred_naive)
results['Naive (직전값)'] = mse_naive

# ============================
# 베이스라인 2: 24시간 전 동시각 값
# ============================
y_pred_24h = combined.shift(60 * 24).loc[val.index].values
# NaN 있으면 평균값으로 대체
mask = ~np.isnan(y_pred_24h)
mse_24h = mean_squared_error(y_true[mask], y_pred_24h[mask])
results['24시간 전 값'] = mse_24h

# ============================
# 베이스라인 3: 60분 이동평균
# ============================
y_pred_ma = combined.rolling(window=60).mean().shift(1).loc[val.index].values
mask = ~np.isnan(y_pred_ma)
mse_ma = mean_squared_error(y_true[mask], y_pred_ma[mask])
results['60분 이동평균'] = mse_ma

# ============================
# 베이스라인 4: 학습셋 전체 평균
# ============================
y_pred_mean = np.full_like(y_true, train[TARGET].mean())
mse_mean = mean_squared_error(y_true, y_pred_mean)
results['전체 평균'] = mse_mean

# ============================
# 결과 출력
# ============================
print("=" * 50)
print(f"베이스라인 MSE 결과 (타깃: {TARGET})")
print("=" * 50)
for name, mse in sorted(results.items(), key=lambda x: x[1]):
    rmse = np.sqrt(mse)
    print(f"{name:20s} | MSE: {mse:10.2f} | RMSE: {rmse:6.2f}")

print("\n" + "=" * 50)
print("해석")
print("=" * 50)
best_name = min(results, key=results.get)
print(f"최고 베이스라인: {best_name}")
print(f"본 모델은 이 MSE({results[best_name]:.2f})를 반드시 이겨야 함")
print(f"RMSE {np.sqrt(results[best_name]):.2f} = 평균적으로 {np.sqrt(results[best_name]):.1f} 단위 오차")
print(f"실제 온도 단위로는 약 {np.sqrt(results[best_name])/10:.2f}℃ 오차")