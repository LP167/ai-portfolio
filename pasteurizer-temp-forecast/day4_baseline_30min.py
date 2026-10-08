import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

train = pd.read_csv('train.csv', parse_dates=['STD_DT'], index_col='STD_DT')
val = pd.read_csv('val.csv', parse_dates=['STD_DT'], index_col='STD_DT')

TARGET = 'MIXA_PASTEUR_TEMP'
HORIZON = 30  # 30분 후 예측

combined = pd.concat([train[TARGET], val[TARGET]])

# 검증셋의 30분 후 실제값을 타깃으로
y_true = combined.shift(-HORIZON).loc[val.index].values
mask = ~np.isnan(y_true)

results = {}

# 1. Naive: 현재값을 30분 후 예측값으로 사용
y_pred_naive = combined.loc[val.index].values
results['Naive (현재값=30분후 예측)'] = mean_squared_error(y_true[mask], y_pred_naive[mask])

# 2. 60분 이동평균
y_pred_ma = combined.rolling(window=60).mean().loc[val.index].values
mask2 = mask & ~np.isnan(y_pred_ma)
results['60분 이동평균'] = mean_squared_error(y_true[mask2], y_pred_ma[mask2])

# 3. 전체 평균
y_pred_mean = np.full(mask.sum(), train[TARGET].mean())
results['전체 평균'] = mean_squared_error(y_true[mask], y_pred_mean)

# 4. 30분 전 값 (=계절성 가정)
y_pred_30ago = combined.shift(HORIZON).loc[val.index].values
mask3 = mask & ~np.isnan(y_pred_30ago)
results['30분 전 값'] = mean_squared_error(y_true[mask3], y_pred_30ago[mask3])

print("=" * 50)
print(f"30분 후 예측 베이스라인 MSE")
print("=" * 50)
for name, mse in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:30s} | MSE: {mse:8.2f} | RMSE: {np.sqrt(mse):5.2f}")

best = min(results, key=results.get)
print(f"\n최고 베이스라인: {best}")
print(f"본 모델은 MSE {results[best]:.2f}를 이겨야 함")