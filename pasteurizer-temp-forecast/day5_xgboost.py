import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# 데이터 로드
train = pd.read_csv('train.csv', parse_dates=['STD_DT'], index_col='STD_DT')
val = pd.read_csv('val.csv', parse_dates=['STD_DT'], index_col='STD_DT')

TARGET = 'MIXA_PASTEUR_TEMP'
HORIZON = 30  # 30분 후 예측

# train + val 합치고 feature 생성 후 다시 분할
df = pd.concat([train, val])

# ============================
# Feature Engineering
# ============================
def make_features(df, target_col):
    df = df.copy()
    
    # Lag features (과거값)
    for lag in [1, 2, 5, 10, 30, 60, 120]:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics (이동 통계)
    for window in [10, 30, 60]:
        df[f'{target_col}_rollmean{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rollstd{window}'] = df[target_col].rolling(window).std()
    
    # MIXB 온도도 보조 feature로 사용
    other = 'MIXB_PASTEUR_TEMP'
    for lag in [1, 30, 60]:
        df[f'{other}_lag{lag}'] = df[other].shift(lag)
    
    # 시간 feature
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['minute'] = df.index.minute
    
    # 타깃: 30분 후 값
    df['target_30min'] = df[target_col].shift(-HORIZON)
    
    return df

df_feat = make_features(df, TARGET)
df_feat = df_feat.dropna()  # NaN 제거 (lag 때문에 처음/끝 일부 손실)

# 학습/검증 분리
val_start = val.index[0]
train_data = df_feat[df_feat.index < val_start]
val_data = df_feat[df_feat.index >= val_start]

print(f"학습 데이터: {train_data.shape}")
print(f"검증 데이터: {val_data.shape}")

# Feature 칼럼만 선택 (target 제외)
feature_cols = [c for c in df_feat.columns if c != 'target_30min' 
                and c not in [TARGET, 'MIXB_PASTEUR_TEMP']]

X_train = train_data[feature_cols]
y_train = train_data['target_30min']
X_val = val_data[feature_cols]
y_val = val_data['target_30min']

print(f"Feature 개수: {len(feature_cols)}")

# ============================
# XGBoost 학습
# ============================
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20,
    eval_metric='rmse'
)

print("\n학습 시작...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# ============================
# 평가
# ============================
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print("\n" + "=" * 50)
print("XGBoost 결과")
print("=" * 50)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"온도 오차: 약 {rmse/10:.2f}℃")
print(f"\n베이스라인(60분 이동평균) MSE: 1317.06")
print(f"개선율: {(1317.06 - mse) / 1317.06 * 100:.1f}%")

# ============================
# Feature importance (PPT 재료)
# ============================
print("\n" + "=" * 50)
print("Feature Importance Top 10")
print("=" * 50)
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(10).to_string(index=False))

# 결과 저장
val_data_out = val_data.copy()
val_data_out['prediction'] = y_pred
val_data_out[['target_30min', 'prediction']].to_csv('xgb_predictions.csv')
print("\nxgb_predictions.csv 저장 완료")