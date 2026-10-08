import pandas as pd
import numpy as np

df = pd.read_csv('pasteurizer.csv')
df['STD_DT'] = pd.to_datetime(df['STD_DT'])
df = df.sort_values('STD_DT').reset_index(drop=True)

print("원본 크기:", df.shape)

# 1. 이상치 제거
df = df[~((df['MIXA_PASTEUR_STATE'] > 1) | (df['MIXB_PASTEUR_STATE'] > 1))]
df = df[df['MIXB_PASTEUR_TEMP'] < 1000]
print("이상치 제거 후:", df.shape)

# 2. 중복 시간 제거 (같은 STD_DT 첫 행만 유지)
df = df.drop_duplicates(subset='STD_DT', keep='first')
print("중복 시간 제거 후:", df.shape)

# 3. TEMP 값 0인 행 = 가동 정지로 간주, 모델 학습에서 제외
df = df[(df['MIXA_PASTEUR_TEMP'] > 0) & (df['MIXB_PASTEUR_TEMP'] > 0)]
print("TEMP 0 제거 후:", df.shape)

# 4. STATE 칼럼은 결측 95%라 제거
df = df.drop(columns=['MIXA_PASTEUR_STATE', 'MIXB_PASTEUR_STATE'])

# 5. 시간순 정렬 + 1분 단위 리샘플링
df = df.set_index('STD_DT').sort_index()
df_resampled = df[['MIXA_PASTEUR_TEMP', 'MIXB_PASTEUR_TEMP']].resample('1min').mean()
print("1분 리샘플링 후:", df_resampled.shape)
print("리샘플링 후 결측치:")
print(df_resampled.isna().sum())

# 6. 결측 보간 (선형) — 1분 단위니까 보간 안전
df_resampled = df_resampled.interpolate(method='linear', limit=60)
df_resampled = df_resampled.dropna()
print("보간 후:", df_resampled.shape)

# 7. 시계열 분할 — 80% 학습 / 10% 검증 / 10% 테스트
n = len(df_resampled)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train = df_resampled.iloc[:train_end]
val = df_resampled.iloc[train_end:val_end]
test = df_resampled.iloc[val_end:]

print("\n" + "=" * 50)
print("최종 분할 결과")
print("=" * 50)
print(f"학습: {train.shape}, 기간: {train.index.min()} ~ {train.index.max()}")
print(f"검증: {val.shape}, 기간: {val.index.min()} ~ {val.index.max()}")
print(f"테스트: {test.shape}, 기간: {test.index.min()} ~ {test.index.max()}")

# 8. CSV로 저장 (Day 4에서 모델 학습에 사용)
train.to_csv('train.csv')
val.to_csv('val.csv')
test.to_csv('test.csv')
print("\ntrain.csv, val.csv, test.csv 저장 완료")