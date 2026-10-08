import pandas as pd

df = pd.read_csv('pasteurizer.csv')
df['STD_DT'] = pd.to_datetime(df['STD_DT'])
df = df.sort_values('STD_DT').reset_index(drop=True)

# 1. 시간 간격 분포 확인
print("=" * 50)
print("1. 시간 간격(분) 분포")
print("=" * 50)
df['interval_min'] = df['STD_DT'].diff().dt.total_seconds() / 60
print(df['interval_min'].value_counts().head(10))

# 2. 월별 데이터 개수
print("\n" + "=" * 50)
print("2. 월별 데이터 개수")
print("=" * 50)
df['month'] = df['STD_DT'].dt.to_period('M')
print(df.groupby('month').size())

# 3. 이상치 개수 확인
print("\n" + "=" * 50)
print("3. 이상치 개수")
print("=" * 50)
print(f"MIXA_STATE > 1: {(df['MIXA_PASTEUR_STATE'] > 1).sum()}")
print(f"MIXB_STATE > 1: {(df['MIXB_PASTEUR_STATE'] > 1).sum()}")
print(f"MIXA_TEMP > 1000: {(df['MIXA_PASTEUR_TEMP'] > 1000).sum()}")
print(f"MIXB_TEMP > 1000: {(df['MIXB_PASTEUR_TEMP'] > 1000).sum()}")

# 4. INSP(품질) 분포
print("\n" + "=" * 50)
print("4. 품질(INSP) 분포")
print("=" * 50)
print(df['INSP'].value_counts())

# 5. TEMP 칼럼의 정상값 범위 (이상치 제외)
print("\n" + "=" * 50)
print("5. TEMP 정상범위 통계 (값 < 1000)")
print("=" * 50)
print("MIXA_TEMP:")
print(df[df['MIXA_PASTEUR_TEMP'] < 1000]['MIXA_PASTEUR_TEMP'].describe())
print("\nMIXB_TEMP:")
print(df[df['MIXB_PASTEUR_TEMP'] < 1000]['MIXB_PASTEUR_TEMP'].describe())