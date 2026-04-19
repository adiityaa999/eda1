import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ===================== STYLE =====================
plt.style.use('dark_background')
sns.set_palette("coolwarm")

# ===================== LOAD DATA =====================
df = pd.read_csv("aqi_data.csv")

# ===================== CLEAN DATA =====================
df['SO2'] = pd.to_numeric(df['SO2'], errors='coerce')
df['NO2'] = pd.to_numeric(df['NO2'], errors='coerce')
df['SPM'] = pd.to_numeric(df['SPM'], errors='coerce')
df['RSPM/PM10'] = pd.to_numeric(df['RSPM/PM10'], errors='coerce')

df['Sampling Date'] = pd.to_datetime(df['Sampling Date'], errors='coerce')

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(subset=['Sampling Date'], inplace=True)

print(" Data cleaned, rows:", len(df))

# ===================== FEATURES =====================
df['year'] = df['Sampling Date'].dt.year
df['month'] = df['Sampling Date'].dt.month

# ===================== OUTPUT FOLDER =====================
os.makedirs("outputs", exist_ok=True)

# ===================== EDA (10 VISUALS) =====================

# 1. PM10 Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['RSPM/PM10'], kde=True, color='cyan')
plt.title("PM10 Distribution", fontsize=14)
plt.grid(alpha=0.3)
plt.savefig("outputs/1_pm10_dist.png")
plt.clf()

# 2. Time Series
plt.figure(figsize=(10,5))
df.groupby('Sampling Date')['RSPM/PM10'].mean().plot(color='orange')
plt.title("PM10 Over Time", fontsize=14)
plt.grid(alpha=0.3)
plt.savefig("outputs/2_timeseries.png")
plt.clf()

# 3. Moving Average
plt.figure(figsize=(10,5))
df['RSPM/PM10'].rolling(30).mean().plot(color='lime')
plt.title("30-Day Moving Average", fontsize=14)
plt.grid(alpha=0.3)
plt.savefig("outputs/3_moving_avg.png")
plt.clf()

# 4. Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[['SO2','NO2','SPM','RSPM/PM10']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("outputs/4_heatmap.png")
plt.clf()

# 5. Scatter Plot
plt.figure(figsize=(7,5))
plt.scatter(df['NO2'], df['RSPM/PM10'], c='magenta', alpha=0.5)
plt.xlabel("NO2")
plt.ylabel("PM10")
plt.title("NO2 vs PM10")
plt.grid(alpha=0.3)
plt.savefig("outputs/5_scatter.png")
plt.clf()

# 6. Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=df['RSPM/PM10'], color='skyblue')
plt.title("PM10 Boxplot")
plt.savefig("outputs/6_boxplot.png")
plt.clf()

# 7. NO2 Distribution
plt.figure(figsize=(7,5))
sns.histplot(df['NO2'], kde=True, color='yellow')
plt.title("NO2 Distribution")
plt.grid(alpha=0.3)
plt.savefig("outputs/7_no2_dist.png")
plt.clf()

# 8. Year-wise PM10
plt.figure(figsize=(8,5))
df.groupby('year')['RSPM/PM10'].mean().plot(kind='bar', color='purple')
plt.title("Year-wise Avg PM10")
plt.grid(axis='y', alpha=0.3)
plt.savefig("outputs/8_yearly.png")
plt.clf()

# 9. SO2 vs PM10 Trend
plt.figure(figsize=(10,5))
df.sort_values('Sampling Date').plot(
    x='Sampling Date',
    y=['SO2','RSPM/PM10'],
    colormap='viridis'
)
plt.title("SO2 & PM10 Trend")
plt.savefig("outputs/9_so2_pm10.png")
plt.clf()

print(" EDA completed (9 visuals)")

# ===================== MODEL =====================
features = ['SO2','NO2','SPM','month']
X = df[features]
y = df['RSPM/PM10']

print("Dataset shape:", X.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(" R2 Score:", r2_score(y_test, pred))

# 10. Feature Importance
importance = model.feature_importances_
plt.figure(figsize=(7,5))
plt.bar(features, importance, color=['red','blue','green','orange'])
plt.title("Feature Importance")
plt.grid(axis='y', alpha=0.3)
plt.savefig("outputs/10_importance.png")
plt.clf()

print(" Model trained")
print(" PROJECT COMPLETED SUCCESSFULLY")