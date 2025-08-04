# pip install pandas numpy matplotlib seaborn plotly scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os


csv_file = 'ev_sales_india.csv'

if not os.path.exists(csv_file):
    raise FileNotFoundError(
        f"{csv_file} not found in current directory. Please place the file in the same folder.")

df = pd.read_csv(csv_file)
print(df.head())


df['Date'] = pd.to_datetime(df['Date'])
print("\nMissing Values:\n", df.isnull().sum())

df['EV_Sales_Quantity'].fillna(df['EV_Sales_Quantity'].median(), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
print("\nDataset Info:")
print(df.info())


# Interactive line chart (Plotly)
fig1 = px.line(df, x='Year', y='EV_Sales_Quantity', color='State',
               title='EV Sales Trend by State')
fig1.show()

# Pie chart by vehicle category
fig2 = px.pie(df, names='Vehicle_Category', values='EV_Sales_Quantity',
              title='EV Sales Distribution by Vehicle Category', hole=0.3)
fig2.show()

# Heatmap: State vs Year
pivot_table = df.pivot_table(
    values='EV_Sales_Quantity', index='State', columns='Year', aggfunc='sum')
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt=".0f")
plt.title("EV Sales Heatmap by State & Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=[
                            'State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type'], drop_first=True)

# Drop unused columns if present
for col in ['Date', 'Month_Name']:
    if col in df_encoded.columns:
        df_encoded.drop(col, axis=1, inplace=True)


X = df_encoded.drop('EV_Sales_Quantity', axis=1)
y = df_encoded['EV_Sales_Quantity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n✅ Root Mean Squared Error: {rmse:.2f}")


importance = model.feature_importances_
feature_importance = pd.Series(
    importance, index=X_train.columns).sort_values(ascending=False)

fig4 = px.bar(
    feature_importance.head(15),
    x=feature_importance.head(15).values,
    y=feature_importance.head(15).index,
    orientation='h',
    title='Top 15 Important Features',
    color=feature_importance.head(15).values,
    color_continuous_scale='Viridis'
)
fig4.show()
