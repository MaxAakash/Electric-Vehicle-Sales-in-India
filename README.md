# 🚗 Electric Vehicle Sales Analysis – Python Project (VS Code)

This project analyzes and predicts Electric Vehicle (EV) sales across Indian states using Python and Machine Learning. It is adapted for execution in a **local environment using VS Code** (not Colab).

---

## 📁 Dataset

Make sure the dataset `ev_sales_india.csv` is placed in the same directory as the script. The dataset includes:
- State-wise EV sales
- Vehicle type and category
- Year, date, and month of sales
- EV sales quantity
- here to download data set https://drive.google.com/file/d/1oq6H3U0MyAXlGBjCUlOO9hR7ryDKhTMs/view?usp=sharing
---

## 📌 Objectives

- Explore EV sales trends across Indian states and vehicle categories
- Preprocess data and perform feature engineering
- Build and evaluate a Random Forest regression model
- Visualize insights using Plotly and Seaborn

---

## 🧰 Tools & Technologies

- Python (Pandas, NumPy)
- Visualization: Matplotlib, Seaborn, Plotly
- Machine Learning: Scikit-learn (RandomForestRegressor)
- Development Environment: Visual Studio Code

---

## 📊 Visualizations

- 📈 Interactive line chart: EV sales trend by state
- 🧩 Pie chart: EV distribution by vehicle category
- 🔥 Heatmap: Sales heat across states and years
- 🧠 Bar plot: Feature importance in prediction model

---

## 🧠 Model Summary

- **Algorithm:** Random Forest Regressor
- **Evaluation Metric:** Root Mean Squared Error (RMSE)
- **Key Features:** Year, State, Vehicle Category, Vehicle Type

---

## 📂 How to Run

1. Clone the repo or copy the script.
2. Place `ev_sales_india.csv` in the same folder.
3. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```
4. Run the script:
```bash
python ev_sales_analysis.py

Colab NoteBook  https://colab.research.google.com/drive/1E9ltmhEPwiFLeMp1mWh1YNfAKCdkcbAh?usp=sharing
```

---

## 📈 Future Improvements

- Add time-series forecasting (Prophet, ARIMA)
- Build a Streamlit or Flask dashboard
- Integrate with Power BI or Tableau for visualization

---

## 📄 License

This project is open-source and free for academic and personal use.

