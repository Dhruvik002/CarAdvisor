# CarAdvisor — Car Price Prediction & Recommendation System

**Author :** Dhruvik Parmar  
**Course :** Integrated BSc + MSc — IT Specialization (Data Management & Visualization Insights)

---

## Project Overview

CarAdvisor is an intelligent car recommendation system designed to assist
car buyers in making informed decisions. It uses machine learning to predict
car prices and recommends the best value cars based on user preferences
including budget, seats, fuel type, and brand.

---

## Project Structure

| File | Description |
|------|-------------|
| `carproject_final.ipynb` | ML pipeline — data cleaning, EDA, model training, recommendation system |
| `app.py` | Streamlit web application (CarAdvisor AI) |
| `Cars Datasets 2025.csv` | Dataset — 1218 cars with price, specs, and fuel type |
| `CarAdvisor.pbix` | Power BI dashboard for visual analytics |
| `carimage/` | Brand logo images used in the web app |
| `requirements.txt` | Python dependencies |

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python |
| Development Tool | Jupyter Notebook |
| Data Analytics | pandas, numpy |
| Visualization | matplotlib, seaborn, Power BI |
| Machine Learning | scikit-learn |
| Web Application | Streamlit |

---

## Dataset

- **Source :** Cars Datasets 2025
- **Total Records :** 1218 cars
- **Features :** Company, Car Name, Engine, CC/Battery Capacity, HorsePower, Top Speed, Performance, Price, Fuel Type, Seats, Torque

---

## Machine Learning Pipeline

1. Data Cleaning — strips symbols, averages ranges, removes missing values
2. Feature Engineering — Log-Price transformation to reduce skewness
3. One-Hot Encoding — company names encoded for ML input
4. Models Trained — Decision Tree, Random Forest, KNN, SVM
5. Evaluation — 5-Fold Cross-Validation using Pipeline (no data leakage)
6. Best Model — SVM with RBF Kernel (highest CV R² score)

---

## Recommendation System

Content-based filtering approach:

- Filters cars by **budget**, **seat count**, **fuel type**, and **brand**
- Ranks results by **Value Score** (HorsePower per dollar) — best value first

---

## How to Run the Web App

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Run the app**
```bash
streamlit run app.py
```

**Step 3 — Open in browser**

http://localhost:8501

---

## How to Run the Notebook

1. Open `carproject_final.ipynb` in Jupyter Notebook or JupyterLab
2. Make sure `Cars Datasets 2025.csv` is in the same folder
3. Run all cells from top to bottom

---

## Requirements

Create a file named `requirements.txt` with the following:

pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit

---

## Project Summary

CarAdvisor successfully demonstrates an end-to-end machine learning pipeline — from raw data cleaning and exploratory data analysis to model training, evaluation, and a fully functional web-based recommendation system. The project contributes to improved customer experiences, increased confidence, and reduced decision-making bias when purchasing a car.

---

*Made by Dhruvik Parmar*
