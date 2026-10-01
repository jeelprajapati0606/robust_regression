# 🏠 Advanced House Price Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-blueviolet?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
</p>

<h3 align="center">
  🏡 Predicting House Prices using Advanced Regression Techniques
</h3>

<p align="center">
  A complete Machine Learning project covering regularization, cross-validation,
  tree-based regression, ensemble learning and Support Vector Regression.
</p>

<p align="center">

<a href="https://github.com/jeelprajapati0606/robust_regression/blob/main/robust_regression_engine/robust_regression_engine_svr-checkpoint.ipynb">
  <img src="https://img.shields.io/badge/📓%20View%20Notebook-181717?style=for-the-badge&logo=github">
</a>

<a href="https://www.dropbox.com/scl/fi/qukc6sswfigj93ti7ddi7/robust_regression.mp4?rlkey=088n8yo7wlpvu6lwiipjjl10i&st=w3c541lx&dl=0">
  <img src="https://img.shields.io/badge/▶️%20Watch%20Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white">
</a>

<a href="https://github.com/jeelprajapati0606/robust_regression/blob/main/robust_regression_engine/Advanced_Regression_HousePrice_Dataset_3800%20-%20Advanced_Regression_HousePrice_Dataset_3800.csv.csv">
  <img src="https://img.shields.io/badge/📊%20Dataset-2E7D32?style=for-the-badge">
</a>

</p>

---

## 📌 About The Project

This project focuses on predicting **house prices in INR** using different
Machine Learning regression techniques.

The project compares regularized linear models, tree-based models,
ensemble methods and Support Vector Regression.

It also demonstrates how **cross-validation and hyperparameter tuning**
can be used to improve model reliability and performance.

---

## 🎯 Project Objectives

- Predict house prices using Machine Learning
- Understand regression techniques
- Apply **Ridge and Lasso Regularization**
- Compare different regression models
- Apply different Cross-Validation strategies
- Perform Hyperparameter Tuning
- Analyze feature coefficients and feature importance
- Evaluate models using RMSE, MAE and R²

---

## 📊 Dataset

The dataset contains **3,800 house records and 12 original columns**.

### Dataset Features

| Feature | Description |
|---|---|
| `property_id` | Unique property identifier |
| `sale_date` | Property sale date |
| `area_sqft` | Area of the property |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `location_score` | Location quality score |
| `property_age` | Age of the property |
| `distance_city_km` | Distance from city |
| `near_school` | Whether school is nearby |
| `near_metro` | Whether metro is nearby |
| `crime_rate_index` | Crime rate index |
| `house_price_inr` | Target house price |

### 🎯 Target Variable

```text
house_price_inr

````markdown
---

# 📊 Detailed Results

## 🔍 Model Comparison

The project compares multiple regression algorithms using **RMSE, MAE, and R² Score**.

| Model | RMSE | MAE | R² Score |
|---|---:|---:|---:|
| Decision Tree | 2,812,489 | 2,047,890 | 0.901781 |
| Random Forest | 2,386,467 | 1,746,069 | 0.929283 |

### 📌 Evaluation Metrics

- **RMSE (Root Mean Squared Error)** → Measures the average prediction error.
- **MAE (Mean Absolute Error)** → Measures the average absolute difference between actual and predicted prices.
- **R² Score** → Shows how well the model explains the variation in house prices.

> Lower RMSE and MAE indicate smaller prediction errors, while a higher R² indicates better explanatory performance.

---

## 🌲 Random Forest Results

The Random Forest model was implemented using multiple decision trees and evaluated on the test dataset.

```text
Random Forest
-------------
MSE  : 5695225134926.978
MAE  : 1746069.2924934211
RMSE : 2386467.0823053434
R²   : 0.9292827570458903
````

### 📈 Random Forest Performance

* **RMSE:** ~2.39 Million
* **MAE:** ~1.75 Million
* **R² Score:** ~0.9293

---

# ⚙️ Hyperparameter Tuning

To improve model performance, **GridSearchCV** was used for hyperparameter tuning.

The project applies hyperparameter tuning to:

* Ridge Regression
* Lasso Regression
* Decision Tree Regression
* SVR

### 🔧 Ridge Parameters

```python
{
    "model__alpha": [0.01, 0.1, 1, 10, 100, 1000]
}
```

### 🔧 Lasso Parameters

```python
{
    "model__alpha": [10, 100, 500, 1000, 5000, 10000]
}
```

### 🔧 Decision Tree Parameters

```python
{
    "max_depth": [3, 5, 8, 10, 15, None],
    "min_samples_split": [2, 5, 10, 20]
}
```

### 🔧 SVR Parameters

```python
{
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", 0.01],
    "model__epsilon": [0.1, 0.2]
}
```

GridSearchCV performs multiple combinations of parameters and selects the configuration based on cross-validation performance.

---

# 🔄 Cross-Validation

Different cross-validation techniques were explored in the project.

### Methods Used

```text
1. K-Fold Cross Validation
2. Stratified K-Fold
3. Leave-One-Out Cross Validation
4. Time Series Split
```

Cross-validation helps evaluate whether a model can generalize well to unseen data.

---

# 📌 Feature Importance

The project also analyzes feature importance using the **Random Forest** model.

```text
Feature Importance
        ↓
Random Forest
        ↓
Identify important input features
        ↓
Understand factors affecting house prices
```

The notebook includes a feature-importance visualization using:

```python
plt.title("Random Forest Feature Importance")
```

This helps understand which features contribute more to the prediction process.

---

# 📊 Model Evaluation Visualization

The notebook generates visual comparisons for model performance.

## RMSE Comparison

```python
plt.title("Model Comparison - RMSE")
plt.ylabel("RMSE")
```

## R² Score Comparison

```python
plt.title("Model Comparison - R² Score")
plt.ylabel("R² Score")
```

These visualizations make it easier to compare the regression models.

---

# 🧪 SVR Model

Support Vector Regression was also implemented.

The project uses a pipeline containing:

```text
StandardScaler
      ↓
SVR
      ↓
Prediction
```

### Linear SVR Configuration

```python
SVR(
    kernel="linear",
    C=1.0,
    epsilon=0.1
)
```

### SVR Result

```text
RMSE : 8982717.606611094
MAE  : 6990273.615578244
R²   : -0.0019127827584033419
```

The notebook also performs hyperparameter tuning for SVR using `GridSearchCV`.

---

# 🎯 Prediction Workflow

The complete prediction workflow can be summarized as:

```text
Raw House Price Dataset
          ↓
Data Cleaning
          ↓
Feature Engineering
          ↓
Date Feature Extraction
          ↓
Train-Test Split
          ↓
Feature Scaling
          ↓
Model Training
          ↓
Hyperparameter Tuning
          ↓
Cross Validation
          ↓
Model Evaluation
          ↓
House Price Prediction
```

---

# 🧠 What I Learned

Through this project, I practiced:

* Regression fundamentals
* Data preprocessing
* Feature engineering
* Train-test splitting
* Feature scaling
* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regression
* Random Forest Regression
* Support Vector Regression
* Cross-validation
* Hyperparameter tuning
* GridSearchCV
* Model evaluation
* Feature importance
* Data visualization
* Model comparison

---

# 💻 Technologies Used

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white">

<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white">

<img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white">

<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">

<img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge">

<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">

</p>

---

# 📂 Project Structure

```text
Advanced-House-Price-Prediction/
│
├── 📓 robust_regression_engine_svr-checkpoint.ipynb
│
├── 📊 dataset/
│   └── house_price_data.csv
│
├── 🖼️ screenshots/
│   ├── model-comparison.png
│   ├── feature-importance.png
│   └── predictions.png
│
├── 🎥 demo/
│   └── project-demo.mp4
│
└── 📄 README.md
```

> Update the file and folder names according to your actual GitHub repository structure.

---


# ⭐ Project Highlights

<div align="center">

| 🔹 | Highlight                        |
| -- | -------------------------------- |
| 🏠 | House Price Prediction           |
| 🤖 | Multiple Regression Models       |
| ⚙️ | Hyperparameter Tuning            |
| 🔄 | Cross Validation                 |
| 📊 | Model Comparison                 |
| 🌲 | Random Forest Feature Importance |
| 📈 | Regression Evaluation            |
| 🧪 | SVR Implementation               |
| 📓 | Jupyter Notebook Based Project   |

</div>

---

# 🔮 Future Improvements

Some possible future improvements are:

* Add more real-world property data
* Perform advanced feature engineering
* Try additional regression algorithms
* Improve model optimization
* Deploy the trained model as a web application
* Create an interactive prediction dashboard
* Add user input for real-time house price prediction
* Use advanced ensemble techniques
* Build an API for model predictions

---

# 🏆 Conclusion

This project demonstrates an end-to-end **House Price Prediction workflow using Machine Learning regression techniques**.

It covers the complete process from:

```text
Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Hyperparameter Tuning
 ↓
Cross Validation
 ↓
Evaluation
 ↓
Prediction
```

The project helped me understand how different regression algorithms behave on the same dataset and how model evaluation techniques can be used to analyze their performance.

---



---



This part is based on the actual sections/results present in your uploaded notebook, including the Random Forest result and the GridSearchCV/SVR sections.
