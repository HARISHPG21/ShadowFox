
# 🏦 Loan Price Prediction using SVM  

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)  
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/stable/)  
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-lightgrey?logo=jupyter)](https://jupyter.org/)  
[![License](https://img.shields.io/badge/License-MIT-green)](#)  

---

## 📘 Overview  

This project predicts **loan approval status** using a **Support Vector Machine (SVM)** model trained on applicant data.  
It demonstrates a simple but effective **machine learning pipeline** that handles missing values, trains an SVM classifier, and evaluates model performance on training and testing sets.  

---

## 🎯 Objective  

Develop a predictive model to determine whether a loan application will be **approved** or **rejected**, based on applicant attributes such as income, loan amount, credit history, and more.  

---

## ⚙️ Key Features  

- ✅ **Algorithm:** Support Vector Classifier (`SVC`)  
- 🧩 **Data Cleaning:** Missing values handled via mean imputation using `SimpleImputer`  
- 🧮 **Evaluation Metric:** Accuracy Score  
- 🧠 **Implementation:** Built with `scikit-learn` and `pandas`  

---

## 📂 Files in Repository  

| File | Description |
|------|--------------|
| `loan_prediction.ipynb` | Jupyter notebook with full implementation |
| `dataset.csv` | Dataset used for model training (if available) |
| `README.md` | Project overview and documentation |

---

## 🧠 Machine Learning Workflow  

1. **Data Import & Preprocessing**  
   - Load dataset using `pandas`  
   - Handle missing values with `SimpleImputer(strategy='mean')`  

2. **Dataset Splitting**  
   - Split into **training** and **testing** sets using `train_test_split()`  

3. **Model Training**  
   - Initialize and train an **SVM classifier** (`SVC`)  

4. **Evaluation**  
   - Compute **accuracy** for both training and testing data  

---

## 📈 Model Performance  

| Dataset  | Accuracy                |
| -------- | ----------------------- |
| Training | ~69.38%                 |
| Testing  | *(Insert test result here)* |

---

## 🧰 Libraries Used  

- **NumPy**  
- **Pandas**  
- **Scikit-learn** (`sklearn`)  

---

## 🔮 Future Improvements  

- Try different **imputation strategies** (median, most_frequent)  
- Experiment with **other classifiers** such as Random Forest or XGBoost  
- Apply **feature scaling** (`StandardScaler`) for better SVM performance  
- Tune hyperparameters using **GridSearchCV**  

---

## 💡 Key Insight  

This project highlights the fundamentals of supervised learning with SVMs — showcasing how preprocessing choices like imputation and scaling directly affect model performance.  

---

### 👨‍💻 Author  

Developed with curiosity and care using **Python & scikit-learn** 🧠  
