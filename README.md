# Iris Flower Classification using Machine Learning & Hyperparameter Tuning

Optimising supervised machine learning models using **GridSearchCV** and **RandomizedSearchCV** on the famous Iris Flower dataset.

![Python](https://img.shields.io/badge/Python-blue)
![Pandas](https://img.shields.io/badge/Pandas-150458)
![NumPy](https://img.shields.io/badge/NumPy-013243)
![Matplotlib](https://img.shields.io/badge/Matplotlib-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-success)
![Classification](https://img.shields.io/badge/Classification-Multiclass-blueviolet)
![GridSearchCV](https://img.shields.io/badge/GridSearchCV-Hyperparameter%20Tuning-blue)
![RandomizedSearchCV](https://img.shields.io/badge/RandomizedSearchCV-Optimisation-green)
![Cross Validation](https://img.shields.io/badge/Cross%20Validation-5--Fold-important)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📖 Project Overview

This project demonstrates how supervised machine learning models can be trained, evaluated, and optimised using hyperparameter tuning techniques on the well-known **Iris Flower dataset**.

The notebook follows a complete end-to-end machine learning workflow, beginning with exploratory data analysis (EDA) and data preparation before training baseline classification models. It then applies **GridSearchCV** and **RandomizedSearchCV** to optimise a Support Vector Machine (SVM) classifier and compare different hyperparameter combinations using cross-validation.

The primary objective of this project is to understand how hyperparameter tuning influences model performance while demonstrating the practical application of model optimisation techniques available in **Scikit-learn**.

---

# 🎯 Problem Statement

Machine learning algorithms rely on hyperparameters that directly influence their predictive performance. Selecting these values manually can be inefficient and may not produce the best-performing model.

The objective of this project is to classify Iris flower species using supervised machine learning while demonstrating how systematic hyperparameter optimisation techniques can improve model performance.

The project compares baseline machine learning models with optimised models obtained through **GridSearchCV** and **RandomizedSearchCV**, highlighting the importance of model selection and parameter tuning in real-world machine learning workflows.

---

# 📊 Dataset

This project uses the **built-in Iris dataset** provided by the **Seaborn** library, which is loaded directly within the notebook using:

```python
df = sns.load_dataset("iris")
```

Since the dataset is included within Seaborn, **no external CSV file is required**.

The dataset contains measurements from **150 Iris flowers** belonging to three different species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Each observation contains four numerical features describing the physical dimensions of the flower.

| Feature | Description |
|----------|-------------|
| sepal_length | Length of the sepal (cm) |
| sepal_width | Width of the sepal (cm) |
| petal_length | Length of the petal (cm) |
| petal_width | Width of the petal (cm) |
| species | Iris Flower Species (Target Variable) |

### Dataset Summary

- Total observations: **150**
- Input features: **4**
- Target classes: **3**
- Machine Learning Problem: **Multiclass Classification**
- Learning Type: **Supervised Learning**

---

# 📈 Exploratory Data Analysis (EDA)

Before training the machine learning models, exploratory data analysis was performed to understand the characteristics of the dataset and identify any potential data quality issues.

The analysis included:

- Inspecting dataset dimensions
- Viewing data types
- Checking for missing values
- Displaying descriptive statistics
- Exploring feature distributions
- Analysing relationships between numerical variables
- Comparing feature distributions across Iris species

Visualisations were created using **Matplotlib** and **Seaborn**, helping to better understand feature patterns and class separation before model training.

---

# 🧹 Data Preprocessing

Prior to model training, the dataset was prepared for supervised learning.

The preprocessing pipeline included:

- Separating input features and target labels
- Encoding the target variable
- Splitting the dataset into training and testing sets
- Preparing the data for classification algorithms

The Iris dataset contains no missing values, allowing the project to focus primarily on model development and hyperparameter optimisation while still following a structured machine learning workflow.

---

# 🛠️ Technologies Used

- 🐍 Python
- 📓 Jupyter Notebook
- 🐼 Pandas
- 🔢 NumPy
- 📊 Matplotlib
- 📈 Seaborn
- 🤖 Scikit-learn
- 🔍 GridSearchCV
- 🎲 RandomizedSearchCV
- ✅ Cross Validation
- 💻 Visual Studio Code
- 🌐 Git
- 📂 GitHub

---

# ⚙️ Machine Learning Workflow

This project follows a structured machine learning pipeline to build, evaluate, and optimise classification models.

1. Import Required Libraries
2. Load the Iris Dataset
3. Perform Exploratory Data Analysis (EDA)
4. Prepare the Dataset
5. Encode the Target Variable
6. Split the Dataset into Training and Testing Sets
7. Train Baseline Machine Learning Models
8. Evaluate Model Performance
9. Perform Hyperparameter Tuning using GridSearchCV
10. Perform Hyperparameter Tuning using RandomizedSearchCV
11. Compare Hyperparameter Combinations
12. Analyse Cross-Validation Results
13. Select the Best Model Configuration

---

# 🤖 Machine Learning Models Implemented

The following supervised machine learning classification algorithms were implemented and evaluated:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Both models were initially trained using their default hyperparameters to establish baseline performance before optimisation.

K-Nearest Neighbors (KNN) served as a distance-based classification algorithm, while Support Vector Machine (SVM) was selected for further optimisation through hyperparameter tuning due to its strong classification performance on the Iris dataset.

---

# 🔍 Hyperparameter Tuning

To optimise model performance, two hyperparameter optimisation techniques available in **Scikit-learn** were implemented.

## GridSearchCV

GridSearchCV performs an exhaustive search by evaluating **every possible combination** of predefined hyperparameter values using cross-validation.

The optimisation workflow included:

- Defining a parameter grid
- Applying **5-Fold Cross Validation**
- Training multiple SVM models
- Evaluating every parameter combination
- Comparing mean cross-validation scores
- Selecting the best-performing configuration

GridSearchCV guarantees finding the optimal parameter combination within the specified search space, although it requires greater computational time compared with random search methods.

---

## RandomizedSearchCV

RandomizedSearchCV performs a random search across the predefined parameter distributions instead of evaluating every possible combination.

The optimisation workflow included:

- Defining parameter distributions
- Randomly sampling parameter combinations
- Applying Cross Validation
- Comparing model performance
- Selecting the highest-performing sampled configuration

RandomizedSearchCV significantly reduces computational cost while often producing results comparable to GridSearchCV, making it particularly useful when working with larger search spaces.

---

# 📈 Model Evaluation

The classification models were evaluated using **Accuracy Score** on the test dataset.

To further assess model robustness and identify optimal hyperparameter values, **5-Fold Cross Validation** was performed using both **GridSearchCV** and **RandomizedSearchCV**.

The evaluation focused on:

- Classification Accuracy
- Cross-Validation Score
- Hyperparameter Optimisation
- Model Comparison

---

# 📊 Results

Two baseline machine learning classifiers were trained and evaluated on the Iris Flower dataset.

| Model | Test Accuracy |
|--------|--------------:|
| K-Nearest Neighbors (KNN) | **100%** |
| Support Vector Machine (SVM) | **100%** |

Both baseline models achieved **100% classification accuracy** on the test dataset.

To investigate whether further optimisation could improve model performance, hyperparameter tuning was performed using **GridSearchCV** and **RandomizedSearchCV** on the Support Vector Machine (SVM) classifier.

---

## GridSearchCV Results (5-Fold Cross Validation)

GridSearchCV evaluated **8 different hyperparameter combinations** using **5-Fold Cross Validation**.

| C | Kernel | Mean Cross-Validation Accuracy |
|---:|:------|-------------------------------:|
| 1 | RBF | **98.00%** |
| 1 | Linear | **98.00%** |
| 10 | RBF | **98.00%** |
| 10 | Linear | 97.33% |
| 20 | RBF | 96.67% |
| 20 | Linear | 96.67% |
| 30 | RBF | 96.00% |
| 30 | Linear | 96.00% |

The exhaustive search systematically evaluated every parameter combination defined in the search space, enabling a comprehensive comparison of SVM configurations.

---

## RandomizedSearchCV Results

RandomizedSearchCV evaluated **4 randomly selected parameter combinations** from the predefined search space.

The sampled configurations achieved mean cross-validation accuracies ranging from **96.00% to 98.00%**, with the best-performing sampled configuration achieving **98.00%**.

This demonstrates that RandomizedSearchCV can identify highly competitive hyperparameter combinations while evaluating fewer models than GridSearchCV, making it a computationally efficient optimisation technique.

---

## 🔑 Key Findings

- ✅ K-Nearest Neighbors achieved **100% test accuracy**.
- ✅ Support Vector Machine achieved **100% test accuracy**.
- ✅ GridSearchCV evaluated **8 different hyperparameter combinations** using **5-Fold Cross Validation**.
- ✅ RandomizedSearchCV evaluated **4 randomly sampled parameter combinations**.
- ✅ The highest mean cross-validation accuracy achieved during hyperparameter tuning was **98.00%**.
- ✅ Both optimisation techniques demonstrated the importance of systematic hyperparameter tuning for selecting high-performing machine learning models.

---

---

# 💡 Skills Demonstrated

Throughout this project, the following machine learning and data science skills were applied:

- Supervised Machine Learning
- Multiclass Classification
- Exploratory Data Analysis (EDA)
- Data Preparation
- Train-Test Split
- Model Training
- Model Evaluation
- Cross Validation
- Hyperparameter Tuning
- GridSearchCV
- RandomizedSearchCV
- Model Comparison
- Performance Optimisation
- Python Programming
- Scikit-learn

---

# 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Building end-to-end supervised machine learning classification models.
- Understanding the importance of hyperparameter optimisation in improving model performance.
- Implementing K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) classifiers using Scikit-learn.
- Applying GridSearchCV to perform exhaustive hyperparameter tuning.
- Using RandomizedSearchCV as a computationally efficient alternative for model optimisation.
- Performing 5-Fold Cross Validation to evaluate model reliability and generalisation.
- Comparing multiple hyperparameter combinations using cross-validation scores.
- Analysing model performance and selecting the optimal configuration.
- Working with Scikit-learn's model selection and evaluation tools.
- Following a structured machine learning workflow from data exploration through to model optimisation.

---

# 🚀 Future Improvements

Potential enhancements for this project include:

- Implement additional classification algorithms such as Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- Evaluate models using additional performance metrics including Precision, Recall, F1 Score, ROC Curve, and Confusion Matrix.
- Experiment with feature engineering and feature selection techniques.
- Compare the impact of different feature scaling and preprocessing methods.
- Apply Bayesian Optimisation for more advanced hyperparameter tuning.
- Build an interactive Streamlit web application for real-time Iris flower species prediction.
- Deploy the application using Streamlit Community Cloud.

---

# 📂 Repository Structure

```text
iris-flower-classification-hyperparameter-tuning/
│
├── iris_classification_hyperparameter_tuning.ipynb
├── iris_classification_hyperparameter_tuning.py
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 👤 Author

**Rushikesh Temghare**

MSc Data Science & Artificial Intelligence  
Bournemouth University

---

# 📄 License

This project is licensed under the **MIT License**.

---
