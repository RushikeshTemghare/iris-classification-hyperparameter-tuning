# Iris Flower Classification using Machine Learning & Hyperparameter Tuning

Optimising machine learning model performance using **GridSearchCV** and **RandomizedSearchCV** on the famous Iris Flower dataset.

![Python](https://img.shields.io/badge/Python-blue)
![Pandas](https://img.shields.io/badge/Pandas-150458)
![NumPy](https://img.shields.io/badge/NumPy-013243)
![Matplotlib](https://img.shields.io/badge/Matplotlib-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-success)
![Classification](https://img.shields.io/badge/Classification-Multiclass-blue)
![GridSearchCV](https://img.shields.io/badge/GridSearchCV-Hyperparameter%20Tuning-success)
![RandomizedSearchCV](https://img.shields.io/badge/RandomizedSearchCV-Model%20Optimisation-purple)
![Cross Validation](https://img.shields.io/badge/Cross%20Validation-5--Fold-informational)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📖 Project Overview

This project demonstrates how **hyperparameter tuning** can improve the performance of supervised machine learning models using the well-known **Iris Flower dataset**.

The notebook follows a complete machine learning workflow, beginning with dataset exploration and preprocessing before training baseline classification models. It then applies two widely used hyperparameter optimisation techniques—**GridSearchCV** and **RandomizedSearchCV**—to identify the optimal model configuration.

The primary objective of this project is to understand how different hyperparameter combinations influence model performance and to compare exhaustive and randomized search strategies for model optimisation.

---

# 🎯 Problem Statement

Machine learning models often rely on **hyperparameters** that significantly influence predictive performance. Selecting appropriate hyperparameter values manually can be time-consuming and may not produce the best-performing model.

The objective of this project is to classify **Iris flower species** using supervised machine learning while demonstrating how systematic hyperparameter tuning techniques can improve model performance. The project compares baseline classification models with optimised models obtained through **GridSearchCV** and **RandomizedSearchCV**, highlighting the importance of model optimisation in practical machine learning workflows.

---

# 📊 Dataset

This project uses the **built-in Iris dataset provided by the Seaborn library**, which is loaded directly within the notebook using:

```python
df = sns.load_dataset("iris")
```

Since the dataset is included within Seaborn, **no external CSV file is required**.

The dataset contains measurements of **150 Iris flowers** belonging to **three different species**:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Each observation consists of four numerical features used for classification.

| Feature | Description |
|----------|-------------|
| sepal_length | Length of the sepal (cm) |
| sepal_width | Width of the sepal (cm) |
| petal_length | Length of the petal (cm) |
| petal_width | Width of the petal (cm) |
| species | Iris flower species (Target Variable) |

### Dataset Summary

- **Total observations:** 150
- **Input features:** 4
- **Target classes:** 3
- **Machine Learning Problem:** Multiclass Classification
- **Learning Type:** Supervised Learning

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

This project follows a structured machine learning workflow to build, evaluate, and optimise classification models.

1. Import Required Libraries
2. Load the Iris Dataset
3. Explore and Understand the Dataset
4. Data Preparation
5. Train-Test Split
6. Train Baseline Machine Learning Models
7. Evaluate Model Performance
8. Hyperparameter Tuning using GridSearchCV
9. Hyperparameter Tuning using RandomizedSearchCV
10. Compare Optimised Models
11. Identify the Best Hyperparameters

---

# 🤖 Machine Learning Models Implemented

The following supervised machine learning classification algorithms were implemented:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Both models were initially trained using their default hyperparameters to establish baseline performance before optimisation.

---

# 🔍 Hyperparameter Tuning

To improve model performance, two hyperparameter optimisation techniques available in Scikit-learn were applied.

## GridSearchCV

GridSearchCV performs an exhaustive search by evaluating every possible combination of predefined hyperparameter values using cross-validation.

The workflow included:

- Defining a parameter grid
- Applying 5-fold Cross Validation
- Training multiple SVM models
- Evaluating every parameter combination
- Selecting the model with the highest cross-validation score

GridSearchCV guarantees finding the best combination within the specified search space but requires a higher computational cost.

---

## RandomizedSearchCV

RandomizedSearchCV performs a random search across the defined hyperparameter distributions instead of evaluating every possible combination.

The workflow included:

- Defining parameter distributions
- Randomly sampling parameter combinations
- Applying Cross Validation
- Comparing model performance
- Selecting the highest-performing configuration

RandomizedSearchCV significantly reduces computation time while often producing results comparable to GridSearchCV.

---

# 📈 Model Evaluation

The machine learning models were evaluated using **classification accuracy** on the test dataset.

To further optimise model performance, **5-fold Cross Validation** was applied using both **GridSearchCV** and **RandomizedSearchCV** on the Support Vector Machine (SVM) classifier.

The evaluation focused on:

- Classification Accuracy
- Cross-Validation Score
- Hyperparameter Optimisation
- Model Comparison

---

# 📊 Results

Two baseline classification models were trained and evaluated on the Iris Flower dataset.

| Model | Test Accuracy |
|--------|--------------:|
| K-Nearest Neighbors (KNN) | **100%** |
| Support Vector Machine (SVM) | **100%** |

Both models achieved perfect classification accuracy on the test dataset.

To investigate whether further optimisation could improve generalisation, hyperparameter tuning was performed using **GridSearchCV** and **RandomizedSearchCV**.

### GridSearchCV Results (5-Fold Cross Validation)

| C | Kernel | Mean Cross-Validation Accuracy |
|---|--------|------------------------------:|
| 1 | RBF | **98.00%** |
| 1 | Linear | **98.00%** |
| 10 | RBF | **98.00%** |
| 10 | Linear | 97.33% |
| 20 | RBF | 96.67% |
| 20 | Linear | 96.67% |
| 30 | RBF | 96.00% |
| 30 | Linear | 96.00% |

GridSearchCV evaluated **8 different hyperparameter combinations** using **5-fold Cross Validation**, allowing a systematic comparison of SVM configurations.

### RandomizedSearchCV Results

RandomizedSearchCV evaluated **4 randomly selected parameter combinations** from the predefined search space.

The best sampled configuration achieved a **98.00% mean cross-validation accuracy**, demonstrating that RandomizedSearchCV can produce performance comparable to GridSearchCV while evaluating fewer combinations and requiring less computational time.

### Key Findings

- ✅ K-Nearest Neighbors achieved **100% test accuracy**.
- ✅ Support Vector Machine achieved **100% test accuracy**.
- ✅ GridSearchCV evaluated every parameter combination within the search space.
- ✅ RandomizedSearchCV efficiently searched the parameter space using random sampling.
- ✅ Both optimisation techniques demonstrated the importance of systematic hyperparameter tuning for improving model selection.

---

# 💡 Skills Demonstrated

Throughout this project, the following machine learning skills were applied:

- Supervised Machine Learning
- Multiclass Classification
- Data Exploration
- Train-Test Split
- Model Training
- Model Evaluation
- Cross Validation
- Hyperparameter Tuning
- GridSearchCV
- RandomizedSearchCV
- Model Comparison
- Python Programming
- Scikit-learn

---

# 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Building supervised machine learning classification models.
- Understanding the importance of hyperparameter optimisation.
- Implementing GridSearchCV for exhaustive parameter tuning.
- Implementing RandomizedSearchCV for efficient model optimisation.
- Applying cross-validation to improve model reliability.
- Comparing different optimisation techniques based on performance and computational efficiency.
- Using Scikit-learn's model selection tools to build more robust machine learning models.

---

# 🚀 Future Improvements

Potential enhancements include:

- Compare additional classification algorithms such as Logistic Regression, Decision Tree, and Random Forest.
- Perform feature scaling comparisons using different preprocessing techniques.
- Evaluate models using Precision, Recall, F1 Score, and Confusion Matrix.
- Apply Bayesian Optimisation for advanced hyperparameter tuning.
- Build an interactive Streamlit application for real-time Iris species prediction.
- Deploy the application using Streamlit Community Cloud.

---

# 📂 Repository Structure

```text
iris-classification-hyperparameter-tuning/
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
