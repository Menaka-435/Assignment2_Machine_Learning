# CS5710 - Machine Learning  
## Homework 2: Programming Questions (Q7–Q9)  

**Student:** Menaka Naga Sai Pothina  
**University:** University of Central Missouri  
**Course:** CS5710 - Machine Learning, Fall 2025  

---

## Overview

This assignment demonstrates **three machine learning tasks** using the **Iris dataset**:

1. **Decision Tree Classification (Q7)**  
2. **k-Nearest Neighbors (kNN) Classification & Decision Boundaries (Q8)**  
3. **kNN Performance Evaluation (Metrics & ROC/AUC) (Q9)**  

The purpose is to train models, visualize decision boundaries, and evaluate classification performance using standard metrics.

---

## Dataset

- **Dataset:** Iris dataset (built-in scikit-learn)  
- **Number of samples:** 150  
- **Features:** 4 (Decision Tree), 2 (kNN visualization: sepal length & sepal width)  
- **Target classes:** 3 (Setosa, Versicolor, Virginica)  

```python
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X_full = iris.data          # All 4 features for Decision Tree
y_full = iris.target

# First 2 features for kNN visualization
X_2f = iris.data[:, :2]
y_2f = iris.target
