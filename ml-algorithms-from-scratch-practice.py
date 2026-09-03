
---

### Clean Corrected Code (`ml_algorithms.py`)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# 1. LINEAR REGRESSION – Boston Housing (CORRECTED)
# ============================================================
print("="*60)
print("LINEAR REGRESSION – Boston Housing")
print("="*60)

boston = fetch_openml(name='boston', version=1, as_frame=True, parser='pandas')
df = boston.frame.copy()

# Correct feature / target split (no leakage)
X = df.drop(columns=['MEDV'])          # all features
y = df['MEDV']                         # target

X = X.astype(float)
y = y.astype(float)

# Cross-validation
lin_reg = LinearRegression()
mse_scores = cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=5)
print("Negative MSE per fold:", mse_scores)
print("Average Negative MSE :", np.mean(mse_scores))

# Train-test split + final model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
print(f"Test MSE : {mean_squared_error(y_test, y_pred):.4f}")
print(f"Test R²  : {r2_score(y_test, y_pred):.4f}")

# ============================================================
# 2. GAUSSIAN NAIVE BAYES – Iris
# ============================================================
print("\n" + "="*60)
print("GAUSSIAN NAIVE BAYES – Iris")
print("="*60)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Naive Bayes – Iris Confusion Matrix")
plt.show()

# ============================================================
# 3. K-NEAREST NEIGHBORS – Iris
# ============================================================
print("\n" + "="*60)
print("K-NEAREST NEIGHBORS (k=5) – Iris")
print("="*60)

# Scale features (important for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))