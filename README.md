# ML Algorithms Practice

Clean, corrected practical implementations of classic machine learning algorithms.

## Algorithms Covered
- **Linear Regression** – Boston Housing dataset
- **Gaussian Naive Bayes** – Iris dataset
- **K-Nearest Neighbors (KNN)** – Iris dataset (with feature scaling)

## Key Fixes Applied
- Removed data leakage in Linear Regression (target was accidentally included in features)
- Proper train-test split with stratification
- Feature scaling for KNN
- Clear evaluation metrics + confusion matrix visualization

## How to Run
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python ml_algorithms.py

Results (approx)

Linear Regression: Realistic MSE after fixing leakage
Naive Bayes: ~96.7% accuracy on Iris
KNN (k=5): ~93.3% accuracy on Iris
