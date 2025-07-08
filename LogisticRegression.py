import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

# Dataset: Exam Score vs Admission (Binary Classification)

logistic_data = {
    'Exam_Score': [55, 60, 65, 70, 75, 80, 85, 90, 95, 99],
    'Admitted':   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}
log_df = pd.DataFrame(logistic_data)

X_log = log_df[['Exam_Score']]
y_log = log_df['Admitted']

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.3, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train_log, y_train_log)
y_pred_log = log_model.predict(X_test_log)


# Evaluation

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test_log, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test_log, y_pred_log))
print("Coefficients:", log_model.coef_)
print("Intercept:", log_model.intercept_)

# Plot

x_vals = np.linspace(50, 100, 200).reshape(-1, 1)
x_vals_df = pd.DataFrame(x_vals, columns=['Exam_Score'])
y_probs = log_model.predict_proba(x_vals_df)[:, 1]

plt.figure(figsize=(6, 4))
plt.scatter(X_log, y_log, color='blue', label='Actual')
plt.plot(x_vals, y_probs, color='red', label='Logistic Curve')
plt.xlabel("Exam Score")
plt.ylabel("Admitted Probability")
plt.title("Logistic Regression - Admission Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
