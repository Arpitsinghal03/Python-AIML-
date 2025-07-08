import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Sample dataset: Study Hours vs Exam Score (hypothetical college student data)

data = {
    'Hours_Studied': [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
    'Exam_Score': [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
}


df = pd.DataFrame(data)

# Splitting dataset into training and testing data
X = df[['Hours_Studied']].values
y = df['Exam_Score'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Physical implementation of linear regression (no sklearn regression)
# Using formula: y = mx + b
# m = (mean(xy) - mean(x)mean(y)) / (mean(x^2) - mean(x)^2)
# b = mean(y) - m*mean(x)

mean_x = np.mean(X_train)
mean_y = np.mean(y_train)

m_num = np.mean(X_train * y_train.reshape(-1, 1)) - mean_x * mean_y
m_den = np.mean(X_train ** 2) - mean_x ** 2
m = m_num / m_den
b = mean_y - m * mean_x

# Predict function
def predict(x):
    return m * x + b

# Predict on training data
y_pred_train = predict(X_train)

# Plotting training data and regression line
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, y_pred_train, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression - Training Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output values
mean_train_score = np.mean(y_train)
m, b, mean_train_score
