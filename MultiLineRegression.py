import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n_samples = 100

data = pd.DataFrame({
    'Hours_Studied': np.random.uniform(1, 10, n_samples),
    'Attendance': np.random.uniform(60, 100, n_samples),
    'Sleep_Hours': np.random.uniform(5, 9, n_samples),
    'Assignments_Submitted': np.random.randint(0, 10, n_samples),
    'Class_Participation': np.random.randint(1, 6, n_samples),
    'Internet_Usage_Hrs': np.random.uniform(1, 6, n_samples),
    'Library_Visits': np.random.randint(0, 10, n_samples),
    'Group_Study_Hrs': np.random.uniform(0, 5, n_samples),
    'Lab_Work_Score': np.random.uniform(50, 100, n_samples),
    'Tutorials_Attended': np.random.randint(0, 20, n_samples),
    'Project_Score': np.random.uniform(50, 100, n_samples),
    'Seminar_Score': np.random.uniform(50, 100, n_samples),
    'Extra_Courses': np.random.randint(0, 5, n_samples),
    'Backlogs': np.random.randint(0, 3, n_samples),
    'Discipline_Score': np.random.uniform(50, 100, n_samples),
    'Doubt_Resolution': np.random.randint(0, 10, n_samples),
    'Weekly_Quiz_Score': np.random.uniform(0, 10, n_samples),
    'Mentor_Meetings': np.random.randint(0, 5, n_samples),
    'Internship_Hours': np.random.randint(0, 100, n_samples),
    'Coding_Practice_Hrs': np.random.uniform(0, 10, n_samples),
    'Python_Score': np.random.uniform(40, 100, n_samples),
    'OS_Score': np.random.uniform(45, 100, n_samples),
    'Automata_Score': np.random.uniform(50, 100, n_samples)
})

# Features and targets
X = data.iloc[:, 0:20]
y = data[['Python_Score', 'OS_Score', 'Automata_Score']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Coefficients shape:", model.coef_.shape)
print("Intercepts:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted on one combined graph
plt.figure(figsize=(7, 5))
plt.scatter(y_test['Python_Score'], y_pred[:, 0], color='blue', label='Python')
plt.scatter(y_test['OS_Score'], y_pred[:, 1], color='green', label='Operating Systems')
plt.scatter(y_test['Automata_Score'], y_pred[:, 2], color='orange', label='Automata')
min_score = min(y_test.min())
max_score = max(y_test.max())
plt.plot([min_score, max_score], [min_score, max_score], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Actual vs Predicted Scores')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
