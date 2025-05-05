# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the dataset
data = pd.read_csv('students.csv')  # Make sure the file is in the same directory

# Step 3: Show the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Step 4: Check for missing values
print("\\nMissing values per column:")
print(data.isnull().sum())

# Step 5: Select important features and target
selected_features = ['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
data_small = data[selected_features]

X = data_small.drop('G3', axis=1)  # Features
y = data_small['G3']               # Target (final grade)

# Step 6: Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Initialize AI models
lr_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=42)

# Step 8: Train the models
lr_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# Step 9: Define evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\\n{model_name} Results:")
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
    print("R² Score:", round(r2_score(y_test, y_pred) * 100, 2), "%")

# Step 10: Evaluate both models
evaluate_model(lr_model, X_test, y_test, "Linear Regression")
evaluate_model(tree_model, X_test, y_test, "Decision Tree")

# Step 11: Visualization of predictions for Linear Regression
y_pred_lr = lr_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.7)
plt.plot([0, 20], [0, 20], 'r--')  # perfect prediction line
plt.xlabel('Actual Final Grades (G3)')
plt.ylabel('Predicted Final Grades (G3)')
plt.title('AI Model: Actual vs Predicted Final Grades (Linear Regression)')
plt.grid(True)
plt.show()


file_path = "/mnt/data/studentsPredict_AI.py"
with open(file_path, "w") as f:
    f.write(code)

file_path