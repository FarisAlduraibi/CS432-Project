import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('student-mat.csv', sep=';')

data_encoded = pd.get_dummies(data, drop_first=True)
X = data_encoded.drop('G3', axis=1)
y = data_encoded['G3']


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=42)

# Train models
lr_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Results:")
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
    print("R² Score:", round(r2_score(y_test, y_pred) * 100, 2), "%")

evaluate_model(lr_model, X_test, y_test, "Linear Regression")
evaluate_model(tree_model, X_test, y_test, "Decision Tree")

# Visualization
y_pred_lr = lr_model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.7)
plt.plot([0, 20], [0, 20], 'r--')
plt.xlabel('Actual Final Grades (G3)')
plt.ylabel('Predicted Final Grades (G3)')
plt.title('Actual vs Predicted Final Grades (Linear Regression)')
plt.grid(True)
plt.show()
