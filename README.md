## 🎓 Predicting Student Academic Performance Using AI

This project was developed for the **CS432 - Artificial Intelligence** course and aims to predict students' final grades using supervised machine learning techniques. We used a real-world dataset and evaluated the performance of two regression models.

---

## 📊 Objective

Use machine learning to predict the final grade (G3) of students using all available features in the dataset, including academic records, demographics, and behavioral attributes.
Categorical features were one-hot encoded to make them compatible with regression models.

---

## 🧠 Models Used

- **Linear Regression**
- **Decision Tree Regressor**

---

## 📁 Dataset

- **Source**: [UCI Machine Learning Repository – Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- **File Used**: `student-mat.csv`
- **Records**: 649 students
- **Features Used**:  
  All available features in the dataset were included in training.  
  Categorical variables (e.g., school, sex, address, etc.) were encoded using one-hot encoding.  
  The target variable is `G3` (final grade).
---

## ⚙️ Tech Stack

- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib

---

## 📈 Evaluation Results

| Model             | MAE  | MSE  | R² Score |
|------------------|------|------|----------|
| Linear Regression | 1.65 | 5.66 | 72.41%   |
| Decision Tree     | 1.14 | 4.20 | 79.50%   |

> ✅ The **Decision Tree Regressor** showed better accuracy and generalization.

---

## 📷 Visualization

The project includes a scatter plot comparing actual vs. predicted final grades using Linear Regression.

---

## 📌 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/FarisAlduraibi/CS432-Project.git
   cd CS432-Project
   Make sure student-mat.csv is in the same folder as the script.
   Run the script: python studentsPredict.py
## 🚧 Challenges and Future Work
Challenges faced:

Integrating and preprocessing a real-world dataset introduced noise, inconsistent formatting, and semicolon-separated values, which required careful handling.

Achieving high model accuracy while preventing overfitting on a relatively small feature set was a key concern.

Balancing simplicity (Linear Regression) vs. performance (Decision Tree) required evaluating trade-offs between interpretability and accuracy.

Planned improvements:

Expand feature engineering to include demographic or behavioral data (e.g., parental education, travel time, internet access).

Apply ensemble models like Random Forest or Gradient Boosting for potentially better accuracy.

Explore deep learning models such as MLPs, or transformer-based models like BERT adapted for tabular data.

Evaluate model robustness with cross-validation and hyperparameter tuning.



## 👥 Team
Faris Ali Alduraibi – 421107654

Saleh Saed Alghool – 422117042



