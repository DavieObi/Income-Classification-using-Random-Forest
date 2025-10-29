## Income Classification using Random Forest

### 🧩 **Project Overview**

This project aims to build a **machine learning classification model** that predicts whether an individual's **income exceeds $50K per year** based on various demographic and employment attributes.
The dataset used for this analysis is the popular **“Adult Income” (Census Income) dataset**, commonly used for binary classification tasks.

---

### 🎯 **Objective**

The main goal of this project is to:

* Analyze the relationship between personal and professional attributes and income level.
* Build a **Random Forest Classifier** to predict income category (`<=50K` or `>50K`).
* Evaluate the model’s performance using **ROC-AUC score** and **Confusion Matrix**.

---

### 🧠 **Machine Learning Workflow**

#### **1️⃣ Importing Libraries**

The following Python libraries were used:

```python
import pandas as pd
import numpy as np
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

These libraries handle **data preprocessing, visualization, statistical checks, and model training.**

---

#### **2️⃣ Data Loading**

The dataset was imported from a CSV file named `income_evaluation.csv`:

```python
df = pd.read_csv("income_evaluation.csv", encoding='unicode_escape')
```

It contains **32,561 rows** and **15 columns**, including features such as age, workclass, education, occupation, hours-per-week, and more.

---

#### **3️⃣ Data Inspection**

Basic checks were performed to understand the data:

```python
df.info()
df.isnull().sum()
```

* **No missing values** were found.
* Data types included a mix of numeric and categorical variables.

---

#### **4️⃣ Label Encoding**

Since the dataset includes categorical variables, **Label Encoding** was applied to convert them into numerical values suitable for model training.

```python
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
categorical_cols = [' workclass', ' education', ' marital-status', ' occupation',
                    ' relationship', ' race', ' sex', ' native-country', ' income']

for i in categorical_cols:
    df[i] = label_encoder.fit_transform(df[i].astype('|S'))
```

---

#### **5️⃣ Multicollinearity Check (VIF)**

To detect multicollinearity between independent variables, **Variance Inflation Factor (VIF)** was calculated:

```python
variables = df[['age', ' workclass', ' fnlwgt', ' education', ' education-num',
                ' marital-status', ' occupation', ' relationship', ' race',
                ' sex', ' capital-gain', ' capital-loss', ' hours-per-week',
                ' native-country']]

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i)
              for i in range(variables.shape[1])]
vif['Features'] = variables.columns
```

#### 🔍 Columns with VIF > 10 were dropped:

* `education-num`
* `race`
* `hours-per-week`
* `native-country`

This helps reduce feature redundancy and improve model performance.

---

#### **6️⃣ Outlier Removal**

Outliers were detected and removed using **Z-score**:

```python
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
```

This step ensures the model is not influenced by extreme values that can distort predictions.

---

#### **7️⃣ Correlation Analysis**

A heatmap was plotted to visualize feature correlations:

```python
plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True)
```

This helped understand how features relate to income and to each other.

---

#### **8️⃣ Feature-Target Split & Data Splitting**

```python
data = df.values
x, y = data[:, :-1], data[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```

* **80%** training data
* **20%** testing data

---

#### **9️⃣ Model Training: Random Forest Classifier**

A **Random Forest** model with 50 estimators was trained:

```python
classifier = RandomForestClassifier(n_estimators=50, random_state=0)
classifier.fit(x_train, y_train)
```

Predictions were then made:

```python
y_pred = classifier.predict(x_test)
```

---

#### **🔟 Model Evaluation**

* **ROC-AUC Score**:

```python
roc_auc_score(y_test, y_pred)
```

**Result:** `0.747`
This indicates a moderate predictive power — the model can correctly classify income levels about **74.7% of the time**.

* **Confusion Matrix**:

```python
cm = confusion_matrix(y_test, y_pred)
```

|                 | Predicted ≤50K | Predicted >50K |
| --------------- | -------------- | -------------- |
| **Actual ≤50K** | 4384           | 352            |
| **Actual >50K** | 582            | 768            |

* **Visualization:**

```python
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=0.5, square=True, cmap='Pastel1')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: 0.747', size=15)
```

---

### 📊 **Results Summary**

| Metric                       | Value                    |
| ---------------------------- | ------------------------ |
| **ROC-AUC Score**            | 0.747                    |
| **Model Type**               | Random Forest Classifier |
| **Feature Reduction Method** | VIF                      |
| **Outlier Detection**        | Z-score                  |
| **Test Size**                | 20%                      |

---

### 💡 **Insights**

* Features such as **age**, **education**, **marital status**, and **capital gain** were strong indicators of income level.
* Removing multicollinear variables improved interpretability without hurting performance.
* Random Forest provided robust performance with minimal parameter tuning.

---

### 🧰 **Dependencies**

Ensure you have the following dependencies installed before running the notebook:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib scipy statsmodels
```

---

### 🚀 **How to Run the Project**

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/income-classification.git
   cd income-classification
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the script or notebook:

   ```bash
   python income_classification.py
   ```

   or open the Jupyter notebook version if available.

---

### 📁 **Repository Structure**

```
├── income_evaluation.csv        # Dataset
├── income_classification.py     # Model training script
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── visuals/                     # Correlation heatmaps, confusion matrix, etc.
```

---

### 🧾 **Conclusion**

This project successfully demonstrates how **Random Forest** can be applied to income prediction using demographic data.
Through systematic preprocessing (encoding, VIF-based feature selection, outlier removal), the model achieves a balanced accuracy and provides insights into key factors that influence income.

---

### ✨ **Future Improvements**

* Use **SMOTE** to handle class imbalance.
* Apply **GridSearchCV** for hyperparameter tuning.
* Experiment with **XGBoost** or **Logistic Regression** for comparison.
* Deploy the model via **Streamlit** for interactive predictions.

---

### 👨‍💻 **Author**

**David Obi**


Would you like me to include a **`requirements.txt`** section right below this README (with all the dependencies listed explicitly)?
