# Customer Churn Prediction

## Contact
**Sami Ullah**  
[GitHub Profile](https://github.com/SamiUllah568)  
[Email](sk2579784@gmail.com)


## Project Aim

This project focuses on predicting customer churn (whether a customer will leave or remain with the company) based on various customer features. Using machine learning algorithms, we can identify patterns in customer behavior that are indicative of a potential churn, enabling the company to take proactive actions to retain valuable customers.

## Objective

To build a model that can predict if a customer will exit (churn) or stay with the company based on features like credit score, geography, age, balance, number of products, has credit card, is active member, etc.

**Key Features Used**:
- Credit Score
- Geography
- Age
- Balance
- Number of Products
- Active Membership Status
- And more...

## Dataset Details
- **Source**: Custom dataset (`Churn_Modelling.csv`)
- **Rows**: 10,000
- **Columns**: 13 features + target (`Exited`)
- **Class Distribution**:
  - 79.63% Non-Churned (0)
  - 20.37% Churned (1)

  ### Key Insights from EDA
- Older customers (>45 years) are more likely to churn.
- Customers with higher balances show higher churn rates.
- Geographic distribution: 50% France, 25% Germany, 25% Spain.

## Installation
### Requirements
```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow xgboost matplotlib seaborn


## Data Manipulation and Analysis Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

## Load Dataset

```python
df = pd.read_csv("/content/Churn_Modelling.csv")
df.shape
df.head()
```

## Data Analysis

### Selecting Relevant Features for Churn Prediction

```python
df = df.iloc[:, 3:14]
print(df.info())
print(df.describe())
df.isnull().sum()
print("Duplicated Values -- >> ", df.duplicated().sum())
for i in df.columns:
    print(f"\nNUnique values in {i} -->> [{df[i].nunique()}]")
    print(f"Unique values in {i} -->> [{df[i].unique()}]\n")
```

### Data Visualization

#### Exited

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=df["Exited"], ax=ax[0])
ax[0].set_title("Distribution of Exited")
ax[0].set_xlabel("Exited")
ax[0].set_ylabel("Count")
ax[1].pie(df["Exited"].value_counts(), labels=df["Exited"].value_counts().index, autopct='%.2f%%', shadow=True, colors=['#ff9999','#66b3ff'])
ax[1].set_title("Exited Proportion")
plt.tight_layout()
plt.show()
```

#### Geography

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=df["Geography"], ax=ax[0])
ax[0].set_title("Distribution of Geography")
ax[0].set_xlabel("Geography")
ax[0].set_ylabel("Count")
ax[1].pie(df["Geography"].value_counts(), labels=df["Geography"].value_counts().index, autopct='%.2f%%', shadow=True, colors=['olivedrab', 'rosybrown', 'gray'], explode=[0.1,0.1,0.1])
ax[1].set_title("Geography Proportion")
plt.tight_layout()
plt.show()
```

#### Gender

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=df["Gender"], ax=ax[0])
ax[0].set_title("Distribution of Gender")
ax[0].set_xlabel("Gender")
ax[0].set_ylabel("Count")
ax[1].pie(df["Gender"].value_counts(), labels=df["Gender"].value_counts().index, autopct='%.2f%%', shadow=True, colors=['#ff9999','#66b3ff'])
ax[1].set_title("Gender Proportion")
plt.tight_layout()
plt.show()
```

#### Age Distribution

```python
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["Age"], kde=True, ax=ax[0])
ax[0].set_title("Distribution of Age")
sns.histplot(data=df, x="Age", hue="Exited", kde=True, ax=ax[1])
ax[1].set_title("Age Distribution by Exited Status")
plt.show()
```

**Insight**: Older customers are more likely to leave the bank.

#### Balance Distribution

```python
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["Balance"], kde=True, ax=ax[0])
ax[0].set_title("Distribution of Balance")
sns.histplot(data=df, x="Balance", hue="Exited", kde=True, ax=ax[1])
ax[1].set_title("Balance Distribution by Exited Status")
plt.show()
```

**Insight**: Customers with higher balances have a higher chance of exiting.

### Correlation Matrix of Numerical Features

```python
num_feature = df.select_dtypes(include='number')
corr_metrix = num_feature.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_metrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=2)
plt.title("Correlation Matrix of Numerical Features", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
```

## Feature Engineering

### Selecting Feature Columns (X) and Target Column (y)

```python
X = df.drop('Exited', axis=1)
y = df["Exited"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### One-Hot Encoding for Categorical Features (Geography and Gender)

```python
ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int8)
x_train_cat = ohe.fit_transform(x_train[["Geography", "Gender"]])
x_test_cat = ohe.transform(x_test[["Geography", "Gender"]])
ohe_feature_names = ohe.get_feature_names_out(["Geography", "Gender"])
x_train_cat = pd.DataFrame(x_train_cat, columns=ohe_feature_names)
x_test_cat = pd.DataFrame(x_test_cat, columns=ohe_feature_names)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
x_train_cat = x_train_cat.reset_index(drop=True)
x_test_cat = x_test_cat.reset_index(drop=True)
x_train.drop(["Geography", "Gender"], axis=1, inplace=True)
x_test.drop(["Geography", "Gender"], axis=1, inplace=True)
x_train = pd.concat([x_train, x_train_cat], axis=1)
x_test = pd.concat([x_test, x_test_cat], axis=1)
```

### Feature Scaling using StandardScaler

```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)
```

### Apply SMOTE (Synthetic Minority Over-sampling Technique) to Balance the Training Data

```python
smote = SMOTE()
x_train, y_train = smote.fit_resample(x_train, y_train)
```

## Model Training

### Artificial Neural Network (ANN)

```python
classifier = Sequential()
classifier.add(Dense(units=11, activation='relu'))
classifier.add(Dense(units=7, activation='relu'))
classifier.add(Dense(units=6, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='auto')
class_weight = {0: 1., 1: 3.}
model_histoy = classifier.fit(x_train, y_train, validation_split=0.33, batch_size=10, epochs=1000, callbacks=early_stop, class_weight=class_weight)
```

### Visualizing Model Accuracy and Loss Over Epochs

```python
plt.plot(model_histoy.history['accuracy'])
plt.plot(model_histoy.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("epochs")
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(model_histoy.history['loss'])
plt.plot(model_histoy.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

### ANN Results

```python
train_pred_ann = (classifier.predict(x_train) > 0.5).astype(int)
test_pred_ann = (classifier.predict(x_test) > 0.5).astype(int)
print("Training Set Metrics:")
print(f"Accuracy: {accuracy_score(y_train, train_pred_ann)}")
print("Confusion Matrix:")
print(confusion_matrix(y_train, train_pred_ann))
print('-' * 20)
print("Testing Set Metrics:")
print(f"Accuracy: {accuracy_score(y_test, test_pred_ann)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred_ann))
```

### RandomForestClassifier

```python
rand = RandomForestClassifier(random_state=42, class_weight='balanced')
rand.fit(x_train, y_train)
train_pred = rand.predict(x_train)
test_pred = rand.predict(x_test)
print("Training Set Metrics:")
print(f"Accuracy: {accuracy_score(y_train, train_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_train, train_pred))
print('-' * 20)
print("Testing Set Metrics:")
print(f"Accuracy: {accuracy_score(y_test, test_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
```

### Logistic Regression

```python
log_reg = LogisticRegression(random_state=42, class_weight='balanced')
log_reg.fit(x_train, y_train)
train_pred = log_reg.predict(x_train)
test_pred = log_reg.predict(x_test)
print("Training Set Metrics:")
print(f"Accuracy: {accuracy_score(y_train, train_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_train, train_pred))
print('-' * 20)
print("Testing Set Metrics:")
print(f"Accuracy: {accuracy_score(y_test, test_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
```

### XGBoost Classifier

```python
from xgboost import XGBClassifier
ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
xgb = XGBClassifier(random_state=42, scale_pos_weight=ratio)
xgb.fit(x_train, y_train)
train_pred = xgb.predict(x_train)
test_pred = xgb.predict(x_test)
print("Training Set Metrics:")
print(f"Accuracy: {accuracy_score(y_train, train_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_train, train_pred))
print('-' * 20)
print("Testing Set Metrics:")
print(f"Accuracy: {accuracy_score(y_test, test_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
```

## Save the Model

```python
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(classifier, file)
with open('model.pkl', 'rb') as file:
    ANN_model = pickle.load(file)
```

## Results

## Results (Test Set)
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| ANN                 | 78.00%   | 0.68      | 0.73   | 0.70     |
| Random Forest       | 86.35%   | 0.85      | 0.38   | 0.52     |
| XGBoost             | 85.65%   | 0.81      | 0.36   | 0.50     |
| Logistic Regression | 68.90%   | 0.37      | 0.84   | 0.51     |

The ANN model performed the best on the Imbalanced Data with good accuracy with good true positive pediction

# Confusion Matrix for ANN (Artificial Neural Network) Model

## Training Set Confusion Matrix:
[[5066 1290]
[1323 5033]]
- **True Negatives (TN):** 5066
- **False Positives (FP):** 1290
- **False Negatives (FN):** 1323
- **True Positives (TP):** 5033

## Testing Set Confusion Matrix:
[[1228 379]
[ 92 301]]

- **True Negatives (TN):** 1228
- **False Positives (FP):** 379
- **False Negatives (FN):** 92
- **True Positives (TP):** 301

---

This confusion matrix is part of the evaluation for the ANN model, showing the performance on both the training and testing sets.

