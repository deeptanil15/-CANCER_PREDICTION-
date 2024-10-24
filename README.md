# Cancer Cell Prediction With Machine Learning

## 1. Project Overview : 

In this project, I developed a system to predict whether a cell is cancerous or not (malignant or benign) using the Logistic Regression algorithm. The system was built using a dataset of features derived from cell nuclei, and Logistic Regression was chosen because it works well for binary classification problems.

## 2. Understanding the Dataset :
   
I used the Breast Cancer Wisconsin dataset, which is widely available in machine learning libraries like scikit-learn. This dataset contains features such as the radius, texture, and smoothness of cell nuclei.

Target Variable: The target variable is binary, where:
0 represents benign (non-cancerous) cells.
1 represents malignant (cancerous) cells.
Features: The dataset includes features such as:
Mean radius
Mean texture
Mean smoothness
Mean compactness, etc.

## 3. Importing Required Libraries :

First, I imported the necessary libraries for data manipulation, model building, and evaluation.

python
copy code
# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## 4. Loading the Dataset :

Next, I loaded the Breast Cancer dataset, which is available in the scikit-learn library.

python
Copy code
# Import dataset from scikit-learn
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()

# Convert to pandas DataFrame for better visualization
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # Adding the target column

## 5. Exploratory Data Analysis (EDA) :

At this stage, I explored the dataset to understand its structure and check for missing values or anomalies.

python
Copy code
# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Descriptive statistics
print(df.describe())
This step allowed me to understand the data distribution, feature importance, and the relationship between features and the target.

## 6. Data Preprocessing :

Before training the model, I prepared the data by splitting it into features (X) and the target (y), and then into training and test sets. I also scaled the features to ensure that all features have equal weight in the model.

python
Copy code
# Split features (X) and target (y)
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features to scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## 7. Logistic Regression Model :

I implemented the Logistic Regression model, trained it on the training data, and then used it to make predictions on the test data.

Code for Model Training:
python
Copy code
# Initialize Logistic Regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, y_train)
## 8. Model Prediction :
Once the model was trained, I used it to predict the target (whether the cell is cancerous or not) on the test data.

python
Copy code
# Predict on test data
y_pred = logreg.predict(X_test)

## 9. Evaluating the Model :

To evaluate how well the model performed, I calculated several metrics like accuracy, confusion matrix, and classification report.

Accuracy Score:
python
Copy code
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
Confusion Matrix:
The confusion matrix shows the true positives, true negatives, false positives, and false negatives.

python
Copy code
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
Classification Report:
The classification report provides precision, recall, and F1-score, which are critical metrics in evaluating model performance.

python
Copy code
# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

## 10. Results and Insights :

After running the model and calculating its performance, I obtained the following metrics:

Accuracy: The model’s accuracy tells how often it correctly predicts whether the cell is cancerous or not.
Precision and Recall: Precision indicates the proportion of correct positive predictions, while recall measures how well the model captures the actual positives (cancerous cases).

## 11. Conclusion

By using the Logistic Regression algorithm, I successfully built a machine learning model to predict cancerous cells. This project demonstrates how Logistic Regression can be applied to binary classification problems in healthcare, specifically in predicting the malignancy of cells. The model performed well with an accuracy of over 90%, and the insights gained from precision, recall, and the confusion matrix allowed me to understand the strengths and limitations of the model.

Full Code:
python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict on test data
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
