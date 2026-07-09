import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle

# Loading the dataset
df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Basic structural info
print(df.shape)
print(df.head())
pd.set_option("display.max_columns", None)
print(df.head())
print(df.info())
print(df.describe())

# Checking for missing values
print(df.isnull().sum())

# Dropping customerID column
df = df.drop(columns=["customerID"])

# Filling the missing values and changing the data type for TotalCharges 
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)

print(df.head())


# Target Variable
print(df["Churn"].value_counts())

sns.countplot(x="Churn",data=df)
plt.title("Customer Churn Distribution")
plt.show()

# Identifying numercial columns
numerical_columns = df.select_dtypes(include=["int64","float64"]).columns
print(numerical_columns)

# Histogram
for column in numerical_columns:

    plt.figure(figsize=(6,4))

    sns.histplot(df[column],kde=True)

    plt.title(column)

    plt.show()

# Boxplots
for column in numerical_columns:

    plt.figure(figsize=(6,4))

    sns.boxplot(x=df[column])

    plt.title(column)

    plt.show()

# Identifying categorical columns
categorical_columns = df.select_dtypes(include=["string"]).columns
print(categorical_columns)

# Categorical feature analysis
for column in categorical_columns:

    plt.figure(figsize=(7,4))

    sns.countplot(data=df,x=column)

    plt.xticks(rotation=45)

    plt.title(column)

    plt.show()


# Churn vs Numercial features
for column in numerical_columns:

    plt.figure(figsize=(6,4))

    sns.boxplot(x="Churn",y=column,data=df)

    plt.show()


# Churn vs Categorical features
for column in categorical_columns:

    plt.figure(figsize=(8,4))

    sns.countplot(data=df,x=column,hue="Churn")

    plt.xticks(rotation=45)

    plt.show()

# Label encoding of Target column
df["Churn"] = df["Churn"].astype(str).str.strip()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(df["Churn"].head())
print(df["Churn"].value_counts())
print(df["Churn"].dtype)


# Recreate column lists after changes
categorical_columns = df.select_dtypes(include="string").columns.tolist()
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical columns:")
print(categorical_columns)

print("Numerical columns:")
print(numerical_columns)

# Encoding categorical columns
label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder

# Heatmap

plt.figure(figsize=(20,15))

sns.heatmap(df.corr(),annot=True,cmap="coolwarm")

plt.show()

# Feature selection
X = df.drop("Churn",axis=1)

y = df["Churn"]

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Training

# Logistic Regression
lr = LogisticRegression(max_iter=5000, solver="liblinear", random_state=42)
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)

# Gradient Boosting
gb = GradientBoostingClassifier(
    learning_rate=0.05,
    n_estimators=200,
    max_depth=3,
    random_state=42)
gb.fit(X_train,y_train)
gb_pred = gb.predict(X_test)

# XGBoost
xgb = XGBClassifier(learning_rate=0.05, n_estimators=300, max_depth=4, random_state=42, eval_metric="logloss")
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)


# Evaluation function

def evaluate_model(model_name,y_test,predictions):

    print("-"*50)
    print(model_name)
    print("-"*50)
    print("Accuracy :",accuracy_score(y_test,predictions))
    print("Precision :",precision_score(y_test,predictions))
    print("Recall :",recall_score(y_test,predictions))
    print("F1 Score :",f1_score(y_test,predictions))
    print()
    print(confusion_matrix(y_test,predictions))
    print()
    print(classification_report(y_test,predictions))


# Call

evaluate_model("Logistic Regression",y_test,lr_pred)
evaluate_model("Random Forest",y_test,rf_pred)
evaluate_model("Gradient Boosting",y_test,gb_pred)
evaluate_model("XGBoost",y_test,xgb_pred)

# Comparing Models

results = pd.DataFrame({

    "Model":[

        "Logistic Regression",

        "Random Forest",

        "Gradient Boosting",

        "XGBoost"

    ],

    "Accuracy":[

        accuracy_score(y_test,lr_pred),

        accuracy_score(y_test,rf_pred),

        accuracy_score(y_test,gb_pred),

        accuracy_score(y_test,xgb_pred)

    ],

    "Precision":[

        precision_score(y_test,lr_pred),

        precision_score(y_test,rf_pred),

        precision_score(y_test,gb_pred),

        precision_score(y_test,xgb_pred)

    ],

    "Recall":[

        recall_score(y_test,lr_pred),

        recall_score(y_test,rf_pred),

        recall_score(y_test,gb_pred),

        recall_score(y_test,xgb_pred)

    ],

    "F1 Score":[

        f1_score(y_test,lr_pred),

        f1_score(y_test,rf_pred),

        f1_score(y_test,gb_pred),

        f1_score(y_test,xgb_pred)

    ]

})

print(results)


# Plotting model comparison
results.set_index("Model").plot(
    kind="bar",
    figsize=(10,6))

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.show()


# Choosing the best model
best_model = lr


# Saving the model
pickle.dump(best_model, open("churn_model.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
