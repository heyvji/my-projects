import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# --- configure data path ---
# Update this path if your file is at a different location.
DATA_PATH = r"C:\Users\Thippesh\my projects\medha_tech_intern\dataset .csv"
# If you uploaded dataset to /mnt/data in some environments, use that:
# DATA_PATH = "/mnt/data/dataset.csv"

# --- load dataset ---
df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)

# --- check for target column ---
TARGET = "Target"
if TARGET not in df.columns:
    raise ValueError(f"'{TARGET}' column not found in the dataset. Rename your label column to '{TARGET}' or update TARGET variable.")

# --- separate X and y ---
X = df.drop(columns=[TARGET])
y = df[TARGET]

# --- detect numeric and categorical columns ---
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# --- preprocessing pipelines ---
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")

# --- train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

# --- models to evaluate ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (RBF)": SVC()
}

# --- train/evaluate each model in a pipeline ---
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    print("\n=== Training:", name, "===")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, preds, zero_division=0))
    # Optional: print confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))