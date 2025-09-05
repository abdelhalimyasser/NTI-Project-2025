import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("heart.csv")

# Define categorical columns
categorical_columns = ['thal', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca']

# Clip outliers and round categorical features
def clip_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    # Round categorical features to integers
    if column in categorical_columns:
        df[column] = df[column].round().astype(int)
    return df

# Apply clipping to all features except target
for col in df.columns:
    if col != 'target':
        df = clip_outliers(df, col)

# Verify unique values for categorical features
print("Unique values after clipping and rounding:")
for col in categorical_columns:
    print(f"{col}: {df[col].unique()}")

# Verify target distribution
print("\nTarget distribution:")
print(df['target'].value_counts())  # Should show 1 = Has Heart Disease, 0 = No Heart Disease

# Split data
X = df.drop('target', axis=1)
y = df['target']  # Ensure no inversion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
logistic_model = LogisticRegression(max_iter=5000, class_weight='balanced')
logistic_model.fit(X_train, y_train)
random_forest_model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
random_forest_model.fit(X_train, y_train)

# Save models
joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(random_forest_model, "random_forest_model.pkl")

# Print feature importances
feature_names = X.columns
importances = random_forest_model.feature_importances_
print("\nFeature Importances:")
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")
