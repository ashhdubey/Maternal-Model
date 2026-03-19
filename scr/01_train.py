#Random Forest is final

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_data

# Load data
df = pd.read_csv("/Users/ashutoshpdy/Developer/GitHub Project (git clone)/MAA/MAA_Maternal-Awareness-Aid/mlops/data/processed/Maternal Health Risk Data Set.csv")

df = preprocess_data(df)

X = df.drop(["risk_level", "risk_score"], axis=1)
y = df["risk_level"]

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Train final model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "/Users/ashutoshpdy/Developer/GitHub Project (git clone)/MAA/MAA_Maternal-Awareness-Aid/mlops/model/rf_model.pkl")
joblib.dump(le, "/Users/ashutoshpdy/Developer/GitHub Project (git clone)/MAA/MAA_Maternal-Awareness-Aid/mlops/model/label_encoder.pkl")

print("Model training complete. Artifacts saved.")