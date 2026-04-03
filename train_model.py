import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# ==========================================
# 1. CUSTOM LOGIC (Fixed attribute name)
# ==========================================
class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        # 👇 app.py এই নামটা খুঁজছে, তাই এটা পরিবর্তন করলাম
        self.features_to_drop = [] 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X 
    
# ==========================================
# 2. DATA LOAD & SPLIT 
# ==========================================
print("📂 Loading Dataset...")

try:
    df = pd.read_csv(r"data\merged_obesity.csv")
except:
    df = pd.read_csv("merged_obesity.csv")

X = df.drop(columns=["NObeyesdad"])
y = df["NObeyesdad"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ==========================================
# 3. PREPROCESSING PIPELINE 
# ==========================================
numeric_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
categorical_features = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# ==========================================
# 4. FULL PIPELINE 
# ==========================================
print("🛠️ Building Pipeline...")
pipeline = Pipeline([
    ('preprocessor', preprocessor),          
    ('corr_drop', CorrelationDropper()),     
    ('model', XGBClassifier(                 
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])

# ==========================================
# 5. TRAINING 
# ==========================================
print("🚀 Training Started...")
pipeline.fit(X_train, y_train)
print("✅ Training Completed!")

# ==========================================
# 6. EVALUATION 
# ==========================================
print("\n📊 Model Performance:")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)

# ==========================================
# 7. SAVING 
# ==========================================
print("💾 Saving Model to .pkl file...")
joblib.dump(pipeline, "models/obesity_best_pipeline.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("🎉 Done! Model is ready for app.py")