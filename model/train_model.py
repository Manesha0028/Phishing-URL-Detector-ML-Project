import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATASET_PATH = os.path.join("..", "dataset", "phishing_url_dataset.csv")

LIVE_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]

df = pd.read_csv(DATASET_PATH)

target_col = "label"

missing = [c for c in LIVE_FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

X = df[LIVE_FEATURES].copy()
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Random Forest ──────────────────────────────────────────────
print("\n=== Random Forest ===")
rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Accuracy :", round(accuracy_score(y_test, y_pred_rf), 4))
print("Precision:", round(precision_score(y_test, y_pred_rf, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred_rf, zero_division=0), 4))
print("F1 Score :", round(f1_score(y_test, y_pred_rf, zero_division=0), 4))

joblib.dump(rf_model, "phishing_live_model.pkl")
joblib.dump(LIVE_FEATURES, "feature_names.pkl")
joblib.dump(False, "uses_scaler.pkl")

# ── Logistic Regression ─────────────────────────────────────────
print("\n=== Logistic Regression ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
print("Accuracy :", round(accuracy_score(y_test, y_pred_lr), 4))
print("Precision:", round(precision_score(y_test, y_pred_lr, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred_lr, zero_division=0), 4))
print("F1 Score :", round(f1_score(y_test, y_pred_lr, zero_division=0), 4))

joblib.dump(lr_model, "phishing_lr_model.pkl")
joblib.dump(scaler, "lr_scaler.pkl")

print("\nSaved:")
print("- phishing_live_model.pkl  (Random Forest)")
print("- feature_names.pkl")
print("- uses_scaler.pkl")
print("- phishing_lr_model.pkl    (Logistic Regression)")
print("- lr_scaler.pkl")