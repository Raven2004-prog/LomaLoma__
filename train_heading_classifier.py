import json
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

# Load enriched feature dataset
with open("data/labeled_data_with_features.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Define features to use (layout + semantic)
features_to_use = [
    "font_size",
    "line_width",
    "line_height",
    "char_count",
    "y_position",
    "is_all_caps",
    "is_title_case",
    "starts_with_number",
    "contains_colon",
    "contains_year",
    "word_count",
    "avg_word_len",
    "named_entity_ratio"
]

X = []
y = []
for item in data:
    try:
        feature_vector = [
            float(item.get(f, 0)) if isinstance(item.get(f), (int, float)) else int(item.get(f, False))
            for f in features_to_use
        ]
        X.append(feature_vector)
        y.append(item["label"])
    except KeyError as e:
        print(f"Skipping line due to missing feature: {e}")

# Encode labels (e.g., H1 → 0, H2 → 1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the XGBoost classifier
clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
used_labels = unique_labels(y_test, y_pred)
used_class_names = le.inverse_transform(used_labels)

print("Classification Report:")
print(classification_report(y_test, y_pred, labels=used_labels, target_names=used_class_names))

# Save model and label encoder
joblib.dump(clf, "models/heading_classifier.joblib")
joblib.dump(le, "models/label_encoder.joblib")
print("✅ XGBoost model and encoder saved.")
