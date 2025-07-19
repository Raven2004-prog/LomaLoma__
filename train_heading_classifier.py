import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your labeled dataset (JSON format)
with open("data/labeled_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare feature vectors and labels
X = []
y = []
for item in data:
    # Features to use — make sure these exist in your labeled JSON
    features = [
        item["font_size"],
        item["line_width"],
        item["line_height"],
        item["char_count"],
        item["y_position"]
    ]
    X.append(features)
    y.append(item["label"])

# Encode labels: H1 → 0, H2 → 1, H3 → 2
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split for sanity check
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification report:")
from sklearn.utils.multiclass import unique_labels

# Get only the labels that actually appear in y_test and y_pred
used_labels = unique_labels(y_test, y_pred)
used_class_names = le.inverse_transform(used_labels)

print(classification_report(y_test, y_pred, labels=used_labels, target_names=used_class_names))


# Save model and label encoder
joblib.dump(clf, "models/heading_classifier.joblib")
joblib.dump(le, "models/label_encoder.joblib")
print("✅ Model and encoder saved.")
