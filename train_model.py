import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training dataset (replace with your real dataset)
emails = [
    "Urgent! Your account password must be updated immediately",
    "Get cheap pills now, limited offer",
    "Hello, here is the report you requested",
    "Congratulations! You've won a free gift",
    "Suspicious login detected in your account",
    "Let's schedule the meeting for tomorrow"
]

labels = [1, 1, 0, 1, 1, 0]  # 1 = spam/malicious, 0 = safe

# Convert text → vectors
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(emails)

# Train Model
model = LogisticRegression()
model.fit(X, labels)

# Save vectorizer & model
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "spam_model.pkl")

print("Model trained & saved successfully!")
