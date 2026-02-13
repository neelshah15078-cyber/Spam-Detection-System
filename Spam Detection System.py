import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', names=["label", "message"])

data["label"] = data["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vector, y_train)

y_pred = model.predict(X_test_vector)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

print("\n📩 Spam Detection System (Type 'exit' to stop)")

while True:
    user_msg = input("\nEnter message: ")

    if user_msg.lower() == "exit":
        print("Program Closed.")
        break

    msg_vector = vectorizer.transform([user_msg])
    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        print("⚠️ This message is SPAM")
    else:
        print("✅ This message is NOT Spam")
        