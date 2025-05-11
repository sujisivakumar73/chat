import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

class IntentClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        self.intents = []
        self.tags = []

    def train(self, intent_file="data/intents.json"):
        with open(intent_file) as f:
            data = json.load(f)

        X, y = [], []
        self.intents = data["intents"]

        for intent in self.intents:
            for pattern in intent["patterns"]:
                X.append(pattern)
                y.append(intent["tag"])

        X_vec = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vec, y)
        self.tags = y

    def predict(self, text):
        X_test = self.vectorizer.transform([text])
        tag = self.classifier.predict(X_test)[0]
        return tag

    def get_response(self, tag):
        for intent in self.intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
