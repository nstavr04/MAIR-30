# This file contains functions to get/train the ML models

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class MLModel:
    def __init__(self, classifier):
        self.clf = classifier
        self.vectorizer = CountVectorizer()

    # For out-of-vocabulary words we don't have to do anything since the CountVectorizer will ignore them
    def ml_classifier(self, x_train, y_train, x_test, y_test):
        x_train_bow = self.vectorizer.fit_transform(x_train)
        x_test_bow = self.vectorizer.transform(x_test)

        self.clf.fit(x_train_bow, y_train)
        y_pred = self.clf.predict(x_test_bow)

        accuracy = accuracy_score(y_test, y_pred) * 100
        c_report = classification_report(y_test, y_pred, zero_division=0)

        return accuracy, c_report

    def prompt_classifier(self, x_train, y_train):
        x_train_bow = self.vectorizer.fit_transform(x_train)
        self.clf.fit(x_train_bow, y_train)

        while True:
            utterance = input("Please enter utterance to be classified (Type '0' to go back): ")
            if utterance == '0':
                print("Exiting...")
                break

            utterance_bow = self.vectorizer.transform([utterance])
            predicted_label = self.clf.predict(utterance_bow)[0]

            print(f"Predicted dialog act label: {predicted_label}")

    def get_classifier(self, x_train, y_train):
        x_train_bow = self.vectorizer.fit_transform(x_train)
        self.clf.fit(x_train_bow, y_train)
        return self.clf, self.vectorizer

# DecisionTree subclass
class DecisionTreeModel(MLModel):
    def __init__(self):
        super().__init__(DecisionTreeClassifier())

# LogisticRegression subclass
class LogisticRegressionModel(MLModel):
    def __init__(self):
        super().__init__(LogisticRegression(max_iter=1000))
