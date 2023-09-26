############################## ML Decision Tree Classifier #############################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


# For out-of-vocabulary words we don't have to do anything since the CountVectorizer will ignore them
def ml_decision_tree_classifier(x_train, y_train, x_test, y_test):
    # Transforming the data
    vectorizer = CountVectorizer()
    x_train_bow = vectorizer.fit_transform(x_train)
    x_test_bow = vectorizer.transform(x_test)

    clf = DecisionTreeClassifier()
    clf.fit(x_train_bow, y_train)

    y_pred = clf.predict(x_test_bow)

    # cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    # disp.plot()
    # plt.title("ML Decision Tree Confusion Matrix (no Duplicates)")
    # plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    c_report = classification_report(y_test, y_pred, zero_division=0)

    return accuracy * 100, c_report


def ml_decision_tree_classifier_prompt(x_train, y_train):
    vectorizer = CountVectorizer()
    x_train_bow = vectorizer.fit_transform(x_train)
    clf = DecisionTreeClassifier()
    clf.fit(x_train_bow, y_train)

    while True:
        utterance = input("Please enter utterance to be classified (Type '0' to go back): ")
        if utterance == '0':
            print("Exiting...")
            break

        utterance_bow = vectorizer.transform([utterance])
        predicted_label = clf.predict(utterance_bow)[0]

        print(f"Predicted dialog act label: {predicted_label}")

#function to get (and train) the logisitc regression classifier
def get_decision_tree_classifier(x_train,y_train):
    vectorizer = CountVectorizer()
    x_train_bow = vectorizer.fit_transform(x_train)

    clf = DecisionTreeClassifier()
    clf.fit(x_train_bow, y_train)
    return clf


############################## ML Decision Tree Classifier #############################

########################### ML Logistic Regression Classifier ##########################

def ml_logistic_regression_classifier(x_train, y_train, x_test, y_test):
    vectorizer = CountVectorizer()
    x_train_bow = vectorizer.fit_transform(x_train)
    x_test_bow = vectorizer.transform(x_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train_bow, y_train)

    y_pred = clf.predict(x_test_bow)

    # cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    # disp.plot()
    # plt.title("ML Logistic Regression Confusion Matrix (no Duplicates)")
    # plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    c_report = classification_report(y_test, y_pred, zero_division=0)

    return accuracy * 100, c_report


def ml_logistic_regression_classifier_prompt(x_train, y_train):
    vectorizer = CountVectorizer()
    x_train_bow = vectorizer.fit_transform(x_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train_bow, y_train)

    while True:
        utterance = input("Please enter utterance to be classified (Type '0' to go back): ")
        if utterance == '0':
            print("Exiting...")
            break

        utterance_bow = vectorizer.transform([utterance])
        predicted_label = clf.predict(utterance_bow)[0]

        print(f"Predicted dialog act label: {predicted_label}")

#function to get (and train) the logisitc regression classifier
def get_logistic_regression_classifier(x_train,y_train):
    vectorizer = CountVectorizer()
    x_train_bow = vectorizer.fit_transform(x_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train_bow, y_train)
    return clf


########################### ML Logistic Regression Classifier ##########################
