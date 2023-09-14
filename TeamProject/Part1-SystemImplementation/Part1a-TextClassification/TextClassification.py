import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load data
df = pd.read_csv('dialog_acts.dat',sep=',',names=['data'])
df[['data']] = df['data'].apply(lambda x: pd.Series(x.lower()))
df[['label','utterance']] = df['data'].apply(lambda x: pd.Series(x.split(' ', 1)))

#save labeles and utterances
y = df['label']
x = df['utterance']
print(len(y))
#remove duplicates
df = df.drop_duplicates(subset=['label','utterance'])
#save cleaned data
y_clean = df['label']
x_clean = df['utterance']

#split data into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=10)
x_clean_train,x_clean_test,y_clean_train,y_clean_test = train_test_split(x_clean,y_clean,test_size=0.15,random_state=10)

#define first baseline prediction methode
def majority_baseline(utterance):
    return 'inform'

#define 2nd baseline prediction method(using rules)

# The order of the rules plays a big role in the accuracy
# I tried to put the class labels first that appear the most times in the dataset. (Hence is makes sense to have inform as first rule)
rules = {
    # I put thank you first because in the text whenever theres thank you and bye at the same sentence
    # it will always be thankyou
    'inform': ['any', 'im looking'],
    'request': ['phone', 'address', 'postcode', 'post code', 'type of food', 'what', 'whats'],
    'thankyou': ['thank you'],
    'ack': ['okay','ok'],
    'affirm': ['yes', 'right', 'yeah'],
    'bye': ['good bye','bye'],
    'deny': ['wrong','not', 'dont'],
    'hello': ['hello', 'hi'],
    'negate': ['no'],
    'repeat': ['repeat', 'again', 'back'],
    'reqalts': ['is there', 'how about', 'anything else', 'what about'],
    'confirm': ['is it', 'does it'],
    'reqmore': ['more'],
    'restart': ['reset', 'start over', 'start again'],
    'null': ['cough', 'unintelligible', 'tv_noise', 'noise', 'sil', 'none']
}

def rules_baseline(utterance):

    for potential_class in rules.keys():
        for key_word in rules[potential_class]:
            if key_word in utterance:
                return potential_class
    return 'inform'

#calc bags of words
vectorizer = CountVectorizer()
vectorizer = vectorizer.fit(x)
x_train_bow = vectorizer.transform(x_train)
x_test_bow = vectorizer.transform(x_test)

x_train_clean_bow = vectorizer.transform(x_clean_train)
x_test_clean_bow = vectorizer.transform(x_clean_test)

#define prediction method for Decision Trees
clf = DecisionTreeClassifier()
clf.fit(x_train_bow,y_train)

#y_pred = clf.predict(x_test_bow)
#print('Accuracy DTC with duplicates: ',accuracy_score(y_test, y_pred))

clf_clean = DecisionTreeClassifier()
clf_clean.fit(x_train_clean_bow,y_clean_train)
#y_pred = clf_clean.predict(x_test_clean_bow)
#print('Accuracy DTC without duplicates: ' + str(accuracy_score(y_clean_test, y_pred)))


def tree_prediction(utterance):
    bow = vectorizer.transform([utterance])
    return clf.predict(bow)[0]

def tree_clean_prediction(utterance):
    bow = vectorizer.transform([utterance])
    return clf_clean.predict(bow)[0]

#define prediction method for Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(x_train_bow,y_train)

#y_pred = log_reg.predict(x_test_bow)
#print('Accuracy LR with duplicates: ',accuracy_score(y_test, y_pred))

log_reg_clean = LogisticRegression()
log_reg_clean.fit(x_train_clean_bow,y_clean_train)
#y_pred = log_reg_clean.predict(x_test_clean_bow)
#print('Accuracy LR without duplicates: ' + str(accuracy_score(y_clean_test, y_pred)))


def lr_prediction(utterance):
    bow = vectorizer.transform([utterance])
    return log_reg.predict(bow)[0]

def lr_clean_prediction(utterance):
    bow = vectorizer.transform([utterance])
    return log_reg_clean.predict(bow)[0]

def main_menu():
    while True:
        print("\nChoose an option:")
        print("1. Predict wit majority baseline model")
        print("2. Predict with rule-based baseline model")
        print("3. Predict with DT model trained on all data")
        print("4. Predict with DT model trained on cleaned data")
        print("5. Predict with LR model trained on all data")
        print("6. Predict with LR model trained on cleaned data")
        print("7. Show accuracies on test set")
        print("0. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            while True:
                utterance = input("Please enter utterance to be classified (Type 'exit' to go back): ")
                if utterance == 'exit':
                    print("Exiting...")
                    break
                print(majority_baseline(utterance))

        elif choice == '2':
            while True:
                utterance = input("Please enter utterance to be classified (Type 'exit' to go back): ")
                if utterance == 'exit':
                    print("Exiting...")
                    break
                print(rules_baseline(utterance))
        elif choice == '3':
            while True:
                utterance = input("Please enter utterance to be classified (Type 'exit' to go back): ")
                if utterance == 'exit':
                    print("Exiting...")
                    break
                print(tree_prediction(utterance))
        elif choice == '4':
            while True:
                utterance = input("Please enter utterance to be classified (Type 'exit' to go back): ")
                if utterance == 'exit':
                    print("Exiting...")
                    break
                print(tree_clean_prediction(utterance))
        elif choice == '5':
            while True:
                utterance = input("Please enter utterance to be classified (Type 'exit' to go back): ")
                if utterance == 'exit':
                    print("Exiting...")
                    break
                print(lr_prediction(utterance))
        elif choice == '6':
            while True:
                utterance = input("Please enter utterance to be classified (Type 'exit' to go back): ")
                if utterance == 'exit':
                    print("Exiting...")
                    break
                print(lr_clean_prediction(utterance))
        elif choice == '7':
            pred_correctly = [x for x in y_test if x == 'inform']
            print(f"Accuracy Majority Baseline Model: {len(pred_correctly) / len(y_test)}")
            counter_correct = 0
            for index in x_test.index:
                if rules_baseline(x_test[index]) == y_test[index]:
                    counter_correct += 1
            print('Accuracy rule-based Baseline Model', counter_correct / len(y_test))
            y_pred = clf.predict(x_test_bow)
            print('Accuracy DTC with duplicates: ', accuracy_score(y_test, y_pred))

            y_pred = clf_clean.predict(x_test_clean_bow)
            print('Accuracy DTC without duplicates: ' + str(accuracy_score(y_clean_test, y_pred)))
            y_pred = log_reg.predict(x_test_bow)
            print('Accuracy LR with duplicates: ', accuracy_score(y_test, y_pred))
            y_pred = log_reg_clean.predict(x_test_clean_bow)
            print('Accuracy LR without duplicates: ' + str(accuracy_score(y_clean_test, y_pred)))
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()