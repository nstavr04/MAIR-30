# This file contains additional functions for using the baseline models

from sklearn.metrics import classification_report
####################### Baseline majority class (inform label) #######################
# Identify the majority class (idk if its needed)
# majority_class = y_train.value_counts().idxmax()
def baseline_majority(y_test, majority_class='inform'):
    total_instances = len(y_test)
    correct_predictions = (y_test == majority_class).sum()

    # y_pred_baseline = np.full((total_instances), majority_class)
    # unique_labels = np.unique(y_test)

    # cm = confusion_matrix(y_test, y_pred_baseline, labels=unique_labels)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    # disp.plot()
    # plt.title("Baseline Majority Class Confusion Matrix")
    # plt.show()

    accuracy = (correct_predictions / total_instances) * 100
    c_report = classification_report(y_test, [majority_class] * total_instances, zero_division=0)

    return accuracy, c_report
####################### Baseline majority class (inform label) #######################

####################### Baseline majority class prediction ###########################
#predicts the classes of the elements in x_test using the baseline majority class model
def baseline_majority_predict(x_test):
    y_pred = ['inform' for x in x_test]
    return y_pred
####################### Baseline majority class prediction ###########################

############################## Baseline keyword matching #############################
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

def baseline_keyword(x_test, y_test):

    y_pred = []

    for utterance in x_test:
        # Our default label is the inform label since it appears the most times
        predicted_label = 'inform'
        for label, keywords in rules.items():
            if any(keyword in utterance for keyword in keywords):
                predicted_label = label
                break
        y_pred.append(predicted_label)

    # Calculate the accuracy from the predictions
    total_instances = len(y_test)
    correct_predictions = (y_test == y_pred).sum()

    # unique_labels = np.unique(y_test)

    # cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    # disp.plot()
    # plt.title("Baseline Keyword Matching Confusion Matrix")
    # plt.show()

    accuracy = (correct_predictions / total_instances) * 100
    c_report = classification_report(y_test, y_pred, zero_division=0)

    return accuracy, c_report
############################## Baseline keyword matching #############################

############################## Baseline prompt predictions #############################
def baseline_prompt():

    while True:
        utterance = input("Please enter utterance to be classified (Type '0' to go back): ")
        if utterance == '0':
            print("Exiting...")
            break

        predicted_label = 'inform'
        for label, keywords in rules.items():
            if any(keyword in utterance for keyword in keywords):
                predicted_label = label
                break

        print(f"Predicted dialog act label: {predicted_label}")
############################## Baseline prompt predictions #############################

####################### Baseline rules prediction ###########################
#predicts the classes of the elements in x_test using the baseline rulse-based model
def baseline_predict(x_test):
    y_pred = []
    for utterance in x_test:
        predicted_label = 'inform'
        for label, keywords in rules.items():
            if any(keyword in utterance for keyword in keywords):
                predicted_label = label
                break
        y_pred.append(predicted_label)
    return y_pred
####################### Baseline majority class prediction ###########################
