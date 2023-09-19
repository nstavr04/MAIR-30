import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# State transition function (Integer with ifs or something)

# Add the classify function from Part1a here
    # Probably train the ml model once at the start of every run

# Function for the keyword matching algorithm (to extract preferences)
    # A subfunction or something for the levenshtein distance before we check for keywords
        # For levenshtein:
            # If distance < 4 we choose the closest one
                # Equal distance we choose random
            # If distance >= 4 system should re-ask the preference with a suitable error message
    # Input to lowercase before anything

# A function for the utterances templates that the system will reply with (e.g Which {preference} do you prefer?) - will see how to do this

# A function that once 3 preferences are filled the system will provide a resto recommendation to the user
    # Basically filter from the .csv file (can keep all the 3 address,phone,postcode for all the suggestions)
    # For multiple suggestions we just pick random
    # If no suggestion, have a message to inform user.

# Check if the test cases are handled correctly

# 1. Welcome
# 2. Ask for correction with error message
# 3. Ask Area
# 4. Ask Price Range
# 5. Ask Food Type
# 6. Express no resto available
# 7. Ask for correction
# 8. Confirm user wants to leave
# 9. Suggest Restaurant
# 10. Provide asked restaurant details
# 11. Goodbye

def state_transition_function(cur_state, cur_dialog_act, cur_utterance):

    next_state = 'Test'

    return next_state

def train_ml_model():

    df = pd.read_csv('dialog_acts.dat', names=['data'])
    df[['label', 'text']] = df['data'].apply(lambda x: pd.Series(x.split(' ', 1)))

    df.drop('data', axis=1, inplace=True)

    # Features and Labels
    x = df['text']
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=11, shuffle=True)

    vectorizer = CountVectorizer()
    x_train_bow = vectorizer.fit_transform(x_train)
    
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(x_train_bow, y_train)

    return vectorizer, clf

def prompt_input(vectorizer, clf):

    utterance = input("Please enter utterance: ").lower()

    utterance_bow = vectorizer.transform([utterance])
    predicted_label = clf.predict(utterance_bow)[0]

    return predicted_label, utterance

def main():

    print("Dialog management system")

    # possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    current_state = 1

    vectorizer, clf = train_ml_model()

    while True:
        predicted_label, utterance = prompt_input(vectorizer, clf)
        print(predicted_label, " | ", utterance)

        next_state = state_transition_function(current_state, predicted_label, utterance)

        # use levenshtein distance here on utterance


if __name__ == "__main__":
    main()