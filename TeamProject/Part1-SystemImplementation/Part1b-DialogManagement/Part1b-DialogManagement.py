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

# fields that need to be filled:
area = None
pricerange = None
food = None


# state transistion function to change the state
# @cur_state: int, the current state
# @cur_dialog_act: string, the predicted dialog act for the current utterance
# @cur_utterance: string, the current utterance provided by the user
def state_transition_function(cur_state, cur_dialog_act, cur_utterance):
    # check the current state
    # first thing to do is to check whether there was a misspelling
    misspelling_detected = False  # TODO REPLACE WITH FUNCTION FOR EACH CASE
    next_state = -1
    match cur_state:
        case 1, 2, 3, 4, 5:
            if misspelling_detected:
                return 2
            if cur_dialog_act != 'inform':
                return checkPreferences()
            # todo update known information using appropriate function

            return checkPreferences()
        case 6, 7:
            if misspelling_detected:
                return 7
            if cur_dialog_act == 'bye' or 'thankyou':
                return 8
            if cur_dialog_act != 'inform':
                return 6  # this is the same (but more efficient) as checkPreferences()
            # todo update known information using appropriate function
            return checkPreferences()
        case 8:
            # check if user really wants to leave
            if cur_dialog_act in ['bye', 'thankyou', 'ack', 'confirm']:
                return 11
            return 6
        case 9:
            if cur_dialog_act == 'request':
                return 10
            if cur_dialog_act in ['bye', 'thankyou']:
                return 11
            return 9
        case 10:
            if cur_dialog_act in ['bye', 'thankyou']:
                return 11
            return 10
        case 11:
            return -1

    return next_state


# Function to perform checks on the presence of the preferences
# for the transition function (see model diagram -> long column of diamonds)
def checkPreferences():
    # if there is no restraunt matching the preferred attributes transite to state 6
    if len(findRestaurants(area, pricerange, food)) == 0:
        return 6
    # if the preferred area is unknown move to transite 3
    if area == None:
        return 3
    # if the preferred pricerange is unknown move to transite 4
    if pricerange == None:
        return 4
    # if preferred Foodtype is unkown transite to state 5
    if food == None:
        return 5
    # transite to state 6 to suggest a restaurant
    return 9


# lookup function to find restaurants fitting the criteria in the .csv file
# attributes should be provided as string
# @return is a numpy array
def findRestaurants(area='X', price='X', food='X', path='restaurant_info.csv'):
    restaurants = pd.read_csv(path)
    if area != 'X':
        restaurants = restaurants[restaurants['area'] == area]
    if price != 'X':
        restaurants = restaurants[restaurants['pricerange'] == price]
    if food != 'X':
        restaurants = restaurants[restaurants['food'] == food]
    return restaurants.values


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
    utterance = input("Please enter utterance: ")

    utterance_bow = vectorizer.transform([utterance])
    predicted_label = clf.predict(utterance_bow)[0]

    return predicted_label, utterance


# 1. Welcome
# 2. Ask for correction with error message
# 3. Ask Area
# 4. Ask Price Range
# 5. Ask Food Type
# 6. Express no resto available
# 7. Confirm user wants to leave
# 8. Suggest Restaurant
# 9. Provide asked restaurant details
# 10. Goodbye

def main():
    print("Dialog management system")

    # possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    current_state = 1

    vectorizer, clf = train_ml_model()

    while True:
        predicted_label, utterance = prompt_input(vectorizer, clf)
        print(predicted_label)

        state_transition_function(current_state, predicted_label, utterance)


if __name__ == "__main__":
    main()
