import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from Read_Restaurant_Data import find_restaurants
from Read_Restaurant_Data import filter_restaurants_opt_requirements
from Read_Restaurant_Data import choose_restaurant

from Input_Output_Functions import print_system_message
from Input_Output_Functions import prompt_input
from Input_Output_Functions import set_configurations
from Analyze_Utterance import check_misspelling_or_preferences
from Analyze_Utterance import identify_details

# Load our configurations
with open("configurations.json", "r") as f:
    configurations = json.load(f)

dialog_restart_on = configurations['dialog_restart_on']
caps_on = configurations['caps_on']
random_preference_order_on = configurations['random_preference_order_on']


# Used to store the preferences of the user
preferenceField = {
    'area': None,
    'pricerange': None,
    'food': None
}

optionalPreferences = {
    'touristic':None,
    'assigned_seats':None,
    'children':None,
    'romantic':None
}



# State transistion function to change the state
# @cur_state: int, the current state
# @cur_dialog_act: string, the predicted dialog act for the current utterance
# @cur_utterance: string, the current utterance provided by the user
def state_transition_function(cur_state, cur_dialog_act, cur_utterance):
    
    match cur_state:
        case 1 | 2 | 3 | 4 | 5:
            if cur_dialog_act != 'inform' and cur_dialog_act != 'reqalts' and cur_dialog_act != 'request':
                 return checkPreferences()

            preferences_or_misspelling = check_misspelling_or_preferences(cur_utterance, cur_state)
            if type(preferences_or_misspelling) == str:
                print_system_message(2,preferenceField=preferenceField,optionalPreferences=optionalPreferences, misspelling=preferences_or_misspelling)
                return 2

            update_preferences(preferences_or_misspelling, current_state=cur_state)
            return checkPreferences()
        
        case 6 | 7:
            if cur_dialog_act == 'bye' or cur_dialog_act == 'thankyou':
                return 8
            if cur_dialog_act != 'inform':
                return 6  
            
            # First thing to do is to check whether there was a misspelling
            preferences_or_misspelling = check_misspelling_or_preferences(cur_utterance, cur_state)
            
            if type(preferences_or_misspelling) == str:
                print_system_message(7,caps_on=caps_on,preferenceField=preferenceField,optionalPreferences=optionalPreferences, misspelling=preferences_or_misspelling)
                return 7

            update_preferences(preferences_or_misspelling, current_state=cur_state)
            return checkPreferences()
        
        case 8:
            if cur_dialog_act in ['bye', 'thankyou', 'ack', 'confirm','affirm']:
                return 12
            return 6

        case 9:
            update_opt_requirements(cur_utterance)
            restaurants = find_restaurants(preferenceField)
            restaurants = filter_restaurants_opt_requirements(restaurants, optionalPreferences)
            if len(restaurants) > 0:
                return 10
            return 9
        
        case 10:
            if cur_dialog_act == 'request':
                return 11
            if cur_dialog_act in ['bye', 'thankyou']:
                return 12
            return 10
        
        case 11:
            if cur_dialog_act in ['bye', 'thankyou']:
                return 12
            return 11
        
        case 12:
            return -1

# Function to update the additional preferences
# @cur_utterance: string, the current utterance provided by the user
def update_opt_requirements(cur_utterance):
    optionalPreferences['touristic'] = True if 'touristic' in cur_utterance else False
    optionalPreferences['assigned_seats'] = True if 'seats' in cur_utterance else False
    optionalPreferences['children'] = True if 'child' in cur_utterance else False
    optionalPreferences['romantic'] = True if 'romantic' in cur_utterance else False

# Function to perform checks on the presence of the preferences
# For the transition function (see model diagram -> long column of diamonds)
def checkPreferences():
    if len(find_restaurants(preferenceField)) == 0:
        return 6
    
    if preferenceField['area'] == None:
        return 3
    
    if preferenceField['pricerange'] == None:
        return 4
    
    if preferenceField['food'] == None:
        return 5
    
    return 9


# Function to train the machine learning model
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




def update_preferences(preferences, current_state):
    for key in ['area','pricerange','food']:
        if preferences[key] is not None:
            if preferenceField[key] is None:
                preferenceField[key] = preferences[key]
                if not random_preference_order_on:
                    return
            elif current_state in [6,7]:
                preferenceField[key] = preferences[key]
        elif preferenceField[key] is None and not random_preference_order_on:
            return




def reset_conversation():
    preferenceField['area'] = None
    preferenceField['pricerange'] = None
    preferenceField['food'] = None
    optionalPreferences['romantic'] = None
    optionalPreferences['children'] = None
    optionalPreferences['touristic'] = None
    optionalPreferences['assigned_seats'] = None



def main():

    restart_after_conf = set_configurations()
    if restart_after_conf:

        return

    restart_flag = False

    candidate_restaurants = []
    suggested_restaurants = [[None]]
    current_restaurant = None
    current_state = 1

    vectorizer, clf = train_ml_model()
    print_system_message(current_state,preferenceField=preferenceField,optionalPreferences=optionalPreferences)

    while True:

        # Restart conversation block
        if dialog_restart_on and restart_flag:
            if caps_on:
                print("Restarting conversation...".upper())
            else:
                print("Restarting conversation...")

            # Reset global variables
            reset_conversation()
            candidate_restaurants = []
            suggested_restaurants = [[None]]
            current_restaurant = None
            current_state = 1   
            restart_flag = False
            #print message after current_state is reset
            print_system_message(current_state,preferenceField=preferenceField,optionalPreferences=optionalPreferences)

        if current_state == 12:
            break
    
        #print("Current state: ", current_state)

        predicted_label, utterance = prompt_input(vectorizer, clf)

        if dialog_restart_on and "restart conversation" in utterance: 
            restart_flag = True
            continue

        current_state = state_transition_function(current_state, predicted_label, utterance)
        
        #print(predicted_label, " | ", utterance, '(', preferenceField['area'], ' ',preferenceField['pricerange'], ' ',preferenceField['food'], ')')

        if current_state == 2 or current_state == 7: #this case is handled inside of state_transition_function
            continue
        # If we want to suggest a restaurant, we have to find one
        if current_state == 10:
            # If candidate restaurants not computed yet, find them now
            if len(candidate_restaurants) == 0:
                candidate_restaurants = find_restaurants(preferenceField)
                candidate_restaurants = filter_restaurants_opt_requirements(candidate_restaurants, optionalPreferences)
            # If not all candidate_restaurants were suggested, choose new restaurant to suggest
            if len(candidate_restaurants) > len(suggested_restaurants) or suggested_restaurants[0][0] is None:
                current_restaurant = choose_restaurant(candidate_restaurants, np.array(suggested_restaurants))
                if suggested_restaurants[0][0] is None:
                    suggested_restaurants = [current_restaurant]
                else:
                    suggested_restaurants.append(current_restaurant)

        detail = identify_details(current_state, utterance)
        print_system_message(current_state,preferenceField=preferenceField,optionalPreferences=optionalPreferences, restaurant=current_restaurant, detail=detail)

if __name__ == "__main__":
    main()
