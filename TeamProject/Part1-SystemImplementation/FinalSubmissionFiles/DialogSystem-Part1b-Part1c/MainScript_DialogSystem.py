import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from Filter_Restaurants import find_restaurants
from Filter_Restaurants import filter_restaurants_opt_requirements
from Filter_Restaurants import choose_restaurant

from Input_Output_Functions import print_system_message
from Input_Output_Functions import prompt_input
from Input_Output_Functions import set_configurations
from Analyze_Utterance import check_misspelling_or_preferences
from Analyze_Utterance import identify_details

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

# Dictonary containing the reasing rule
# Each attribute has a set of rules. If each rules is met by a restaurant it has that attribute
reasoning_rules = {
    'touristic' : [[1,'cheap',True],[7,'good',True],[3,'romanian',False]],
    'assigned_seats' : [[8,'busy',True]],
    'children' : [[9,'long_stay',False]],
    'romantic' : [[8,'busy',False],[9,'long_stay',True]]
}

# Load our configurations
with open("configurations.json", "r") as f:
    configurations = json.load(f)

dialog_restart_on = configurations['dialog_restart_on']
caps_on = configurations['caps_on']
random_preference_order_on = configurations['random_preference_order_on']

# State transistion function to change the state
# @cur_state: int, the current state
# @cur_dialog_act: string, the predicted dialog act for the current utterance
# @cur_utterance: string, the current utterance provided by the user
def state_transition_function(cur_state, cur_dialog_act, cur_utterance):
    
    match cur_state:
        case '1_Welcome' | '2_AskCorrection' | '3_AskArea' | '4_AskPriceRange' | '5_AskFoodType':
            if cur_dialog_act != 'inform' and cur_dialog_act != 'reqalts' and cur_dialog_act != 'request':
                 return checkPreferences()

            preferences_or_misspelling = check_misspelling_or_preferences(cur_utterance, cur_state)
            if type(preferences_or_misspelling) == str:
                print_system_message('2_AskCorrection',preferenceField=preferenceField,optionalPreferences=optionalPreferences, misspelling=preferences_or_misspelling)
                return '2_AskCorrection'

            update_preferences(preferences_or_misspelling, current_state=cur_state)
            return checkPreferences()
        
        case '6_NoRestaurantExists' | '7_AskCorrection':
            if cur_dialog_act == 'bye' or cur_dialog_act == 'thankyou':
                return '8_ConfirmExit'
            if cur_dialog_act != 'inform':
                return '6_NoRestaurantExists'  
            
            # First thing to do is to check whether there was a misspelling
            preferences_or_misspelling = check_misspelling_or_preferences(cur_utterance, cur_state)
            
            if type(preferences_or_misspelling) == str:
                print_system_message('7_AskCorrection',caps_on=caps_on,preferenceField=preferenceField,optionalPreferences=optionalPreferences, misspelling=preferences_or_misspelling)
                return '7_AskCorrection'

            update_preferences(preferences_or_misspelling, current_state=cur_state)
            return checkPreferences()
        
        case '8_ConfirmExit':
            if cur_dialog_act in ['bye', 'thankyou', 'ack', 'confirm','affirm']:
                return '12_Goodbye'
            return '6_NoRestaurantExists'

        case '9_AskAdditionalRequirements':
            update_opt_requirements(cur_utterance)
            restaurants = find_restaurants(preferenceField)
            restaurants = filter_restaurants_opt_requirements(restaurants, optionalPreferences, reasoning_rules)
            if len(restaurants) > 0:
                return '10_SuggestRestaurants'
            return '9_AskAdditionalRequirements'
        
        case '10_SuggestRestaurants':
            if cur_dialog_act == 'request':
                return '11_ProvideRestaurantDetails'
            if cur_dialog_act in ['bye', 'thankyou']:
                return '12_Goodbye'
            return '10_SuggestRestaurants'
        
        case '11_ProvideRestaurantDetails':
            if cur_dialog_act in ['bye', 'thankyou']:
                return '12_Goodbye'
            return '11_ProvideRestaurantDetails'
        
        case '12_Goodbye':
            return '-1'

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
        return '6_NoRestaurantExists'
    
    if preferenceField['area'] == None:
        return '3_AskArea'
    
    if preferenceField['pricerange'] == None:
        return '4_AskPriceRange'
    
    if preferenceField['food'] == None:
        return '5_AskFoodType'
    
    return '9_AskAdditionalRequirements'

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
    global preferenceField

    for key in ['area','pricerange','food']:
        if preferences[key] is not None:
            if preferenceField[key] is None:
                preferenceField[key] = preferences[key]
                if not random_preference_order_on:
                    return
            elif current_state in ['6_NoRestaurantExists','7_AskCorrection']:
                if random_preference_order_on:
                    preferenceField = {
                        'area': None,
                        'pricerange': None,
                        'food': None
                    }
                    update_preferences(preferences, '1_Welcome')
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
    current_state = '1_Welcome'

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
            current_state = '1_Welcome'   
            restart_flag = False
            # Print message after current_state is reset
            print_system_message(current_state,preferenceField=preferenceField,optionalPreferences=optionalPreferences)

        if current_state == '12_Goodbye':
            break

        predicted_label, utterance = prompt_input(vectorizer, clf)

        if dialog_restart_on and "restart conversation" in utterance: 
            restart_flag = True
            continue

        current_state = state_transition_function(current_state, predicted_label, utterance)
        
        # Used for debugging purposes
        if configurations['debugging_on']:
            print("For debugging purposes: ", "Predicted Dialog Act: ", predicted_label, " | Utterance: ", utterance, '( Area: ', preferenceField['area'], ' PriceRange: ',preferenceField['pricerange'], ' Food: ',preferenceField['food'], ')')

        if current_state == '2_AskCorrection' or current_state == '7_AskCorrection': # This case is handled inside of state_transition_function
            continue
        # If we want to suggest a restaurant, we have to find one
        if current_state == '10_SuggestRestaurants':
            # If candidate restaurants not computed yet, find them now
            if len(candidate_restaurants) == 0:
                candidate_restaurants = find_restaurants(preferenceField)
                candidate_restaurants = filter_restaurants_opt_requirements(candidate_restaurants, optionalPreferences, reasoning_rules)
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
