import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import Levenshtein
import random

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

# Fields that need to be filled:
preferenceField = {
    'area': None,
    'pricerange': None,
    'food': None
}

domain_terms_dict = {
    'PriceRange': ['cheap', 'moderate', 'expensive'],
    'Area': ['north', 'south', 'east', 'west', 'central'],
    'Food': ['african', 'asian oriental', 'australasian', 'bistro', 'british',
             'catalan', 'chinese', 'cuban', 'european', 'french', 'fusion',
             'gastropub', 'indian', 'international', 'italian', 'jamaican',
             'japanese', 'korean', 'lebanese', 'mediterranean',
             'modern european', 'moroccan', 'north american', 'persian',
             'polynesian', 'portuguese', 'romanian', 'seafood', 'spanish',
             'steakhouse', 'swiss', 'thai', 'traditional', 'turkish', 'tuscan',
             'vietnamese']
}


# State transistion function to change the state
# @cur_state: int, the current state
# @cur_dialog_act: string, the predicted dialog act for the current utterance
# @cur_utterance: string, the current utterance provided by the user
def state_transition_function(cur_state, cur_dialog_act, cur_utterance):
    # check the current state

    match cur_state:
        case 1, 2, 3, 4, 5:
            # first thing to do is to check whether there was a misspelling
            preferences_or_misspelling = check_misspelling_update_preferences(cur_utterance)
            if type(preferences_or_misspelling) == str:
                #print error here because misspelled word is known
                print_system_message(2, misspelling=preferences_or_misspelling)
                return 2
            if cur_dialog_act != 'inform':
                return checkPreferences()

            update_preferences(preferences_or_misspelling, current_state=cur_state)
            return checkPreferences()
        case 6, 7:
            # first thing to do is to check whether there was a misspelling
            preferences_or_misspelling = check_misspelling_update_preferences(cur_utterance)
            if type(preferences_or_misspelling) == str:
                # print error here because misspelled word is known
                print_system_message(7, misspelling=preferences_or_misspelling)
                return 7
            if cur_dialog_act == 'bye' or 'thankyou':
                return 8
            if cur_dialog_act != 'inform':
                return 6  # this is the same (but more efficient) as checkPreferences()
            update_preferences(preferences_or_misspelling, current_state=cur_state)
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


# Function to perform checks on the presence of the preferences
# for the transition function (see model diagram -> long column of diamonds)
def checkPreferences():
    # if there is no restraunt matching the preferred attributes transite to state 6
    if len(findRestaurants(preferenceField['area'], preferenceField['pricerange'], preferenceField['food'])) == 0:
        return 6
    # if the preferred area is unknown move to transite 3
    if preferenceField['area'] == None:
        return 3
    # if the preferred pricerange is unknown move to transite 4
    if preferenceField['pricerange'] == None:
        return 4
    # if preferred Foodtype is unkown transite to state 5
    if preferenceField['food'] == None:
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

#function to randomly choose a restaurant
#@parameters:
#@restaurants: numpy array that contains restaurants from which one is choosen randomly
#@alreadyUsedRestaurants: numpy array, sublist of restaurant with previously suggest restaurants
#@return: a numpy array containing a restaurant from @restaurants | template: [name,area,pricerange,food,phone,addr,postcode]
def chooseRestaurant(restaurants, alreadyUsedRestaurants):
    if len(restaurants) == 0:
        return None
    if len(restaurants) == len(alreadyUsedRestaurants):
        alreadyUsedRestraunts = []

    restaurants = [i for i in restaurants if i[0] not in alreadyUsedRestraunts[:,0]]
    return restaurants[random.randint(0,len(restaurants)-1)]


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


# TODO Implment the keyword matching algorithm
def keyword_matching(utterance):
    keywords = {
        'area': None,
        'pricerange': None,
        'food': None
    }

    return keywords


def check_misspelling_update_preferences(cur_utterance):
    # We need some logic with the keyword_matching (maybe not all utterances need to run this function)
    preferences = {}  # save preferences in the dictonary

    keywords = keyword_matching(cur_utterance)

    # We check for all the keywords if we have a big mispelling
    # If yes we will raise the misspelling flag
    for keyword_type, keyword in keywords:
        if keyword is not None:
            preferences, misspelling = levenshtein_distance(keyword, keyword_type, domain_terms_dict, preferences)
            if len(misspelling) > 0:
                return misspelling
    return preferences


def levenshtein_distance(keyword, keyword_type, domain_terms_dict, preferences):
    min_distance = 4
    closest_terms = []

    for term in domain_terms_dict[keyword_type]:
        distance = Levenshtein.distance(keyword, term)
        if distance < min_distance:
            min_distance = distance
            closest_terms = [term]
        elif distance == min_distance:
            closest_terms.append(term)

    if min_distance <= 3:
        preferences[0][keyword_type] = random.choice(closest_terms)
        return preferences, ''
    else:
        print("Big mispelling, need an according error message")
        return preferences, keyword


def update_preferences(preferences, current_state):
    match current_state:
        case 1, 2, 3, 4, 5:
            for key in preferences.keys:
                if preferenceField[key] is None:
                    preferenceField[key] = preferences[key]
        case 6, 7:
            for key in preferences.keys:
                preferenceField[key] = preferences[key]


#function to handle system outputs
#@paramters
#@current_state: int, the current state of the system
#@misspelling: str, the misspelled word, only needed if @current_state is 2 or 7
#@restaurant: darray, the restaurant suggested by the system, only needed if @current_state = 9 or 10
#@detail: string, the requested detail of the restaurant, only needed if @current_state = 10, can be either "phone","addr","postcode"
def print_system_message(current_state, misspelling='', restaurant=None, detail=None):
    match current_state:
        case 1:
            print("Error detected, System in state 1 [Greeting]")
        case 2:
            print(f"Could not recognize word '{misspelling}', please rephrase your input!")
        case 3:
            print("What part of town do you have in mind?")
        case 4:
            print("Would you like something in the cheap , moderate , or expensive price range?")
        case 5:
            print("What kind of food would you like?")
        case 6:
            area = f"in the {preferenceField['area']} part of the town" if preferenceField['area'] is not None else ""
            pricerange = preferenceField['pricerange'] if preferenceField['pricerange'] is not None else ""
            food = f" serving {preferenceField['food']} food" if preferenceField['food'] is not None else ""
            print(f"Sorry, but there is no {pricerange} restaurant{area}{food}.")
        case 7:
            print(f"Could not recognize word '{misspelling}', please rephrase your input!")
        case 8:
            print("Please confirm that you want to leave")
        case 9:
            name = restaurant[0]
            area = f"in the {restaurant[2]} part of the town" if restaurant[2] != '' else ""
            pricerange = restaurant[1] if restaurant[1] != '' else ""
            food = f" serving {restaurant[3]} food" if restaurant[3] != '' else ""
            print(f"{name} is a nice {pricerange} restaurant{area}{food}")
        case 10:
            phone = ''
            addr = '',
            postcode = ''
            if 'phone' in detail:
                phone = f', the phone number is {restaurant[4]}'
            elif 'addr' in detail:
                addr = f', the address is {restaurant[5]}'
            elif 'postcode' in detail:
                postcode = f', the post code is {restaurant[6]}'
            else:
                print(f"Sorry, request information is unknown")
                return
            print(f'Sure{phone}{addr}{postcode}')
        case 11:
            print('Goodbye')

    return None


def main():
    print("Dialog management system")

    # possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    current_state = 1
    next_state = 1

    vectorizer, clf = train_ml_model()
    candidate_restaurants = []
    suggested_restaurants = []
    current_restaurant = None

    while True:

        if next_state == 11:
            print("System outputs Goodbye")
            break

        current_state = next_state

        predicted_label, utterance = prompt_input(vectorizer, clf)
        print(predicted_label, " | ", utterance)

        next_state = state_transition_function(current_state, predicted_label, utterance)
        if next_state == 2 or next_state == 7: #this case is handled inside of state_transition_function
            continue

        #if we want to suggest a restaurant, we have to find one
        restaurant = None
        if next_state == 9:
            #if candidate restaurants not computed yet, find them now
            if len(candidate_restaurants) == 0:
                candidate_restaurants = findRestaurants(preferenceField['area'],preferenceField['pricerange'],preferenceField['food'])
            #if not all candidate_restaurants were suggested, choose new restaurant to suggest
            if len(candidate_restaurants) > len(suggested_restaurants):
                current_restaurant = chooseRestaurant(candidate_restaurants,suggested_restaurants)
                suggested_restaurants.append(current_restaurant)
        detail = ''
        if next_state == 10:
            #check what detail is requested
            if 'phone' in utterance or 'number' in utterance:
                detail = 'phone'
            if 'address' in utterance or 'where' in utterance:
                detail += 'addr'
            if 'post' in utterance or 'code' in utterance:
                detail += 'postcode'
            if detail == '':
                detail = 'unknown'

        print_system_message(next_state,restaurant=current_restaurant, detail=detail)


if __name__ == "__main__":
    main()
