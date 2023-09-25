import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import Levenshtein
import random
import string
import re

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

# MAKE SURE TO MAKE THE CHATBOT WORK WITH DONTCARES AS WELL (REPRESENTED BY X IN FIND_RESTAURANTS FUNCTION)

# Fields that need to be filled:
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

# Possibly more food types, need to check
domain_terms_dict = {
    'pricerange': ['cheap', 'moderate', 'expensive'],
    'area': ['north', 'south', 'east', 'west', 'centre'],
    'food': ['world', 'african', 'asian oriental', 'australasian', 'bistro', 'british',
             'catalan', 'chinese', 'cuban', 'european', 'french', 'fusion',
             'gastropub', 'indian', 'international', 'italian', 'jamaican',
             'japanese', 'korean', 'lebanese', 'mediterranean',
             'modern european', 'moroccan', 'north american', 'persian',
             'polynesian', 'portuguese', 'romanian', 'seafood', 'spanish',
             'steakhouse', 'swiss', 'thai', 'traditional', 'turkish', 'tuscan',
             'vietnamese', 'swedish', 'russian', 'welsh', 'austrian', 'belgian', 'brazilian']
}


# function to update the additional preferences
# @cur_utterance: string, the current utterance provided by the user
def update_opt_requirements(cur_utterance):
    optionalPreferences['touristic'] = True if 'touristic' in cur_utterance else False
    optionalPreferences['assigned_seats'] = True if 'seats' in cur_utterance else False
    optionalPreferences['children'] = True if 'child' in cur_utterance else False
    optionalPreferences['romantic'] = True if 'romantic' in cur_utterance else False


# State transistion function to change the state
# @cur_state: int, the current state
# @cur_dialog_act: string, the predicted dialog act for the current utterance
# @cur_utterance: string, the current utterance provided by the user
def state_transition_function(cur_state, cur_dialog_act, cur_utterance):
    
    # Check the current state
    match cur_state:

        case 1 | 2 | 3 | 4 | 5:
            
            # THIS WAS UNCOMMENTED. NEED CHECK THAT IT STILL WORKS CORRECTLY
            # Moved it so it catches reqalts as well with a preference

            if cur_dialog_act != 'inform' and cur_dialog_act != 'reqalts':
                 return checkPreferences()

            # First thing to do is to check whether there was a misspelling
            preferences_or_misspelling = check_misspelling_or_preferences(cur_utterance, cur_state)

            if type(preferences_or_misspelling) == str:
                # Print error message here because misspelled word is known
                print_system_message(2, misspelling=preferences_or_misspelling)
                return 2

            update_preferences(preferences_or_misspelling, current_state=cur_state)
            return checkPreferences()
        
        case 6 | 7:
            if cur_dialog_act == 'bye' or cur_dialog_act == 'thankyou':
                return 8
            if cur_dialog_act != 'inform':
                # checkPreferences() will always return 6 so we just return 6
                return 6  
            
            # First thing to do is to check whether there was a misspelling
            preferences_or_misspelling = check_misspelling_or_preferences(cur_utterance, cur_state)
            
            if type(preferences_or_misspelling) == str:
                # Print error message here because misspelled word is known
                print_system_message(7, misspelling=preferences_or_misspelling)
                return 7

            update_preferences(preferences_or_misspelling, current_state=cur_state)
            return checkPreferences()
        
        case 8:
            # Check if user really wants to leave
            if cur_dialog_act in ['bye', 'thankyou', 'ack', 'confirm','affirm']:
                return 11
            return 6

        case 9:
            update_opt_requirements(cur_utterance)
            restaurants = find_restaurants()
            restaurants = filter_restaurants_opt_requirements(restaurants)
            if len(restaurants) > 0:
                return 10
            return 9
        
        case 10:
            # Check if user wants a restaurant suggestion
            if cur_dialog_act == 'request':
                return 11
            if cur_dialog_act in ['bye', 'thankyou']:
                return 12
            return 10
        
        case 11:
            # Check if user doesn't want any other restaurant detail
            if cur_dialog_act in ['bye', 'thankyou']:
                return 12
            return 11
        
        case 12:
            return -1

# Function to perform checks on the presence of the preferences
# For the transition function (see model diagram -> long column of diamonds)
def checkPreferences():
    # If there is no restraunt matching the preferred attributes transite to state 6
    if len(find_restaurants(preferenceField['area'], preferenceField['pricerange'], preferenceField['food'])) == 0:
        return 6
    
    # If the preferred area is unknown move to transite 3
    if preferenceField['area'] == None:
        return 3
    
    # If the preferred pricerange is unknown move to transite 4
    if preferenceField['pricerange'] == None:
        return 4
    
    # If preferred Foodtype is unkown transite to state 5
    if preferenceField['food'] == None:
        return 5
    
    # Transite to state 10 to suggest a restaurant
    return 9

# Lookup function to find restaurants fitting the criteria in the .csv file
# Attributes should be provided as string
# @return is a numpy array
def find_restaurants(area='X', price='X', food='X', path='restaurant_info_extended.csv'):
    restaurants = pd.read_csv('restaurant_info_extended.csv',sep=';')

    if preferenceField['area'] != 'X' and preferenceField['area'] is not None:
        restaurants = restaurants[restaurants['area'] == preferenceField['area']]
    if preferenceField['pricerange'] != 'X' and preferenceField['pricerange'] is not None:
        restaurants = restaurants[restaurants['pricerange'] == preferenceField['pricerange']]
    if preferenceField['food'] != 'X' and preferenceField['food'] is not None:
        restaurants = restaurants[restaurants['food'] == preferenceField['food']]

    return restaurants.values

# Lookup function to filter restaurants fitting the additional requirements
# restaurants is a numpy array, requirements should be provided as boolean
# @return is a numpy array
def filter_restaurants_opt_requirements(restaurants, touristic=False, assigned_seats=False, children=False, romantic=False):

    if optionalPreferences['touristic']:
        restaurants = restaurants[restaurants[:,1] == 'cheap']
        restaurants = restaurants[restaurants[:,7] == 'good']
        restaurants = restaurants[restaurants[:,3] != 'romanian']
    if optionalPreferences['assigned_seats']:
        restaurants = restaurants[restaurants[:,8] == 'busy']
    if optionalPreferences['children']:
        restaurants = restaurants[restaurants[:,9] != 'long']
    if optionalPreferences['romantic']:
        restaurants = restaurants[restaurants[:,8] != 'busy']
        restaurants = restaurants[restaurants[:,9] == 'long']

    return restaurants

# Function to randomly choose a restaurant
# @parameters:
# @restaurants: numpy array that contains restaurants from which one is choosen randomly
# @alreadyUsedRestaurants: numpy array, sublist of restaurant with previously suggest restaurants
# @return: a numpy array containing a restaurant from @restaurants | template: [name,area,pricerange,food,phone,addr,postcode]
def choose_restaurant(restaurants, already_used_restaurants):
    if len(restaurants) == 0:
        return None

    restaurants = [i for i in restaurants if i[0] not in already_used_restaurants[:,0]]

    return restaurants[random.randint(0, len(restaurants)-1)]

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

def keyword_matching(utterance,cur_state):
    dont_cares = ['any',"don't care","i don't care","i do not care","i don't mind","i do not mind"]



    keywords = {
        'area': None,
        'pricerange': None,
        'food': None
    }

    if utterance in dont_cares:
        print('inside')
        if cur_state == 3:
            keywords['area'] = 'X'
            return keywords
        if cur_state == 4:
            keywords['pricerange'] = 'X'
            return keywords
        if cur_state == 5:
            keywords['food'] = 'X'
            return keywords

    regex_patterns = {
    'food': re.compile(r'\b(\w+)\s+(food|restaurant|place|restaurantin)\b'),
    'pricerange': re.compile(r'\b(\w+)\s+(priced|price)\b'),
    'area': re.compile(r'\b(\w+)\s+(part|area)|in\s+the\s+(\w+)\b'),
    }

    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    clean_utterance = utterance.translate(translator)
    
    # First we check for exact matches
    tokens = clean_utterance.split(' ')
    for token in tokens:
        for pref_type, pref in domain_terms_dict.items():
            for pref_term in pref:
                if token==pref_term:
                    if token == 'any':
                        token = 'X'
                    keywords[pref_type] = token

    # User only provided one word
    if len(tokens) == 1:
        token = tokens[0]
        closest_term, pref_type = levenshtein_distance_single(token)
        if closest_term: #todo remove this if? closest_term is not a boolean so what happens here?
            if closest_term == 'any':
                closest_term = 'X'
            keywords[pref_type] = closest_term

    else:

        ignore_words = {'a', 'the', 'in', 'cheap', 'hi', 'hey'}
        # Check for regex patterns
        for pref_type, pattern in regex_patterns.items():
            # We only check for the preferences that we didn't find an exact match
            if keywords[pref_type] is None:
                match = pattern.search(clean_utterance)
                if match:
                    # Go through all the subgroups of the regex
                    first_group = match.group(1)
                    second_group = match.group(2)
                    third_group = None
                    # Third group is to catch cases of "word + in the + area"
                    if len(match.groups()) == 3:
                        third_group = match.group(3)
                    if first_group == "any":
                        if second_group == "food" or second_group == "restaurant" or second_group == "place" or second_group == "restaurantin":
                            keywords['food'] = "X"
                        elif second_group == "priced" or second_group == "price":
                            keywords['pricerange'] = "X"
                        elif second_group == "part" or second_group == "area":
                            keywords['area'] = "X"
                    elif third_group:
                        closest_term = levenshtein_distance_regex(third_group, pref_type)
                        # If we found a close term we save it as a keyword
                        if closest_term:
                            keywords[pref_type] = closest_term
                    elif first_group and first_group not in ignore_words:
                        closest_term = levenshtein_distance_regex(first_group, pref_type)
                        # If we found a close term we save it as a keyword
                        if closest_term:
                            keywords[pref_type] = closest_term

    return keywords

def check_misspelling_or_preferences(cur_utterance, cur_state):
    # We need some logic with the keyword_matching (maybe not all utterances need to run this function)
    preferences = {
        'area': None,
        'pricerange': None,
        'food': None
    }  # save preferences in the dictonary

    keywords = keyword_matching(cur_utterance, cur_state)

    print('Keywords are: ', keywords)

    # We check for all the keywords if we have a big mispelling
    # If yes we will raise the misspelling flag
    for keyword_type, keyword in keywords.items():

        if keyword is not None and keyword != 'X':
            preferences, misspelling = levenshtein_distance(keyword, keyword_type, preferences)
            if len(misspelling) > 0:
                return misspelling
        
        if keyword == 'X':
            preferences[keyword_type] = 'X'

    return preferences

def levenshtein_distance_single(keyword):
    min_distance = 4
    closest_terms = []
    # Since it's a single word we need to check for all keyword types
    for keyword_type in domain_terms_dict.keys():
        for term in domain_terms_dict[keyword_type]:
            distance = Levenshtein.distance(keyword, term)
            # if keyword_type == 'food':
            #     print(term, ' ', distance)
            if distance < min_distance:
                min_distance = distance
                closest_terms = [term]
            elif distance == min_distance:
                closest_terms.append(term)
    
        # print(closest_terms, min_distance)
        if min_distance <= 3:
            return random.choice(closest_terms), keyword_type
        
    return None, None

def levenshtein_distance_regex(keyword, keyword_type):
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
        return random.choice(closest_terms)
    else:
        return None

def levenshtein_distance(keyword, keyword_type, preferences):
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
        preferences[keyword_type] = random.choice(closest_terms)
        return preferences, ''
    else:
        print("Big mispelling, need an according error message")
        return preferences, keyword

def update_preferences(preferences, current_state):
    match current_state:
        case 1| 2| 3| 4| 5:
            for key in preferences.keys():
                if preferenceField[key] is None:
                    preferenceField[key] = preferences[key]
        case 6| 7:
            for key in preferences.keys():
                if preferences[key] is not None:
                    preferenceField[key] = preferences[key]

# Function to handle system outputs
# @paramters
# @current_state: int, the current state of the system
# @misspelling: str, the misspelled word, only needed if @current_state is 2 or 7
# @restaurant: darray, the restaurant suggested by the system, only needed if @current_state = 9 or 10
# @detail: string, the requested detail of the restaurant, only needed if @current_state = 10, can be either "phone","addr","postcode"
def print_system_message(current_state, misspelling='', restaurant=None, detail=None):
    print(current_state)
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
            area = f" in the {preferenceField['area']} part of the town" if preferenceField['area'] is not None else ""
            pricerange = preferenceField['pricerange'] if preferenceField['pricerange'] is not None else ""
            food = f" serving {preferenceField['food']} food" if preferenceField['food'] is not None else ""
            print(f"Sorry, but there is no {pricerange} restaurant{area}{food}.")

        case 7:
            print(f"Could not recognize word '{misspelling}', please rephrase your input!")

        case 8:
            print("Please confirm that you want to leave")

        case 9:
            if optionalPreferences['touristic'] is None:
                print('Do you have additional requirements (should the restaurant be touristic, suitable for '
                      'children, romantic or have assigned seats)')
            else:
                print('Sorry, there is no restaurant fullfilling your additional preferences. Please choose other '
                      'additional requirements')

        case 10:
            name = restaurant[0]
            area = f" in the {restaurant[2]} part of the town" if restaurant[2] != '' else ""
            pricerange = restaurant[1] if restaurant[1] != '' else ""
            food = f" serving {restaurant[3]} food" if restaurant[3] != '' else ""
            print(f"{name} is a nice {pricerange} restaurant{area}{food}")
            print(optionalPreferences)
            if optionalPreferences['touristic'] == True:
                print(f"{name} is touristic because it has good and popular food.")
            if optionalPreferences['assigned_seats'] == True:
                print(f"{name} is a busy restaurant, therefore the waiter will decide where you sit.")
            if optionalPreferences['children'] == True:
                print(f"At {name} people do not tend to stay there for a long time. Therefore your children will not get bored waiting for the food.")
            if optionalPreferences['romantic'] == True:
                print(f"You can have a lovely date for a longer time at {name}")

        case 11:
            phone = ''
            addr = ''
            postcode = ''
            if 'phone' in detail:
                phone = f', the phone number is {restaurant[4]}'
            if 'addr' in detail:
                addr = f', the address is {restaurant[5]}'
            if 'postcode' in detail:
                postcode = f', the post code is {restaurant[6]}'
            if detail == 'unknown':
                print(f"Please specify whether you want the phone number, the address, or the postcode")
                return
            print(f'Sure{phone}{addr}{postcode}')

        case 12:
            print('Goodbye. Have a nice day!')

    return None

def main():
    print("Hello, welcome to the Group 30 restaurant recommendation system. You can ask for restaurants by area, price range or food type. How may I help you?")

    # possible_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    current_state = 1
    next_state = 1

    vectorizer, clf = train_ml_model()
    
    # Restaurants that fit the preferences
    candidate_restaurants = []
    # Restaurants that were suggested so far
    suggested_restaurants = [[None]]
    current_restaurant = None

    while True:

        if next_state == 12:
            break
    
        current_state = next_state
        print("Current state: ", current_state)

        predicted_label, utterance = prompt_input(vectorizer, clf)

        next_state = state_transition_function(current_state, predicted_label, utterance)
        
        print(predicted_label, " | ", utterance, '(', preferenceField['area'], ' ',preferenceField['pricerange'], ' ',preferenceField['food'], ')')

        if next_state == 2 or next_state == 7: #this case is handled inside of state_transition_function
            continue
        # If we want to suggest a restaurant, we have to find one
        if next_state == 9:
            # If candidate restaurants not computed yet, find them now
            if len(candidate_restaurants) == 0:
                candidate_restaurants = find_restaurants(preferenceField['area'], preferenceField['pricerange'], preferenceField['food'])
                candidate_restaurants = filter_restaurants_opt_requirements(candidate_restaurants)
            # If not all candidate_restaurants were suggested, choose new restaurant to suggest
            if len(candidate_restaurants) > len(suggested_restaurants) or suggested_restaurants[0][0] is None:
                current_restaurant = choose_restaurant(candidate_restaurants, np.array(suggested_restaurants))
                if suggested_restaurants[0][0] is None:
                    suggested_restaurants = [current_restaurant]
                else:
                    suggested_restaurants.append(current_restaurant)

        detail = ''
        if next_state == 10:
            # Check what detail is requested
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
