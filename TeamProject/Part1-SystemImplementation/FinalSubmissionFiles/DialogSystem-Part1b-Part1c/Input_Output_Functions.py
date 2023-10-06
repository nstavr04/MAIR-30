# This file contains functions to handle Speech-Input and the outputs of the system in the CLI

import ASR_userUtterance
import json
import time

# Load our configurations
with open("configurations.json", "r") as f:
    configurations = json.load(f)

dialog_restart_on = configurations['dialog_restart_on']
ASR_on = configurations['ASR_on']
caps_on = configurations['caps_on']
delay_on = configurations['delay_on']

# Function to handle system outputs
# @paramters
# @current_state: int, the current state of the system
# @misspelling: str, the misspelled word, only needed if @current_state is 2 or 7
# @restaurant: darray, the restaurant suggested by the system, only needed if @current_state = 9 or 10
# @detail: string, the requested detail of the restaurant, only needed if @current_state = 10, can be either "phone","addr","postcode"
def print_system_message(current_state,preferenceField,optionalPreferences, misspelling='', restaurant=None, detail=None):
    if delay_on:
        time.sleep(1)
    out = ""
    match current_state:
        case '1_Welcome':
            restart =  "\nYou can restart the conversation at any point by typing 'restart conversation'" if dialog_restart_on else ""
            out =  f"Hello, welcome to the Group 30 restaurant recommendation system. You can ask for restaurants by area, price range or food type. How may I help you?{restart}"
        case '2_AskCorrection':
            out = f"Could not recognize word '{misspelling}', please rephrase your input!"

        case '3_AskArea':
            out = "What part of town do you have in mind?"

        case '4_AskPriceRange':
            out = "Would you like something in the cheap, moderate, or expensive price range?"

        case '5_AskFoodType':
            out = "What kind of food would you like?"

        case '6_NoRestaurantExists':
            area = (
            f" in {preferenceField['area']} part of the town" if preferenceField['area'] not in [None, 'dontcare']
            else " in any part of the town" if preferenceField['area'] == 'dontcare'
            else ""
            )
            pricerange = (
            preferenceField['pricerange'] if preferenceField['pricerange'] not in [None, 'dontcare'] 
            else 'any priced range' if preferenceField['pricerange'] == 'dontcare' 
            else ""
            )
            food = (
            f" serving {preferenceField['food']} food" if preferenceField['food'] not in [None, 'dontcare']
            else " serving any food" if preferenceField['food'] == 'dontcare'
            else ""
            )
            out = f"Sorry, but there is no {pricerange} restaurant{area}{food}."

        case '7_AskCorrection':
            out = f"Could not recognize word '{misspelling}', please rephrase your input!"

        case '8_ConfirmExit':
            out = "Please confirm that you want to leave"

        case '9_AskAdditionalRequirements':
            if optionalPreferences['touristic'] is None:
                out = 'Do you have additional requirements (should the restaurant be touristic, suitable for ' \
                       'children, romantic or have assigned seats) '
            else:
                out = 'Sorry, there is no restaurant fulfilling your additional preferences. Please choose other ' \
                      'additional requirements:'

        case '10_SuggestRestaurants':
            name = restaurant[0]
            area = f" in the {restaurant[2]} part of the town" if restaurant[2] != '' else ""
            pricerange = restaurant[1] if restaurant[1] != '' else ""
            food = f" serving {restaurant[3]} food" if restaurant[3] != '' else ""
            out = f"{name} is a nice {pricerange} restaurant{area}{food}."
            touristic = ""
            assigned_seats = ""
            children = ""
            romantic = ""
            if optionalPreferences['touristic'] == True:
                touristic = f"\n{name} is touristic because it has good and popular food."
            if optionalPreferences['assigned_seats'] == True:
                assigned_seats += f"\n{name} is a busy restaurant, therefore the waiter will decide where you sit."
            if optionalPreferences['children'] == True:
                children += f"\nAt {name} people do not tend to stay there for a long time. Therefore your children will not get bored waiting for the food."
            if optionalPreferences['romantic'] == True:
                romantic += f"\nYou can have a lovely date for a longer time at {name}."
            out = f"{name} is a nice {pricerange} restaurant{area}{food}.{touristic}{assigned_seats}{children}{romantic}"
        case '11_ProvideRestaurantDetails':
            phone = ''
            addr = ''
            postcode = ''
            if 'phone' in detail:
                phone = f', the phone number is {restaurant[4]}'
            if 'addr' in detail:
                addr = f', the address is {restaurant[5]}'
            if 'postcode' in detail:
                postcode = f', the post code is {restaurant[6]}'
            out = f'Sure{phone}{addr}{postcode}.'
            if detail == 'unknown':
                out = f"Please specify whether you want the phone number, the address, or the postcode."

        case '12_Goodbye':
            out = 'Goodbye. Have a nice day!'

    if caps_on:
        print(out.upper())
    else:
        print(out)

# Handles the input from the user and classifies it into a dialog_act
def prompt_input(vectorizer, clf):

    utterance = None

    while utterance is None:
        if ASR_on:
            filename = 'recorded_audio.wav'
            print("Speak to the microphone...")
            ASR_userUtterance.record_audio('recorded_audio.wav')
            ASR_userUtterance.preprocess_audio('recorded_audio.wav')
            try:
                utterance = ASR_userUtterance.recognize_speech('recorded_audio.wav').lower()
            except Exception as e:  # This will catch any type of exception, you can specify the type if you want
                print(f"An error occurred: {e}")
                print("Could not recognize the speech. Please try again.")

        else:
            utterance = input("Please enter utterance: ").lower()

    utterance_bow = vectorizer.transform([utterance])
    predicted_label = clf.predict(utterance_bow)[0]

    return predicted_label, utterance

# Function to set the configuration from the CLI
def set_configurations():
    with open("configurations.json", "r") as f:
        configurations = json.load(f)

    dialog_restart_on = configurations['dialog_restart_on']
    ASR_on = configurations['ASR_on']
    caps_on = configurations['caps_on']
    levenshtein_dis = configurations['levenshtein_dis']
    delay_on = configurations['delay_on']
    random_preference_order_on = configurations["random_preference_order_on"]

    print('Before starting the dialog system you have the option to change the current configuration settings')
    print('The following options are implemented and can be activated or used:')
    print(f'Restart the dialog system at any time, currently active: {dialog_restart_on}')
    print(f'Speech recognition, currently active: {ASR_on}')
    print(f'System outputs in Caps, currently active: {caps_on}')
    print(f'Set levenshtein_dis for automatic misspelling correction, current distance: {levenshtein_dis}')
    print(f'System outputs with delay, currently active: {delay_on}')
    print(f'Preferences can be stated in random order, currently active: {random_preference_order_on}')
    print(f'Debbuging mode, currently active: {configurations["debugging_on"]}')
    time.sleep(1)
    print()
    print("==========IMPORTANT==========")
    print("If you want to change the current configurations, type 'change', if you want to keep the current options, type 'continue'")
    print("WARNING: if you change the current configurations, the application will be closed and has to be restarted!")
    print()
    utterance = None
    while utterance is None:
        utterance = input("Please enter your choice: ").lower()
        if utterance == 'continue':
            print("\n Dialog System is starting... \n")
            return False
        elif utterance == 'change':
            utterance = None
            while utterance is None:
                utterance = input("Type 'yes' to activate restart option and 'no' to deactivate: ").lower()
                if utterance == 'exit':
                    return False
                elif utterance == 'yes':
                    configurations['dialog_restart_on'] = True
                elif utterance == 'no':
                    configurations['dialog_restart_on'] = False
                else:
                    utterance = None
            utterance = None
            while utterance is None:
                    utterance = input("Type 'yes' to activate speech recogonition and 'no' to deactivate: ").lower()
                    if utterance == 'exit':
                        return False
                    elif utterance == 'yes':
                        configurations['ASR_on'] = True
                    elif utterance == 'no':
                        configurations['ASR_on'] = False
                    else:
                        utterance = None
            utterance = None
            while utterance is None:
                    utterance = input("Type 'yes' to put dialog systems utterances in caps, otherwise type 'no': ").lower()
                    if utterance == 'exit':
                        return False
                    elif utterance == 'yes':
                        configurations['caps_on'] = True
                    elif utterance == 'no':
                        configurations['caps_on'] = False
                    else:
                        utterance = None
            utterance = None
            while utterance is None:
                    utterance = input("Please provide an integer to set the levenshtein distance: ").lower()
                    try:
                        dist =  int(utterance)
                        configurations['levenshtein_dis'] = dist
                    except:
                        if utterance == 'exit':
                            return False
                        else:
                            utterance = None
            utterance = None

            while utterance is None:
                    utterance = input("Type 'yes' to activate a delay before dialog system outputs and 'no' to deactivate: ").lower()
                    if utterance == 'exit':
                        return False
                    elif utterance == 'yes':
                        configurations['delay_on'] = True
                    elif utterance == 'no':
                        configurations['delay_on'] = False
                    else:
                        utterance = None
            utterance = None
            while utterance is None:
                    utterance = input("Type 'yes' to allow stating preferences in random order and 'no' to not allow: ").lower()
                    if utterance == 'exit':
                        return False
                    elif utterance == 'yes':
                        configurations['random_preference_order_on'] = True
                    elif utterance == 'no':
                        configurations['random_preference_order_on'] = False
                    else:
                        utterance = None
            utterance = None
            while utterance is None:
                    utterance = input("Type 'yes' to allow debugging mode and 'no' to not allow: ").lower()
                    if utterance == 'exit':
                        return False
                    elif utterance == 'yes':
                        configurations['debugging_on'] = True
                    elif utterance == 'no':
                        configurations['debugging_on'] = False
                    else:
                        utterance = None

            with open('configurations.json', 'w') as f:
                    json.dump(configurations, f)

            print('Changes were successfully saved!')
            print('The application will now terminate, please restart the application for the changed preferences.')
            return True
        else:
            utterance = None




