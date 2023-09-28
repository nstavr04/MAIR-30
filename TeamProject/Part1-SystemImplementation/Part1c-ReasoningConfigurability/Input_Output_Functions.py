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
        case 1:
            if dialog_restart_on:
                out =  "Hello, welcome to the Group 30 restaurant recommendation system. You can ask for restaurants by area, price range or food type. How may I help you? \nYou can restart the conversation at any point by typing 'restart conversation'"
            else:
                out =  "Hello, welcome to the Group 30 restaurant recommendation system. You can ask for restaurants by area, price range or food type. How may I help you?"
        case 2:
            out = f"Could not recognize word '{misspelling}', please rephrase your input!"

        case 3:
            out = "What part of town do you have in mind?"

        case 4:
            out = "Would you like something in the cheap , moderate , or expensive price range?"

        case 5:
            out = "What kind of food would you like?"

        case 6:
            area = f" in the {preferenceField['area']} part of the town" if preferenceField['area'] is not None else ""
            pricerange = preferenceField['pricerange'] if preferenceField['pricerange'] is not None else ""
            food = f" serving {preferenceField['food']} food" if preferenceField['food'] is not None else ""
            out = f"Sorry, but there is no {pricerange} restaurant{area}{food}."

        case 7:
            out = f"Could not recognize word '{misspelling}', please rephrase your input!"

        case 8:
            out = "Please confirm that you want to leave"

        case 9:
            if optionalPreferences['touristic'] is None:
                out = 'Do you have additional requirements (should the restaurant be touristic, suitable for ' \
                       'children, romantic or have assigned seats) '
            else:
                out = 'Sorry, there is no restaurant fullfilling your additional preferences. Please choose other ' \
                      'additional requirements'

        case 10:
            name = restaurant[0]
            area = f" in the {restaurant[2]} part of the town" if restaurant[2] != '' else ""
            pricerange = restaurant[1] if restaurant[1] != '' else ""
            food = f" serving {restaurant[3]} food" if restaurant[3] != '' else ""
            out = f"{name} is a nice {pricerange} restaurant{area}{food}"
            #print(optionalPreferences)
            if optionalPreferences['touristic'] == True:
                out += f"\n {name} is touristic because it has good and popular food."
            if optionalPreferences['assigned_seats'] == True:
                out += f"\n {name} is a busy restaurant, therefore the waiter will decide where you sit."
            if optionalPreferences['children'] == True:
                out += f"\nAt {name} people do not tend to stay there for a long time. Therefore your children will not get bored waiting for the food."
            if optionalPreferences['romantic'] == True:
                out += f"\nYou can have a lovely date for a longer time at {name}"
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
            out = f'Sure{phone}{addr}{postcode}'
            if detail == 'unknown':
                out = f"Please specify whether you want the phone number, the address, or the postcode"


        case 12:
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