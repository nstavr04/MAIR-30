
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