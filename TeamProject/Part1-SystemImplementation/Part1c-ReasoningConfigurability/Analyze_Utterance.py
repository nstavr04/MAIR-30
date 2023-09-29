#This document contains additional functions to analyze an utterance for preferences, misspellings, or other information.


import re
import string
import Leven_distance
import json

# All the possible values for the preferences
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

#Load Levenshtein distance configuration
# Load our configurations
with open("configurations.json", "r") as f:
    configurations = json.load(f)

levenshtein_dis = configurations['levenshtein_dis']

# We find the keywords from the utterance that are relevant for the preferences
def keyword_matching(utterance, cur_state):
    # Words that will be identified as 'dontcare'
    dont_cares = ['any', "don't care", "i don't care", "i do not care", "i don't mind", "i do not mind", "anything"]

    keywords = {
        'area': None,
        'pricerange': None,
        'food': None
    }

    # We mark our dontcares with an X
    if utterance in dont_cares:
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
        'food': re.compile(r'\b(\w+)\s+(food|restaurant|place|restaurantin|type)\b'),
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
                if token == pref_term:
                    keywords[pref_type] = token

    # List of words to ignore that have a <=3 levenshtein distance to a preference
    ignore_words = {'a', 'the', 'in', 'cheap', 'hi', 'hey', 'play', 'thanks'}

    # User only provided one word
    if len(tokens) == 1:
        token = tokens[0]
        if token not in ignore_words:
            closest_term, pref_type = Leven_distance.levenshtein_distance_single(token, domain_terms_dict,
                                                                                 levenshtein_dis)
            # They if basically means that closest_term is not None
            if closest_term:
                if closest_term == 'any':
                    closest_term = 'X'
                keywords[pref_type] = closest_term

    else:
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
                        if second_group in {"food", "restaurant", "place", "restaurantin", "type"}:
                            keywords['food'] = "X"
                        elif second_group in {"priced", "price"}:
                            keywords['pricerange'] = "X"
                        elif second_group in {"part", "area"}:
                            keywords['area'] = "X"

                    elif third_group:
                        closest_term = Leven_distance.levenshtein_distance_regex(third_group, pref_type,
                                                                                 domain_terms_dict, levenshtein_dis)
                        # If we found a close term we save it as a keyword
                        if closest_term:
                            keywords[pref_type] = closest_term
                    elif first_group and first_group not in ignore_words:
                        closest_term = Leven_distance.levenshtein_distance_regex(first_group, pref_type,
                                                                                 domain_terms_dict, levenshtein_dis)
                        # If we found a close term we save it as a keyword
                        if closest_term:
                            keywords[pref_type] = closest_term

    return keywords


# The function returns either a dictionary with the preferences or a string with the misspelled word
def check_misspelling_or_preferences(cur_utterance, cur_state):
    # Used as a temporary storage for the preferences before assigning them to the preferenceField
    preferences = {
        'area': None,
        'pricerange': None,
        'food': None
    }

    keywords = keyword_matching(cur_utterance, cur_state)

    # We check for all the keywords if we have a big mispelling
    # If yes we will raise the misspelling flag
    for keyword_type, keyword in keywords.items():

        if keyword is not None and keyword != 'X':
            preferences, misspelling = Leven_distance.levenshtein_distance(keyword, keyword_type, preferences,
                                                                           domain_terms_dict, levenshtein_dis)
            if len(misspelling) > 0:
                return misspelling

        if keyword == 'X':
            preferences[keyword_type] = 'X'

    return preferences


def identify_details(current_state, utterance):
    detail = ''
    if current_state == 11:
        # Check what detail is requested
        if 'phone' in utterance or 'number' in utterance:
            detail = 'phone'
        if 'address' in utterance or 'where' in utterance:
            detail += 'addr'
        if 'post' in utterance or 'code' in utterance:
            detail += 'postcode'
        if detail == '':
            detail = 'unknown'

    return detail