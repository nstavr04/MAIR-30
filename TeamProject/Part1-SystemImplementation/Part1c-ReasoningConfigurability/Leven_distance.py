# This document contains functions to compute the Levenshtein distance to possibly misspelled words.

import random
import Levenshtein

# Implementation of variations of levenstein distance used for the keyword mathing algorithm

def levenshtein_distance_single(keyword, domain_terms_dict, levenshtein_dis):
    min_distance = levenshtein_dis + 1
    closest_terms = []
    # Since it's a single word we need to check for all keyword types
    for keyword_type in domain_terms_dict.keys():
        for term in domain_terms_dict[keyword_type]:
            distance = Levenshtein.distance(keyword, term)
            if distance < min_distance:
                min_distance = distance
                closest_terms = [term]
            elif distance == min_distance:
                closest_terms.append(term)
    
        if min_distance <= levenshtein_dis:
            return random.choice(closest_terms), keyword_type
        
    return keyword, 'area'

def levenshtein_distance_regex(keyword, keyword_type, domain_terms_dict, levenshtein_dis):
    min_distance = levenshtein_dis + 1
    closest_terms = []
    for term in domain_terms_dict[keyword_type]:
        distance = Levenshtein.distance(keyword, term)
        if distance < min_distance:
            min_distance = distance
            closest_terms = [term]
        elif distance == min_distance:
            closest_terms.append(term)
    if min_distance <= levenshtein_dis:
        return random.choice(closest_terms)
    else:
        return keyword

def levenshtein_distance(keyword, keyword_type, preferences, domain_terms_dict, levenshtein_dis):
    min_distance = levenshtein_dis + 1
    closest_terms = []
    for term in domain_terms_dict[keyword_type]:
        distance = Levenshtein.distance(keyword, term)
        if distance < min_distance:
            min_distance = distance
            closest_terms = [term]
        elif distance == min_distance:
            closest_terms.append(term)

    if min_distance <= levenshtein_dis:
        preferences[keyword_type] = random.choice(closest_terms)
        return preferences, ''
    else:
        return preferences, keyword