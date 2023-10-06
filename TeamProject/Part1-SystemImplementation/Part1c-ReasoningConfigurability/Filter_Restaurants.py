# This document contains additional functions to find suitable restaurants fitting the preferences of the user

import pandas as pd
import random


# Lookup function to find restaurants fitting the criteria in the .csv file
# Attributes should be provided as string
# @return is a numpy array
def find_restaurants(preferenceField, path='restaurant_info_extended.csv'):
    restaurants = pd.read_csv('restaurant_info_extended.csv',sep=';')

    if preferenceField['area'] != 'dontcare' and preferenceField['area'] is not None:
        restaurants = restaurants[restaurants['area'] == preferenceField['area']]
    if preferenceField['pricerange'] != 'dontcare' and preferenceField['pricerange'] is not None:
        restaurants = restaurants[restaurants['pricerange'] == preferenceField['pricerange']]
    if preferenceField['food'] != 'dontcare' and preferenceField['food'] is not None:
        restaurants = restaurants[restaurants['food'] == preferenceField['food']]

    return restaurants.values

# Lookup function to filter restaurants fitting the additional requirements
# Restaurants is a numpy array, requirements should be provided as boolean
# @return is a numpy array
def filter_restaurants_opt_requirements(restaurants, optionalPreferences, reasoning_rules):
    for key in optionalPreferences:
        if optionalPreferences[key]:
            for column,value,condition in reasoning_rules[key]:
                restaurants = restaurants[(restaurants[:,column] == value) == condition]
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