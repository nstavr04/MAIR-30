import numpy as np
# Function to apply inference rules if more than one restaurant is available given the preferences
def apply_inference_rules(consequent, restaurants):
    # The additional requirements are the consequent(s) of the rules
    # food quality, crowdedness, length of stay

    # rules:
    # if cheap AND good food --> touristic (true)
    # if romanian --> touristic (false)
    # if busy --> assigned seats (true)
    # if long stay --> children (false)
    # if busy --> romantic (false)
    # if long stay --> romantic (true)

    restaurants_to_delete = []
    message = ""

    for restaurant in restaurants:
        if 'touristic' in consequent:
            #TODO check rule 1 and 2
            if (restaurant['price range'] == "cheap" and restaurant['food quality'] == 'good food' and restaurant['food']!= 'romanian'):
                message += "The restaurant is touristic, because it is cheap, has good food and is not romanian."
            #elif (restaurant['food'] == "romanian"):
            #    restaurant
            else:
                restaurants_to_delete.append(restaurant)
        elif 'romantic' in consequent:
            #TODO check rule 5 and 6
            if (restaurant['crowdedness'] != 'busy' and restaurant['length of stay'] == 'long stay'):
                message += "The restaurant is romantic, because it is not busy and you can stay for a long time."
            else:
                restaurants_to_delete.append(restaurant)
        elif 'assigned seats' in consequent:
            #TODO check rule 3
            if (restaurant['crowdedness'] == busy):
                message += "The restaurant has assigned seats, becausey it is a busy restaurant."
            else:
                restaurants_to_delete.append(restaurant)
        elif 'children' in consequent:
            #TODO check rule 4
            if (restaurant['length of stay'] != 'long stay'):
               message += "The restaurant is suitable for children, because the length of stay is not long."
            else:
                restaurants_to_delete.append(restaurant)

    #return (restaurants - restaurants_to_delete)
    #TODO add a message
    return (np.delete(restaurants, np.array(restaurants_to_delete)), "message")

#def check_restaurants(restaurants):
