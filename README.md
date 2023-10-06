# Restaurant Recommendation System - Methods In AI - Group 30

This is the repository for the Restaurant Recommendation System project for the 'Methods in AI' course at UU.

**Team members:**
MILO TREANOR
LINUS ERREN
ROEMER SCHWAANHUYSER
NIKOLAS STAVROU
RENATE BUREMA 

## Getting Started

Make sure to have the following packages installed:

### Python version
python=3.10.13

### Required libraries
numpy=1.25.2
pandas=2.0.3
scikit-learn=1.3.0
matplotlib=3.7.2

#### ASR libraries
pyaudio=0.2.11
SpeechRecognition=3.10.0
scipy=1.11.1

## Files Required

### Data files

**dialog_acts.dat** - Used to train our ML classifier model for classifying dialog acts at the start of the program.

**restaurant_info.csv** - The dataset of restaurants used for the recommendation system when recommending restaurants given certain preferences.

**restaurant_info_extended.csv** - An extended dataset of restaurants that includes the additional preferences as well from part 1c.

### Config file

**configurations.json** - Includes the different configurations of the system.
The configurations can be changed through the CLI interface of the recommendation system.

### Python files for recommendation system

**MainScript_DialogSystem.py** - The main script of the system. Run this file to start the system.
Includes the state transition function for controlling the flow of the recommendation system. 

**Leven_Distance.py** - Implementation of variations of levenstein distance, used for the keyword mathing algorithm.

**Input_Output_Functions.py** - Contains functions to handle Speech-Input and the outputs of the system in the CLI. 

**Filter_Restaurants.py** - Contains functions to find suitable restaurants fitting the preferences of the user.

**Analyze_Utterance.py** - Contains functions to analyze an utterance such as for preferences and misspellings.
Includes the keyword matching algorithm.

**ASR_userUtterance.py** - Contains functions to handle the Speech Regocnition - Speech-to-Text functionality of the system.

### Additional files

These files are used for Part 1a of the project.

**Baseline_Classifiers.py** - Contains additional functions for using the baseline models

**ML_Classifiers.py** - Contains functions to get/train the ML models

**Data_Preparation.py** - Data preprocessing

**Evaluation.py** - Contains functions to evaluate the trained ML models

**P1A-TextClassification.py** - The main script for running Part 1a of the project.

## Running the system

After you make sure all files are in the same directory, run the MainScript_DialogSystem.py file to start the system.
**python MainScript_DialogSystem.py**

CLI interface will appear, where you can change the configurations of the system.

Typing 'change' will allow you to change the configurations one by one. The configurations will be saved in the configurations.json file, and you can start the program again with the new configurations.

Typing 'continue' will start the recommendation system.

## Additional information

In the folder Part 1a you will find the files used for Part 1a of the project.
In the folder DialogSystem-Part1b-Part1c you will find the files used for Part 1b and Part 1c of the project (the recommendation system).

DialogSystemStateDiagram.png is the state diagram of the recommendation system.

The report of the project can also be found in .pdf format.