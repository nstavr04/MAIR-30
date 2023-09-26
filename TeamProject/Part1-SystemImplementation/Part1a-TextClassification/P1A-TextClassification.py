from sklearn.model_selection import train_test_split
from DataPreparation import get_data


from BaselineClassifiers import baseline_majority
from BaselineClassifiers import baseline_keyword
from BaselineClassifiers import baseline_prompt
from ML_Classifiers import ml_decision_tree_classifier
from ML_Classifiers import ml_decision_tree_classifier_prompt
from ML_Classifiers import ml_logistic_regression_classifier
from ML_Classifiers import ml_logistic_regression_classifier_prompt



################################### Dataset ##########################################

#get the data, x contains the acts, y contains the labels
x,y,x_deduplicated,y_deduplicated = get_data()


# Splitting the dataset into training and test sets
# 85% of the data is used for training and 15% for testing
# random state is like seed I think to just keep the same split and shuffling
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True)

x_train_unique, x_test_unique, y_train_unique, y_test_unique = train_test_split(x_deduplicated, y_deduplicated, test_size=0.15, shuffle=True)


# Scikit needs numpy
x_train_unique_np = x_train_unique.values
y_train_unique_np = y_train_unique.values

x_test_unique_np = x_test_unique.values
y_test_unique_np = y_test_unique.values

#function to print the system output with respect to the choice of the user
def print_system_output(choice):
    if choice == '1':
        baseline_majority_accuracy, c_report = baseline_majority(y_test)
        print(f"Baseline majority accuracy: {baseline_majority_accuracy:.2f}%")
    elif choice == '2':
        baseline_keyword_accuracy, c_report = baseline_keyword(x_test, y_test)
        print(f"Baseline keyword accuracy: {baseline_keyword_accuracy:.2f}%")
    elif choice == '3':
        baseline_prompt()
    elif choice == '4':
        ml_decision_tree_classifier_accuracy_dups, c_report = ml_decision_tree_classifier(x_train, y_train, x_test,
                                                                                          y_test)
        print(
            f"Machine Learing Decision Tree Classifier accuracy (with Duplicates): {ml_decision_tree_classifier_accuracy_dups:.2f}%")

        ml_decision_tree_classifier_prompt(x_train, y_train)
    elif choice == '5':
        ml_decision_tree_classifier_accuracy_nodups, c_report = ml_decision_tree_classifier(x_train_unique_np,
                                                                                            y_train_unique_np,
                                                                                            x_test_unique_np,
                                                                                            y_test_unique_np)
        print(
            f"Machine Learing Decision Tree Classifier accuracy (without Duplicates): {ml_decision_tree_classifier_accuracy_nodups:.2f}%")

        ml_decision_tree_classifier_prompt(x_train_unique_np, y_train_unique_np)
    elif choice == '6':
        ml_logistic_regression_classifier_accuracy_dups, c_report = ml_logistic_regression_classifier(x_train, y_train,
                                                                                                      x_test, y_test)
        print(
            f"Machine Learing Logistic Regression Classifier accuracy (with Duplicates): {ml_logistic_regression_classifier_accuracy_dups:.2f}%")

        ml_logistic_regression_classifier_prompt(x_train, y_train)
    elif choice == '7':
        ml_logistic_regression_classifier_accuracy_nodups, c_report = ml_logistic_regression_classifier(
            x_train_unique_np, y_train_unique_np, x_test_unique_np, y_test_unique_np)
        print(
            f"Machine Learing Logistic Regression Classifier accuracy (without Duplicates): {ml_logistic_regression_classifier_accuracy_nodups:.2f}%")

        ml_logistic_regression_classifier_prompt(x_train_unique_np, y_train_unique_np)
    elif choice == '8':
        baseline_majority_accuracy, c_report1 = baseline_majority(y_test)
        print(f"Baseline majority accuracy: {baseline_majority_accuracy:.2f}%")
        print(c_report1)
        baseline_keyword_accuracy, c_report2 = baseline_keyword(x_test, y_test)
        print(f"Baseline keyword accuracy: {baseline_keyword_accuracy:.2f}%")
        print(c_report2)
        ml_decision_tree_classifier_accuracy_dups, c_report3 = ml_decision_tree_classifier(x_train, y_train, x_test,
                                                                                           y_test)
        print(
            f"Machine Learing Decision Tree Classifier accuracy (with Duplicates): {ml_decision_tree_classifier_accuracy_dups:.2f}%")
        print(c_report3)
        ml_decision_tree_classifier_accuracy_nodups, c_report4 = ml_decision_tree_classifier(x_train_unique_np,
                                                                                             y_train_unique_np,
                                                                                             x_test_unique_np,
                                                                                             y_test_unique_np)
        print(
            f"Machine Learing Decision Tree Classifier accuracy (without Duplicates): {ml_decision_tree_classifier_accuracy_nodups:.2f}%")
        print(c_report4)
        ml_logistic_regression_classifier_accuracy_dups, c_report5 = ml_logistic_regression_classifier(x_train, y_train,
                                                                                                       x_test, y_test)
        print(
            f"Machine Learing Logistic Regression Classifier accuracy (with Duplicates): {ml_logistic_regression_classifier_accuracy_dups:.2f}%")
        print(c_report5)
        ml_logistic_regression_classifier_accuracy_nodups, c_report6 = ml_logistic_regression_classifier(
            x_train_unique_np, y_train_unique_np, x_test_unique_np, y_test_unique_np)
        print(
            f"Machine Learing Logistic Regression Classifier accuracy (without Duplicates): {ml_logistic_regression_classifier_accuracy_nodups:.2f}%")
        print(c_report6)
    elif choice == '0':
        print("Exiting...")
    else:
        print("Invalid choice. Please try again.")

#function to print the menu of the application
def print_menu():
    print("\nChoose an option:")
    print("1. Run Baseline Majority Class")
    print("2. Run Baseline Keyword Matching")
    print("3. Run Baseline Keyword Matching Prompt Predictions")
    print("4. Run ML Decision Tree Classifier Algorithm (with Duplicates)")
    print("5. Run ML Decision Tree Classifier Algorithm (without Duplicates)")
    print("6. Run ML Logistic Regression Classifier Algorithm (with Duplicates)")
    print("7. Run ML Logistic Regression Classifier Algorithm (without Duplicates)")
    print("8. Run everything - Classification Reports")
    print("0. Exit")

def main_menu():
    while True:
        #print possible options
        print_menu()

        choice = input("Enter your choice: ")

        #print system output to user request
        print_system_output(choice)

        #user wants to leave
        if choice == 0:
            break

if __name__ == "__main__":
    main_menu()