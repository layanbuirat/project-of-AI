#! Installing all packages needed
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import time

#! Load our datasets into DataFrames
data1 = pd.read_csv('dataset1.csv', delimiter='\t')
data2 = pd.read_csv('dataset2.csv', delimiter='\t')
data3 = pd.read_csv('dataset3.csv', delimiter='\t')

#! Letting the user decide which dataset he/she wants
print("Last ID for dataset1:2768")
print("Last ID for dataset2:768")
print("Last ID for dataset3:4303, Note:(This dataset has more number of features)")

dataset_choice = input("Enter the dataset you want to test (1 or 2 or 3): ")

if dataset_choice == '1':
    data = data1
elif dataset_choice == '2':
    data = data2
elif dataset_choice == '3':
    data = data3
else:
    print("Invalid dataset choice.")
    exit()

#! Separate features (X) and labels (y), X for all features except Outcome,y for Outcome feature
X = data.drop('Outcome', axis=1)
y = data['Outcome']

#! Split the dataset into training and testing sets (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

while True:
    #! Ask the user to choose the algorithm
    print("\n\t")
    print("Choose the algorithm you want to test:")
    print("1. Decision Tree")
    print("2. Naive Bayes")
    print("3. Neural Network")
    print("4. Random Forest")
    print("5. Exit")
    choice = input("Enter your choice (1, 2, 3, 4, or 5): ")

    if choice == '1':
        #! Initialize the decision tree model
        model = DecisionTreeClassifier()
    elif choice == '2':
        #! Initialize the Naive Bayes model
        model = GaussianNB()
    elif choice == '3':
        #! Initialize the Neural Network model
        model = MLPClassifier(random_state=1, max_iter=300)
    elif choice == '4':
        #! Initialize the Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif choice == '5':
        break
    else:
        print("Invalid choice.")
        continue

    #! Train the model
    model.fit(X_train, y_train)

    #! Start time
    start_time = time.time()

    #! Make predictions on the test set
    y_pred = model.predict(X_test)

    #! Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    #! Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #! Round the evaluation metrics to 5 decimal places
    accuracy = round(accuracy, 5)
    precision = round(precision, 5)
    recall = round(recall, 5)
    f1 = round(f1, 5)

    #! Round the time elapsed to 5 decimal places
    time_elapsed = round(time.time() - start_time, 5)

    #! Print algorithm name
    if choice == '1':
        print("\nDecision Tree Algorithm:-\n")
    elif choice == '2':
        print("\nNaive Bayes Algorithm:-\n")
    elif choice == '3':
        print("\nNeural Network Algorithm:-\n")
    elif choice == '4':
        print("\nRandom Forest Algorithm:-\n")

    #! Print confusion matrix
    print("Confusion Matrix:")
    print("\t\t\tActual Positive\t\t\tActual Negative")
    print("Classified Positive\t\t", conf_matrix[1, 1], "\t\t\t", conf_matrix[1, 0])
    print("Classified Negative\t\t", conf_matrix[0, 1], "\t\t\t", conf_matrix[0, 0])

    #! Print evaluation metrics
    print("\n")
    print("Time elapsed:", time_elapsed, "seconds")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("\n")

    #! Ask the user to enter values for each feature
    value1 = input(f"Do you want to test values from your own of this algorithm (y/n)")
    if value1.lower() == 'y':
        print("\nEnter values for each feature:")
        features = {}
        for column in X.columns:
            value = input(f"Enter value for {column}: ")
            features[column] = [value]

        #! Create a DataFrame with the user-provided values
        user_data = pd.DataFrame(features)

        #! Make predictions using the trained model
        user_pred = model.predict(user_data)

        #! Display the prediction
        if user_pred[0] == 1:
            print("The expected result for values entered is to have diabetes :( ")
        else:
            print("The expected result for values entered is to (NOT) have diabetes :) ")
    else:
        continue
