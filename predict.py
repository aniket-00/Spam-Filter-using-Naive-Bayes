import numpy as np
import csv
import sys
import json

from validate import validate


"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_nb.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n', dtype=str)
    X=np.genfromtxt("train_X_nb.csv",delimiter="\n",dtype=str)
    Y=np.genfromtxt("train_Y_nb.csv",delimiter=",",dtype=np.float64)
    
    return test_X,X,Y

    
def remove_spl_char(s):
    new_s = ""
    for i in range(len(s)):
        if (ord(s[i]) >= ord("A") and ord(s[i]) <= ord('Z') or ord(s[i]) >= ord('a') and ord(s[i]) <= ord('z') or ord(s[i]) == ord(' ')):
           new_s += s[i]

    return new_s


def preprocessing(s):
    a = []
    for i in range(len(s)):
      s[i] = remove_spl_char(s[i])
      a.append([s[i]])

    return a


def class_wise_words_frequency_dict(X, Y):
    X = list(X)
    class_wise_frequency_dict = dict()

    for i in range(len(X)):
        words = X[i][0].split()
        for token_word in words:
            y = Y[i]
            if y not in class_wise_frequency_dict:
                class_wise_frequency_dict[y] = dict()
            if token_word not in class_wise_frequency_dict[y]:
                class_wise_frequency_dict[y][token_word] = 0
            class_wise_frequency_dict[y][token_word] += 1

    return class_wise_frequency_dict


def compute_prior_probabilities(Y):
    Y = list(Y)
    classes = list(set(Y))
    n_docs = len(Y)
    prior_probabilities = dict()
    for c in classes:
        prior_probabilities[c] = Y.count(c) / n_docs
    return prior_probabilities


def get_class_wise_denominators_likelihood(X, Y,class_wise_frequency_dict):
    
    classes=list(set(Y))
    class_wise_denominators = dict()
    vocabulary = []
    for c in classes:
        frequency_dict = class_wise_frequency_dict[c]
        class_wise_denominators[c] = sum(list(frequency_dict.values()))
        vocabulary += list(frequency_dict.keys())

    vocabulary = list(set(vocabulary))

    for c in classes:
        class_wise_denominators[c] += len(vocabulary)

    return class_wise_denominators


def compute_likelihood(test_X, c,class_wise_denominators, class_wise_frequency_dict):
    likelihood = 0
    
    words = test_X[0].split()
    for word in words:
        count = 0
        words_frequency = class_wise_frequency_dict[c]
        if word in words_frequency:
            count = class_wise_frequency_dict[c][word]
        likelihood += np.log((count + 0.5)/class_wise_denominators[c])
    return likelihood


def predicts(test_X, X, Y,class_wise_denominators, class_wise_frequency_dict,classes):

    best_likelihood=-99999
    best_c=-1
    c=[]
    

    for i in classes:
        likelihood = compute_likelihood(test_X, i, class_wise_denominators, class_wise_frequency_dict)
        if likelihood>best_likelihood:
            best_likelihood=likelihood
            best_c=i
     
    return best_c


def predict_target_values(test_X, X, Y):
    test_X = list(test_X)
    test_X = preprocessing(test_X)
    validation_set=X[800:2000]
    class_wise_frequency_dict = class_wise_words_frequency_dict(X, Y)
    class_wise_denominators = get_class_wise_denominators_likelihood(X, Y,class_wise_frequency_dict)
    
    b = []
    Y=list(Y)
    classes=list(set(Y))
    
    for i in range(len(test_X)):
        a = predicts(test_X[i], validation_set, Y,class_wise_denominators,class_wise_frequency_dict,classes)
        b.append([a])

    b = np.array(b)
    return b

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.
    """
    test_X, X,Y = import_data_and_model(test_X_file_path)
    X = preprocessing(X)
    pred_Y = predict_target_values(test_X,X,Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 
