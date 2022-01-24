import numpy as np
import csv
import json


def import_package():

    X=np.genfromtxt("train_X_nb.csv",delimiter="\n",dtype=str,skip_header=1)
    Y=np.genfromtxt("train_Y_nb.csv",delimiter=",",dtype=np.float64)

    return X,Y


def remove_spl_char(s):
    new_s = ""
    for i in range(len(s)):
        if (ord(s[i]) >= ord("A") and ord(s[i]) <= ord('Z') or ord(s[i]) >= ord('a') and ord(s[i]) <= ord('z') or ord(s[i]) == ord(' ')):
           new_s += s[i]

    return new_s


def preprocessing(s):
    a=[]
    for i in range(len(s)):
      s[i] = remove_spl_char(s[i])
      a.append([s[i]])
    
    return a


def class_wise_words_frequency_dict(X, Y):
    X=list(X)
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
    Y=list(Y)
    classes = list(set(Y))
    n_docs = len(Y)
    prior_probabilities = dict()
    for c in classes:
        prior_probabilities[c] = Y.count(c) / n_docs
    return prior_probabilities


def get_class_wise_denominators_likelihood(X, Y):
    class_wise_frequency_dict = class_wise_words_frequency_dict(X, Y)

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

def compute_likelihood(test_X, c):
    likelihood = 0
    
    words = test_X[0].split()
    for word in words:
        count = 0
        words_frequency = class_wise_frequency_dict[c]
        if word in words_frequency:
            count = class_wise_frequency_dict[c][word]
        likelihood += np.log((count + 1)/class_wise_denominators[c])
    return likelihood





def model(X,Y):

    validation_set=X[800:2000]
    like=[]
    for i in range(len(X)):
        likelihood=compute_likelihood(X[i][0],Y[i])
        like.append([likelihood])
    
    
    model=like
    print(json.dumps(model))
    with open("./MODEL_FILE.json", "w") as outfile:
        json.dump(model,outfile)
 
    






if __name__=="__main__":
   
   X,Y=import_package()
   X=preprocessing(X)
   classes=list(set(Y))
   class_wise_denominators = get_class_wise_denominators_likelihood(X, Y)
   class_wise_frequency_dict = class_wise_words_frequency_dict(X, Y)
   prior_probabilities=compute_prior_probabilities(Y)
   model(X,Y)

