from __future__ import print_function
import nltk
import os
import random

from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer
from nltk import NaiveBayesClassifier, classify
from pyspark import SparkContext, SparkConf

import time


def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r', encoding='latin1')
        a_list.append(f.read())
    f.close()
    return a_list

spam = init_lists('linguistspam/spam/')
ham = init_lists('linguistspam/ham/')

all_emails = [(email, 'spam') for email in spam]
all_emails += [(email, 'ham') for email in ham]


# Randomly suffling email
random.shuffle(all_emails)

# lammetizing and tokenizing words in email
lemmatizer = WordNetLemmatizer()
def preprocess(sentence):
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]

stoplist = stopwords.words('english')

def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]

#------------Naive Bayes classification method-------------
#Training a classifier
def naive_bayes_train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')

    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier


#------------Decision tree classification method-------------
#Training a classifier
def decision_tree_train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')

    classifier = nltk.DecisionTreeClassifier.train(train_set)
    return train_set, test_set, classifier


#--------------MaxEnt classification method-------------------
#Training a classifier
def maxent_train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')

    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm,max_iter=3)

    return train_set, test_set, classifier

#Evaluating your classifier performance
def evaluate(train_set, test_set, classifier):
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)) + '\n')

#Create RDDs
conf = SparkConf()
sc = SparkContext(conf=conf)
	
data = sc.parallelize(all_features)

#print(data.first())

rdd = data.collect()

# Apply Naive Bayes Classification
print('Classification using Naive Bayes.')
NB_start_time = time.time()
train_set, test_set, classifier = naive_bayes_train(rdd, 0.8)
evaluate(train_set, test_set, classifier)
print("--- %s seconds ---" % (time.time() - NB_start_time))
classifier.show_most_informative_features(20)

# Apply Decision Tree Classification
print('\n Classification using Decision Tree')
DT_start_time = time.time()
train_set, test_set, classifier = decision_tree_train(rdd, 0.8)
evaluate(train_set, test_set, classifier)
print("--- %s seconds ---" % (time.time() - DT_start_time))

# Apply MaxEnt Classification
print('\n Classification using MaxEnt')
ME_start_time = time.time()
train_set, test_set, classifier = maxent_train(rdd, 0.8)
evaluate(train_set, test_set, classifier)
print("--- %s seconds ---" % (time.time() - ME_start_time))