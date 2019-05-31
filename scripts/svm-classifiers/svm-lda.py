import os
import re
import sys

import numpy
import sklearn
import string
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split



def main(argv):
    if len(argv) < 2:
        print("usage: python svm.py <file directory>")
        exit()

    dataset = []
    test_labels = []
    file_with_paths = argv[1]
    lda_gamma_matrix = argv[2]

    for file_name in open(file_with_paths):
        test_labels.append(re.sub(r"\d", '', file_name.strip()))


    dataset = numpy.loadtxt(lda_gamma_matrix)
    labels = {}
    c = 0
    for name in test_labels:
        if name not in labels:
            labels[name] = c
            c += 1

    for i in range(len(test_labels)):
        test_labels[i] = labels[test_labels[i]]


    print(len(dataset))
    print(len(test_labels))


    x_train, x_test, y_train, y_test = train_test_split(dataset, test_labels, test_size=0.2)
    c_values = [100.0, 50.0, 10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]


    print("RBF kernel:")
    for c in c_values:
        classifier = svm.SVC(C=c)
        classifier.fit(x_train, y_train)
        acc = classifier.score(x_test, y_test)
        print("C = {:.3}, Accuracy = {}".format(c, acc))

    print("Linear kernel:")
    for c in c_values:
        classifier = svm.LinearSVC(C=c)
        classifier.fit(x_train, y_train)
        acc = classifier.score(x_test, y_test)
        print("C = {:.3}, Accuracy = {}".format(c, acc))






if __name__ == "__main__":
    main(sys.argv)
