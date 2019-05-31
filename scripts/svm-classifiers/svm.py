import os
import re
import sys

import numpy
import sklearn
import string
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def prune(word):
    result = []
    tmp = ""
    for char in word:
        if char in string.punctuation:
            if len(tmp) > 2:
                result.append("".join(tmp))
                tmp = ""
        else:
            tmp += char.lower()
    if len(tmp) > 2:
        result.append("".join(tmp))
    return result


def main(argv):
    if len(argv) < 2:
        print("usage: python svm.py <file directory>")
        exit()

    dataset = []
    test_labels = []
    file_dir = argv[1]

    for file_name in os.listdir(file_namedir):
        with open(file_dir + '/' + file_name) as file:
            document = []
            try:
                a = map(lambda s: s.strip().split(' '), file)

                for sentence in a:
                    for raw_word in sentence:
                        for word in prune(raw_word):
                            document.append(word.lower())

                dataset.append(' '.join(document))
                test_labels.append(re.sub(r"\d", '', file_name))

            except UnicodeDecodeError as e:
                pass
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

    vectorizer = TfidfVectorizer(min_df=0)
    tfidf_matrix = vectorizer.fit_transform(dataset)

    x_train, x_test, y_train, y_test = train_test_split(tfidf_matrix, test_labels, test_size=0.2)
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
