import os
import io
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = pd.concat([data, dataFrameFromDirectory("C:\MLCourse\emails\spam", "spam")]);
data = pd.concat([data, dataFrameFromDirectory("C:\MLCourse\emails\ham", "ham")])


# print(data.head())

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values) #splits message into list of words 

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)



examples = ['Slim Down - Guaranteed to lose 10-12 lbs in 30 days http://www.freeyankee.com/cgi/fy2/to.cgi?l=822slim1','Fight The Risk of Cancer! http://www.freeyankee.com/cgi/fy2/to.cgi?l=822nic1','Get the Child Support You Deserve - Free Legal Advice http://www.freeyankee.com/cgi/fy2/to.cgi?l=822ppl1']
examples_counts = vectorizer.transform(examples) #conver each message into words & their frequencies. 
print(examples)
predictions = classifier.predict(examples_counts) #once converted, use predict function on list of words to predict.
print(predictions)