# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:30:31 2017

@author: pulki
"""


import pandas as pd
import numpy as np


readtext = pd.read_csv('document.txt' , names = ['ctg' , 'doc'] )
dataset = pd.DataFrame(columns = ['ctg' , 'doc'])
n = int(readtext.iloc[0 , 0])
readtext = readtext.iloc[1: , :]
for i in range(0,n):
    dataset.loc[i] =[ int(str(readtext.iloc[i , 0])[0]) ,   str(readtext.iloc[i , 0])[1:] ]
    

    
# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
cleaned = list()
for i in range(0, 5485):
    article = re.sub('[^a-zA-Z]', ' ', dataset['doc'][i])
    article = article.split()
    article = [lemmatizer.lemmatize(word) for word in article if not word in set(stopwords.words('english'))]
    article = ' '.join(article)
    cleaned.append(article)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 750)
X = cv.fit_transform(cleaned).toarray()
y = dataset.iloc[: , 0].values.astype(int)
y = keras.utils.to_categorical(y)



# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 300, init = 'uniform', activation = 'relu', input_dim = 750))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'softmax')) 

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X[:5000 , :], y[:5000 , :], batch_size = 10, nb_epoch = 50)


# Part 3 - Making the predictions and evaluating the model

# analyzing the performance of our model
score = classifier.evaluate( X[5000: , :] , y[5000: , :] )

y_pred = classifier.predict( X[5000: , :])

y_pred = y_pred.argmax( axis = 1)
y_test = y[5000: , :].argmax( axis = 1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix (y_test , y_pred)





