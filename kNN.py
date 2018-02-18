
# coding: utf-8

# # Digit Recognizer

# An introduction to classification with k-NN on the handwritten digits problem 
# 
# MNIST Data from [Kaggle](https://www.kaggle.com/c/digit-recognizer)


# ## 0. Import Libraries

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ## 1. Load and Format Datasets

print 'Loading data...\n'
path_to_file = '~/Desktop/DigitRecognizer/'
train = pd.read_csv(path_to_file+'data/train.csv')
test = pd.read_csv(path_to_file+'data/test.csv')

print 'Train Set Shape: {}'.format(train.shape)
print 'Test Set Shape: {}\n'.format(test.shape)

print 'Head of Train Set:\n'
print train.head()

# Separate the label from the feature vectors
train_y = train['label']
del train['label']


# ## 2. Visualize a Sample Image

print '\nVisualizing a Sample Image...\n'
# Select a sample image 
sample_label = train_y[5]
sample_image = train.iloc[5]
pixels = np.array(sample_image, dtype='uint8')

# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = pixels.reshape((28, 28))

# Plot
plt.title('Label is {}'.format(sample_label))
plt.imshow(pixels, cmap='gray')
plt.show()


# ## 3. Model Selection

# Split data into hold-out set
print 'Splitting data into hold-out set..\n'
X_train, X_test, y_train, y_test = train_test_split(train, train_y, test_size=0.2, random_state=42)

# Generate list of potential K values
k_list = range(1,11)

# ### For Loop Approach
print 'Optimizing k...\n'
print('k', 'Error Rate')
for k in k_list:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    err_rate = 1 - accuracy_score(y_test, preds)
    print(k, err_rate)


# # ### GridSearchCV Approach
# from sklearn.model_selection import GridSearchCV

# start = time.time()

# parameters = {'n_neighbors': k_list}
# knn = KNeighborsClassifier()
# clf = GridSearchCV(knn, parameters, scoring='accuracy', refit=True)
# clf.fit(X_train, y_train)

# end = time.time()
# run_time = float(end - start)/60

# print 'Best K: {}'.format(clf.best_params_)
# print 'Best GridSearchCV accuracy score: {}'.format(clf.best_score_)
# print 'Run time: {:0.2f} minutes'.format(run_time)


# ## 5. Model Fitting and Predictions

# Fit k-NN
# If we took the GridSearchCV approach, the classifier would already be fit to full train set,
# and we could skip to the prediction part.
print '\nFitting model...\n'
k = 1
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(train, train_y)

# Make predictions on test set
print '\nMaking predictions...\n'
start = time.time()

preds = clf.predict(test)

end = time.time()
run_time = float(end - start)/60
print 'Run time of prediction generation: {:0.2f} minutes\n'.format(run_time)


# ## 6. Export Results

# Format submission, 1-indexed, column labels ImageId and Label
print 'Formatting and exporting predictions...\n'
ImageId = np.array(range(1,len(preds)+1))
d = {'ImageId': ImageId, 'Label': preds}
df_preds = pd.DataFrame(d)

print 'Head of Prediction DataFrame:\n'
print df_preds.head()

sub_number = 1
df_preds.to_csv(path_to_file+'submissions/submission{}.csv'.format(sub_number), index=False)

print '\nProgram complete!'
