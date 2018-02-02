import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load datasets
train = pd.read_csv('~/Desktop/DigitRecognizer/data/train.csv')
test = pd.read_csv('~/Desktop/DigitRecognizer/data/test.csv')

print 'Train Set Shape: {}'.format(train.shape)
print 'Test Set Shape: {}'.format(test.shape)

#train.head()

# Separate the label from the feature vectors
train_y = train['label']
del train['label']

# Visualize a sample image 
sample_label = train_y[5]
sample_image = train.iloc[5]
pixels = np.array(sample_image, dtype='uint8')

# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = pixels.reshape((28, 28))

# Plot
plt.title('Label is {}'.format(sample_label))
plt.imshow(pixels, cmap='gray')
plt.show()

k_list = range(1,11)

# Split data into hold-out set
X_train, X_test, y_train, y_test = train_test_split(train, train_y, test_size=0.2, random_state=42)

print('k', 'Error Rate')
for k in k_list:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    err_rate = 1 - accuracy_score(y_test, preds)
    print(k, err_rate)

# # Also see GridSearchCV in the model_selection class
# # from sklearn.model_selection import GridSearchCV
# parameters = {'n_neighbors': k_list}
# clf = GridSearchCV(KNeighborsClassifier, parameters)
# clf.fit(X_train, y_train)
# print 'Best Parameter: {}'.format(best_params_)

# Fit k-NN
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(train, train_y)

# Make predictions on test set
preds = clf.predict(test)

# Format submission, 1-indexed, column labels ImageId and Label
ImageId = np.array(range(1,len(preds)+1))
d = {'ImageId': ImageId, 'Label': preds}
df_preds = pd.DataFrame(d)
df_preds.head()

# Export results to CSV
sub_number = 3
df_preds.to_csv('submissions/submission{}.csv'.format(sub_number), index=False)

