import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

test = pd.read_csv('test_edit.csv')
train = pd.read_csv('train_edit.csv')

# creating model
cols = ['Pclass_1', 'Pclass_2', 'Pclass_3',
        'Female', 'Male', 'Child', 'Not Child']

# split data into train_x, train_y, test_x, test_y
all_x = train[cols]
all_y = train['Survived']

train_x, test_x, train_y, test_y = train_test_split(
    all_x, all_y, test_size=0.2, random_state=0)

# using logistic regression
lr = LogisticRegression(solver='lbfgs')
lr.fit(train_x, train_y)
predictions = lr.predict(test_x)
accuracy = accuracy_score(test_y, predictions)
print('Accuracy: {}'.format(accuracy))

conf_matrix = confusion_matrix(test_y, predictions)
df = pd.DataFrame(conf_matrix, columns=['Survived', 'Died'],
            index=[['Survived', 'Died']])
print(df)
