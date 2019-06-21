import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/home/adam/Documents/kaggle/titanic/'

test = pd.read_csv(path+'data/test.csv')
train = pd.read_csv(path+'data/train.csv')

# inspecting data
print('Test data shape: {}'.format(test.shape))
print('Train data shape: {}'.format(train.shape))
print('Columns in data: {}'.format(test.columns))
print('Train head\n{}'.format(train.head(10)))

# misc pivot tables to view initial assumptions
sex_pt = train.pivot_table(index='Sex', values='Survived')
print(sex_pt)
pclass_pt = train.pivot_table(index='Pclass', values ='Survived')
print(pclass_pt)
train['Adult'] = np.where(train['Age'] < 18, 'no',
                            np.where(train['Age'] >=18, 'yes', 'unknown'))
adult_pt = train.pivot_table(index='Adult', values='Survived')
print(adult_pt)

# diving into ages
'''
survived = train[train['Survived'] == 1]
died = train[train['Survived'] == 0]
survived['Age'].plot.hist(alpha=0.5, color='green', bins=50)
died['Age'].plot.hist(alpha=0.5, color='red', bins=50)
plt.legend(['Survived', 'Died'])
plt.savefig(path+'images/plot.png', format='png')
'''

# formatting data for model
print(train['Pclass'].value_counts())
# split ticket class
train['Pclass_1'] = np.where(train['Pclass'] == 1, 1, 0)
train['Pclass_2'] = np.where(train['Pclass'] == 2, 1, 0)
train['Pclass_3'] = np.where(train['Pclass'] == 3, 1, 0)
print(train[['Pclass', 'Pclass_1', 'Pclass_2', 'Pclass_3']].head(5))
test['Pclass_1'] = np.where(test['Pclass'] == 1, 1, 0)
test['Pclass_2'] = np.where(test['Pclass'] == 2, 1, 0)
test['Pclass_3'] = np.where(test['Pclass'] == 3, 1, 0)
print(test[['Pclass', 'Pclass_1', 'Pclass_2', 'Pclass_3']].head(5))
# split sex
print(train['Sex'].value_counts())
train['Female'] = np.where(train['Sex'].str.contains('female'), 1, 0)
train['Male'] = np.where(train['Sex'].str.match('male'), 1, 0)
print(train[['Sex', 'Female', 'Male']].head(5))
test['Female'] = np.where(test['Sex'].str.contains('female'), 1, 0)
test['Male'] = np.where(test['Sex'].str.match('male'), 1, 0)
print(test[['Sex', 'Female', 'Male']].head(5))
# adult vs not adult
train['Child'] = np.where(train['Age'] < 18, 1, 0)
train['Not Child'] = np.where(train['Age'] >= 18, 1, 0)
print(train[['Age','Child','Not Child']].head(15))
test['Child'] = np.where(test['Age'] < 18, 1, 0)
test['Not Child'] = np.where(test['Age'] >= 18, 1, 0)
print(test[['Age','Child','Not Child']].head(25))
