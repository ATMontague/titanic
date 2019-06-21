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
survived = train[train['Survived'] == 1]
died = train[train['Survived'] == 0]
survived['Age'].plot.hist(alpha=0.5, color='green', bins=50)
died['Age'].plot.hist(alpha=0.5, color='red', bins=50)
plt.legend(['Survived', 'Died'])
plt.savefig(path+'images/plot.png', format='png')
