import pandas as pd

path = '/home/adam/Documents/kaggle/titanic/data/'

test = pd.read_csv(path+'test.csv')
train = pd.read_csv(path+'train.csv')

# inspecting data
print('Test data shape: {}'.format(test.shape))
print('Train data shape: {}'.format(train.shape))
print('Columns in data: {}'.format(test.columns))
print('Train head\n{}'.format(train.head(10)))
