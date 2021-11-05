import pandas as pd

# We will display only 3 decimals per sample
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Function to read data
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset


# Load the dataset
dataset = load_dataset('data/adult.data')
data = dataset.values
print(data)

x = data[:, :2]
y = data[:, 2]

# To see the sizes of the dataset
print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X:", x.shape)
print("Dimensionalitat de l'atribut Y:", y.shape)

# To see all the columns with no values
# En este dataset estan con "?", hay que buscar alguna manera TODO
"""
x = dataset.isnull().sum()
null_columns = dataset.columns[dataset.isnull().any()]
print(null_columns)
print(dataset[null_columns].isnull().sum())
print("Total de valors no existents:", dataset.isnull().sum().sum())
"""
