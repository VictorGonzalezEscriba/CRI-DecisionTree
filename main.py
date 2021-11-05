import pandas as pd

# We will display only 3 decimals per sample
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Function to read data
def load_dataset(path):
    # Add the columns headers, see adult.names
    head = ['Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum', 'MaritalStatus', 'Occupation', 'Relationship',
            'Race', 'Sex', 'CapitalGain', 'CapitalLoss', 'HPW', 'NativeCountry', 'Income']
    df = pd.read_csv(path, header=None, names=head, delimiter=', ', engine='python')
    return df


# En este dataset estan con "?", hay que buscar alguna manera TODO
def clean_dataset(d):
    return 0


# Load the dataset
dataset = load_dataset('data/adult.data')
# clean_dataset = clean_dataset(dataset)


