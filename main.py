import pandas as pd
import numpy as np
# from ID3 import ID3
from Tree import *
import time
import math
# To avoid pandas error/warning in cut/qcut
pd.options.mode.chained_assignment = None


# Function to read data
def load_dataset(path):
    # Add the columns headers, see adult.names. 15 attributes
    head = ['Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum', 'MaritalStatus', 'Occupation', 'Relationship',
            'Race', 'Sex', 'CapitalGain', 'CapitalLoss', 'HPW', 'NativeCountry', 'Income']
    df = pd.read_csv(path, header=None, names=head, delimiter=', ', engine='python')
    return df


# Function to clean the data
def clean_dataset(df, advanced):
    # For the advanced settings, we consider the "?" values as a possible value
    if not advanced:
        # Delete all the "?" values
        df = df[~df.eq('?').any(1)]

    # Treatment of continuous variables
    # We use pd.cut() because we can set the division of the labels
    df['Age'] = pd.cut(df['Age'], bins=[16, 29, 39, 49, 65, 91], labels=['Jove', 'Adult', 'AdultMadur', 'Gran', 'Avi'])
    df['HPW'] = pd.cut(df['HPW'], bins=[0, 1, 20, 40, 100], labels=['SenseFeina', 'MitjaJornada', 'JornadaCompleta', 'MesHores'])
    df['EducationNum'] = pd.cut(df['EducationNum'], bins=[0, 6, 9, 13, 17], labels=['Baix', 'Mig', 'Alt', 'MoltAlt'])
    df['CapitalGain'] = pd.cut(df['CapitalGain'], bins=[-0.1, 3500, 10000, 25000, 100000], labels=['Baix', 'Mig', 'Alt', 'MoltAlt'])
    df['CapitalLoss'] = pd.cut(df['CapitalLoss'], bins=[-0.1, 1000, 2000, 3000, 10000], labels=['Baix', 'Mig', 'Alt', 'MoltAlt'])

    # Here we group the countries by continents
    # Maybe add a new attribute continent and compare
    df.loc[df["NativeCountry"].isin(['United-States', 'Canada', 'Columbia']), "NativeCountry"] = 'NorthAmerica'
    df.loc[df["NativeCountry"].isin(['Ecuador', 'Puerto-Rico', 'Cuba', 'Honduras', 'Jamaica', 'Mexico',
                                     'Dominican-Republic', 'Haiti', 'Guatemala','Nicaragua', 'El-Salvador',
                                     'Trinadad&Tobago', 'Peru']), "NativeCountry"] = 'SouthAmerica'
    df.loc[df["NativeCountry"].isin(['England', 'Germany', 'Greece', 'Italy', 'Poland', 'Portugal', 'Ireland',
                                     'France', 'Hungary', 'Scotland', 'Yugoslavia', 'Holand-Netherlands']), "NativeCountry"] = 'Europe'
    df.loc[df["NativeCountry"].isin(['South']), "NativeCountry"] = 'Africa'
    df.loc[df["NativeCountry"].isin(['Cambodia', 'India', 'Japan', 'China', 'Iran', 'Vietnam', 'Laos', 'Taiwan',
                                     'Thailand', 'Hong', 'Outlying-US(Guam-USVI-etc)', 'Philippines']), "NativeCountry"] = 'Asia&Oceania'

    # We drop some attributes that we think are redundant to predict the income: 12 attributes
    # fnlwgt because it is used to see if two samples have similar characteristics and we don't need
    # Education, because we can see how far they have come in the studies with EducationNum
    # Relationship, because we find more important the attribute marital-status
    return df.drop(['fnlwgt', "Education", "Relationship"], axis="columns")


def main():
    # Load the dataset
    dataset = load_dataset('data/adult.data')

    # Process dataset
    data = clean_dataset(dataset, advanced=True)
    train, data = split_data(data[:100])
    print('hola')
    """
    # ID3
    start = time.time()
    # Criteria: 'e' for entropy, 'g' for gini
    tree = ID3(data, criteria='e')
    end = time.time()
    print((end - start) / 60)


    # C.45
    start = time.time()
    # Criteria: 'e' for entropy, 'g' for gini
    tree = C45(data, criteria='e')
    end = time.time()
    print((end - start) / 60)
    """

main()

