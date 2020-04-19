import pandas as pd
from numpy import nan
from pandas import DataFrame
from seaborn import load_dataset
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLWarmup:

    def getPlanetsDataset(self):
        planets_df = load_dataset("planets")
        print(planets_df)
        column_names = ['method', 'number', 'orbital_period', 'mass', 'distance', 'year']
        # ilość wartości pustych w kolumnach
        print(planets_df.isnull().sum())
        # usuń wszystkie kolumny zawierające więcej niż połowę wartości pustych - 1035
        half_dataset_no = int(len(planets_df)/2)
        planets_df1 = planets_df
        for c in column_names:
            if(planets_df[c].isnull().sum() > half_dataset_no):
                planets_df1 = planets_df1.drop(c,axis=1)
        print(planets_df1.isnull().sum())
        # usuń wszystkie te wiesze kóre zawierają ponad połowę wartości pustych -> planets_df
        planets_df2 = planets_df
        planets_df2 = planets_df2.dropna(thresh=4)
        print(planets_df2)
        # uzupełnianie pustych danych
        impFreq = SimpleImputer(missing_values=nan, strategy='most_frequent')
        planets_df3 = impFreq.fit_transform(planets_df)
        planets_df3 = DataFrame(planets_df3, columns=list(planets_df.columns))
        print(planets_df3.isnull().sum())
        print(planets_df3)
        # mapowanie danych jakościowych na liczby porządkowe
        planets_df4 = planets_df
        method_mapper = {}
        for index, num_category in enumerate(planets_df.method.unique()):
            method_mapper[num_category] = index
        print(method_mapper)
        planets_df4['method'] = planets_df['method'].map(method_mapper)
        # print(planets_df4)
        # uzupełnienie wartości pustych
        impMean = SimpleImputer(missing_values=nan, strategy='mean')
        planets_df4 = impMean.fit_transform(planets_df)
        planets_df4 = DataFrame(planets_df4, columns=list(planets_df.columns))
        print(planets_df4.isnull().sum())
        print(planets_df4)
        # skalowanie danych
        std = StandardScaler()
        planets_df4 = std.fit_transform(planets_df4)
        planets_df4 = DataFrame(planets_df4, columns=list(planets_df.columns))
        print(planets_df4)
        corr = planets_df4.corr()
        pd.set_option('display.max.columns', None)
        print(corr)

    def getIrisDataset(self):
        self.iris = load_iris()
        index = 0
        while(index < len(self.iris['target'])):
            print(self.iris['data'][index], self.iris['target'][index], self.iris['target_names'][self.iris['target'][index]],
                  sep=' | ')
            index += 1
    def splitDataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self.iris['data'], self.iris['target'], train_size=0.6)
        index = 0
        print("TRENINGOWY", len(X_train))
        while (index < len(y_train)):
            print(X_train[index], y_train[index],sep=' | ')
            index += 1

ml = MLWarmup()
ml.getIrisDataset()
ml.splitDataset()
# ml.getPlanetsDataset()