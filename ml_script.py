import pandas as pd
from pandas import np, DataFrame
from seaborn import load_dataset
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer


class MLWarmup:
    def getIrisDataset(self):
        iris = load_iris()
        index = 0
        while(index < len(iris['target'])):
            print(iris['data'][index], iris['target'][index], iris['target_names'][iris['target'][index]],
                  sep=' | ')
            index += 1
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
        impFreq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
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
        impMean = SimpleImputer(missing_values=np.nan, strategy='mean')
        planets_df4 = impMean.fit_transform(planets_df)
        planets_df4 = DataFrame(planets_df4, columns=list(planets_df.columns))
        print(planets_df4.isnull().sum())
        print(planets_df4)
        # skalowanie danych

ml = MLWarmup()
# ml.getIrisDataset()
ml.getPlanetsDataset()