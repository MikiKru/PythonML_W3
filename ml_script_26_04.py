import seaborn
import pandas as pd
from numpy import nan
from pandas import get_dummies
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class Classifiers:
    def splitDatasetIntoTrainAndTest(self, X, y, train_split_percent = 0.6):
        # pd.set_option('display.max_columns', None)
        # print(X)
        print(X.info())
        # print(X.describe())
        # print(X.describe(include=[pd.np.number]))
        # print(X.describe(include=[pd.np.object]))
        # print(X.describe(include=['category']))
        # print(X.describe(include={'boolean'}))
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_split_percent)
        return X_train, X_test, y_train, y_test
    def datasetPreprocessing(self, X, columns_to_drop, columns_to_map,
                             nan_to_median_columns, nan_to_most_freq_columns):
        # usuwanie
        X_clean = X.drop(columns_to_drop, axis=1)
        # mapowanie
        for column_name in columns_to_map:
            # konstruowanie mappera
            mapper = {}
            for index, category in enumerate(X_clean[column_name].unique()):
                mapper[category] = index
            # mapowanie
            X_clean[column_name] = X_clean[column_name].map(mapper)
        # uzupełnianie
        for column_name in nan_to_median_columns:
            X_clean[column_name] = X_clean[column_name].fillna(X_clean[column_name].median())
        # for column_name in nan_to_most_freq_columns:
        #     impFreq = SimpleImputer(missing_values=nan, strategy='most_frequent')
        #     X_clean[column_name] = impFreq.fit_transform(X_clean[column_name])
        print(X_clean.isnull().sum())
    def trainAndTestClassifier(self, clf, X_train, X_test, y_train):
        # trenowanie
        clf.fit(X_train, y_train)
        # testowanie
        y_pred = clf.predict(X_test)
        return y_pred

c = Classifiers()
c.datasetPreprocessing(
    X = seaborn.load_dataset("titanic").iloc[:, 1:],
    columns_to_drop = ['sex','embarked','class','adult_male','deck','alive'],
    columns_to_map = ['who','embark_town', 'alone'],
    nan_to_median_columns = ['age'],
    nan_to_most_freq_columns = ['embark_town']
    )
# X_train, X_test, y_train, y_test = c.splitDatasetIntoTrainAndTest(
#     X=seaborn.load_dataset("titanic").iloc[:, 1:],
#     y=seaborn.load_dataset("titanic")['survived'])
# y_pred_knn5 = c.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5), X_train,X_test,y_train)
# print(y_pred_knn5)
# y_pred_tree = c.trainAndTestClassifier(DecisionTreeClassifier(), X_train,X_test,y_train)
# print(y_pred_tree)