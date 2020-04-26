import seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Classifiers:
    def splitDatasetIntoTrainAndTest(self, X, y, train_split_percent = 0.6):
        # pd.set_option('display.max_columns', None)
        # print(X)
        # print(X.info())
        # print(X.describe())
        # print(X.describe(include=[pd.np.number]))
        # print(X.describe(include=[pd.np.object]))
        # print(X.describe(include=['category']))
        # print(X.describe(include={'boolean'}))
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_split_percent)
        return X_train, X_test, y_train, y_test

    def trainAndTestClassifier(self, clf, X_train, X_test, y_train):
        # trenowanie
        clf.fit(X_train, y_train)
        # testowanie
        y_pred = clf.predict(X_test)
        return y_pred

c = Classifiers()
X_train, X_test, y_train, y_test = c.splitDatasetIntoTrainAndTest(
    X=seaborn.load_dataset("titanic").iloc[:, 1:],
    y=seaborn.load_dataset("titanic")['survived'] )
print(X_train)
print(y_train)
# c.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5))