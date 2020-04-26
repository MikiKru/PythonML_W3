import seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Classifiers:
    def splitDatasetIntoTrainAndTest(self, X, y, train_split_percent = 0.6):
        pd.set_option('display.max_columns', None)
        print(X)
        print(X.info())
        print(X.describe())
        print(X.describe(include=[pd.np.number]))
        print(X.describe(include=[pd.np.object]))
        print(X.describe(include=['category']))
        print(X.describe(include={'boolean'}))
        X_train, X_test, y_train, y_test = train_test_split(X,y)
        return X_train, X_test, y_train, y_test

    def trainAndTestClassifier(self, clf, X_train, X_test, y_train):
        # trenowanie
        clf.fit(X_train, y_train)
        # testowanie
        y_pred = clf.predict(X_test)
        return y_pred

c = Classifiers()
# for c in seaborn.load_dataset("titanic").columns:
#     print("'"+c+"'", end=',')
c.splitDatasetIntoTrainAndTest(
    X=seaborn.load_dataset("titanic")[{'pclass','sex','age','sibsp','parch','fare','embarked','class','who','adult_male','deck','embark_town','alive','alone'}],
    y=seaborn.load_dataset("titanic")['survived'] )
# c.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5))