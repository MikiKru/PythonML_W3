from seaborn import load_dataset
from sklearn.datasets import load_iris

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
        # print(planets_df)
        # ilość wartości pustych w kolumnach
        print(planets_df.isnull().sum())

ml = MLWarmup()
# ml.getIrisDataset()
ml.getPlanetsDataset()