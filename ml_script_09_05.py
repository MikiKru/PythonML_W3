# klasteryzacja
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


class ClusteringAlgorithms:
    def getIris(self):
        self.iris = load_iris()
        # print(self.iris)
    def clustering(self, cls):
        y_pred = cls.fit_predict(self.iris['data'])
        print(y_pred)
        clusters = [list(y_pred).index(0), list(y_pred).index(1), list(y_pred).index(2)]
        if(clusters[0] < clusters[1] and clusters[0] < clusters[2]):
            clusters[0] = 0
            if(clusters[1] < clusters[2]):
                clusters[1] = 1
                clusters[2] = 2
            else:
                clusters[1] = 2
                clusters[2] = 1
        elif(clusters[1] < clusters[0] and clusters[1] < clusters[2]):
            clusters[1] = 0
            if (clusters[0] < clusters[2]):
                clusters[0] = 1
                clusters[2] = 2
            else:
                clusters[0] = 2
                clusters[2] = 1
        elif (clusters[2] < clusters[0] and clusters[2] < clusters[1]):
            clusters[2] = 0
            if (clusters[0] < clusters[1]):
                clusters[0] = 1
                clusters[1] = 2
            else:
                clusters[0] = 2
                clusters[1] = 1


        clusters_ordered = sorted(clusters)
        print(clusters)
        print(clusters_ordered)
        y_cls = []
        index = 0
        while(index < len(y_pred)):
            print(y_pred[index])
            cls_index = clusters.index(y_pred[index])
            print(cls_index)
            y_cls.append(clusters_ordered[cls_index])

            index += 1
        print(y_cls)
        # print(accuracy_score(self.iris['target'], y_pred))

c = ClusteringAlgorithms()
c.getIris()
c.clustering(KMeans(n_clusters=3, init='random'))
# c.clustering(KMeans(n_clusters=3, init='k-means++'))

