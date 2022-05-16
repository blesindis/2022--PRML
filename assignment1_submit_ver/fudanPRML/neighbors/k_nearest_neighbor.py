from dis import dis
import imp


import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distance_no_loop(X)
        return self.predict_labels(dists, k=k)

    def compute_distance_no_loop(self, X):
        dists = np.zeros((X.shape[0], self.X_train.shape[0]))
        for i in range(X.shape[0]):
            subs = np.subtract(X[i], self.X_train)
            subs = np.linalg.norm(subs, axis=1)
            dists[i] = subs
        return dists

    
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            closest_y = []
            sort_dists = np.argsort(dists[i])
            for j in range(k):
                closest_y.append(self.y_train[sort_dists[j]])
            closest_y = np.array(closest_y)
            
            if closest_y.size > 1:
                unique_y = np.unique(closest_y)
                count_y = []
                for data in unique_y:
                    count_y.append(np.sum(closest_y==data))
                sort_count = np.argsort(np.array(count_y))                
                y_pred[i] = unique_y[sort_count[-1]]
            else:
                y_pred[i] = closest_y
            
        return y_pred
