from collections import Counter
import numpy as np

from fudanPRML.utils import is_numerical

class DTNode:
    def __init__(self, 
             feat_idx=None, 
             threshold=None,
             split_score=None,
             left=None, 
             right=None,
             value=None,
             leaf_num=None):
        
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.split_score = split_score
        self.value = value
        self.left = left
        self.right = right

        self.leaf_num = leaf_num


class BaseDecisionTree:
    def __init__(self, 
             criterion,
             max_depth,
             max_features,
             min_samples_split, 
             min_impurity_split,
             random_split,
             random_state):

        self.criterion = criterion
        self.max_depth = float("inf") if max_depth is None else max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.random_split = random_split
        self.random_state = random_state

        self.root = None # 树根
        self.feature_importances_ = None # 特征重要性
        self.feature_scores_ = None # 特征重要性（未归一化）

        self.tree_leaf_num = 0 # 整个树有多少叶子节点
        self.tree_depth = 0 # 树的深度

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.root = self._build_tree(X, y)
        # normalize feature scores
        self.feature_importances_ = (
            self.feature_scores_ / self.feature_scores_.sum())

    def _build_tree(self, X, y, curr_depth=1):
        n_samples, n_feats = X.shape
        self.feature_scores_ = np.zeros(n_feats, dtype=float)

        split_score = 0
        if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            split, split_score = self._split(X,y,self.random_split)

        leaf_num_before = self.tree_leaf_num
        if split_score > self.min_impurity_split:
            left = self._build_tree(split["l_X"], split["l_y"], curr_depth + 1)
            right = self._build_tree(split["r_X"], split["r_y"], curr_depth + 1)
            self.feature_scores_[split["feat_idx"]] += split_score
            return DTNode(feat_idx=split["feat_idx"], threshold=split["threshold"],
                          split_score = split_score, left=left, right=right,
                          leaf_num=self.tree_leaf_num-leaf_num_before)
        else:
            leaf_val = self._aggregation_func(y)
            self.tree_leaf_num += 1
            if curr_depth > self.tree_depth:
                self.tree_depth = curr_depth
            return DTNode(split_score= split_score, value=leaf_val,leaf_num=1)

    def _split(self, X, y, random_split=None):
        Xy = np.concatenate((X, y), axis=1)
        n_feats = X.shape[1]

        max_score = 0.0
        best_split = {"feat_idx": 0, "threshold": 0, "l_X": None, "r_X": None, "l_y": None, "r_y": None}

        k = self._get_n_feats(self.max_features, n_feats)
        if self.random_state != None:
            np.random.seed(self.random_state)
        cols = np.random.choice(range(n_feats), k, replace=False)
        
        for c in cols:
            value_c = np.sort(np.unique(X[:,c]))
            for i in range(value_c.shape[0]-1):
                thr = (int(value_c[i]) + int(value_c[i+1])) / 2
                mask_l = Xy[:,c] <= thr
                mask_r = Xy[:,c] > thr
                l_y = Xy[mask_l][:,-1]
                l_y = np.reshape(l_y,[l_y.shape[0], 1])
                r_y = Xy[mask_r][:,-1]
                r_y = np.reshape(r_y, [r_y.shape[0], 1])
                score = self._score_func(y, l_y, r_y)
                if score > max_score:
                    max_score = score
                    best_split = {"feat_idx": c, "threshold": thr, "l_X": Xy[mask_l][:,:-1], "r_X": Xy[mask_r][:,:-1], "l_y": l_y, "r_y": r_y}

        #TODO: random_split
        return best_split, max_score

    @staticmethod
    def _get_n_feats(max_feats, n_feats):
        if isinstance(max_feats, int):
            return max_feats
        elif isinstance(max_feats, float):
            return int(max_feats * n_feats)
        elif isinstance(max_feats, str):
            if max_feats == "sqrt":
                return int(np.sqrt(n_feats))
            elif max_feats == "log2":
                return int(np.log2(n_feats + 1))
        return n_feats

    def predict(self, X):
        if X.ndim == 1:
            return self._predict_sample(X)
        else:
            return np.array([self._predict_sample(sample) for sample in X])

    def _score_func(self, *args, **kwargs):
        raise NotImplementedError

    def _aggregation_func(self, *args, **kwargs):
        raise NotImplementedError


    def _predict_sample(self, x, node=None):
        if node is None:
            node = self.root

        if node.value is not None:
            return node.value

        feat = x[node.feat_idx]
        if is_numerical(feat):
            node = node.left if feat < node.threshold else node.right
        else:
            node = node.left if feat == node.threshold else node.right 
        return self._predict_sample(x, node=node)

    @staticmethod
    def _divide(data, col, thr):
        if is_numerical(thr):
            mask = data[:, col] < thr
        else:
            mask = data[: col] == thr
        return data[mask], data[~mask]


class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, 
             criterion="info_gain", 
             max_depth=None,
             max_features=None,
             min_samples_split=2,
             min_impurity_split=0.0,
             random_split=False,
             random_state=None):
        assert criterion in ("info_gain", "info_gain_ratio", "gini","error_rate")
        super().__init__(criterion, max_depth, max_features, 
                         min_samples_split, min_impurity_split, random_split,random_state)

    def _score_func(self, y, l_y, r_y):
        if self.criterion == "info_gain":
            return self.__info_gain(y, l_y, r_y)
        elif self.criterion == "info_gain_ratio":
            return self.__info_gain(y, l_y, r_y, with_ratio=True)
        elif self.criterion == "gini":
            return self.__gini_index(y, l_y, r_y)
        elif self.criterion == "error_rate":
            return self.__error_rate(y, l_y, r_y)

    @staticmethod
    def __info_gain(y, l_y, r_y, with_ratio=False):
        label_y = np.unique(y)
        prob_y = []
        for label in label_y:
            prob_y.append(np.sum(y==label) / y.shape[0])
        prob_y = np.array(prob_y)
        entropy_y = - np.sum(np.multiply(prob_y, np.log2(prob_y)))

        label_ly = np.unique(l_y)
        prob_ly = []
        for label in label_ly:
            prob_ly.append(np.sum(l_y==label) / l_y.shape[0])
        prob_ly = np.array(prob_ly)
        entropy_ly = - np.sum(np.multiply(prob_ly, np.log2(prob_ly)))

        label_ry = np.unique(r_y)
        prob_ry = []
        for label in label_ry:
            prob_ry.append(np.sum(r_y==label)/r_y.shape[0])
        prob_ry = np.array(prob_ry)
        entropy_ry = - np.sum(np.multiply(prob_ry, np.log2(prob_ry)))

        gain_split = entropy_y - l_y.shape[0]/y.shape[0] * entropy_ly - r_y.shape[0]/y.shape[0] * entropy_ry
        if with_ratio == False:
            info_gain = gain_split
        else:
            temp = np.array([l_y.shape[0], r_y.shape[0]])
            split_info = - np.sum(np.multiply(temp/y.shape[0], np.log2(temp/y.shape[0])))
            info_gain = gain_split / split_info

        return info_gain
        
    @staticmethod
    def __gini_index(y, l_y, r_y):
        before, after = -1, -1

        label_y = np.unique(y)
        prob_y = []
        for label in label_y:
            prob_y.append(np.sum(y==label)/y.shape[0])
        prob_y = np.array(prob_y)
        gini_y = 1 - np.sum(np.square(prob_y))

        label_ly = np.unique(l_y)
        prob_ly = []
        for label in label_ly:
            prob_ly.append(np.sum(l_y==label)/l_y.shape[0])
        prob_ly = np.array(prob_ly)
        gini_ly = 1 - np.sum(np.square(prob_ly))

        label_ry = np.unique(r_y)
        prob_ry = []
        for label in label_ry:
            prob_ry.append(np.sum(r_y==label)/r_y.shape[0])
        prob_ry = np.array(prob_ry)
        gini_ry = 1 - np.sum(np.square(prob_ry))

        before = gini_y
        after = gini_ry + gini_ly
        return before - after

    @staticmethod
    def __error_rate(y, l_y, r_y):
        before, after = -1, -1

        label_y = np.unique(y)
        prob_y = np.array([])
        for label in label_y:
            prob_y = np.append(prob_y, np.sum(y==label)/y.shape[0])
        error_y = 1 - prob_y[np.argmax(prob_y)]

        label_ly = np.unique(l_y)
        prob_ly = np.array([])
        for label in label_ly:
            prob_ly = np.append(prob_ly, np.sum(l_y==label)/l_y.shape[0])
        error_ly = 1 - prob_ly[np.argmax(prob_ly)]

        label_ry = np.unique(r_y)
        prob_ry = np.array([])
        for label in label_ry:
            prob_ry = np.append(prob_ry, np.sum(r_y==label)/r_y.shape[0])
        error_ry = 1 - prob_ry[np.argmax(prob_ry)]

        before = error_y
        after = error_ly + error_ry
        
        return before - after

    def _aggregation_func(self, y):
        res = Counter(y.reshape(-1))
        return int(res.most_common()[0][0])