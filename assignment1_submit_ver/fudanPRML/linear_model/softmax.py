import numpy as np

def softmax(X):
    result = None
    
    b = np.max(X, axis=1)
    b = np.reshape(b, [b.shape[0],1])
    x = X - b
    sums = np.sum(np.exp(x), axis=1)
    sums = np.reshape(sums, [sums.shape[0],1])
    result = np.exp(x) / sums

    return result

class SoftmaxClassifier(object):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxClassifier, self).__init__()
        self.W = np.zeros(shape=[input_dim+1, output_dim])

        self.X = None
        self.outputs = None
        self.dW = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        N,D = X.shape
        result = None
        
        b = np.ones([N, 1])
        X = np.concatenate([X,b], axis=1)
        f_X = np.matmul(X, self.W)
        result = softmax(f_X)
        self.outputs = result
        self.X = X

        return result
    
    def backward(self, labels):
        self.dW = -np.matmul(self.X.T, labels-self.outputs) / len(labels)

        assert self.dW.shape == self.W.shape        
