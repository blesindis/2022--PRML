import numpy as np

class Layer(object):
    def __init__(self, name):
        self.name = name
        self.params, self.grads = None, None # self.params保存需要更新的参数，self.grads保存对应的梯度值
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grads):
        raise NotImplementedError # 反向传播由后面子类实现


class Linear(Layer):
    def __init__(self, in_features, out_features, weight_init=np.random.standard_normal, bias_init=np.zeros):
        super().__init__(self.__class__.__name__)
        self.params = {}
        self.params['W'] = weight_init([in_features,out_features])
        self.params['b'] = bias_init([1,out_features])
        self.grads = {}
        self.name = 'Linear'

        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        outputs = np.matmul(inputs, self.params['W']) + self.params['b']
        z_max = np.max(outputs, axis=1)
        z_max = np.reshape(z_max, [z_max.shape[0],1])
        z_min = np.min(outputs, axis=1)
        z_min = np.reshape(z_min, [z_min.shape[0],1])
        outputs = (outputs - z_min) / (z_max - z_min)
        return outputs
    
    def backward(self, grads):
        self.grads['W'] = np.matmul(self.inputs.T, grads)
        self.grads['b'] = np.sum(grads, axis = 0)
        
        grads = np.matmul(grads, self.params['W'].T)
        return grads


class Logistic(Layer):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.name = 'Logistic'
        self.outputs = None
        self.inputs = None

    def forward(self, inputs):
        outputs = None
        mask1 = inputs > 0
        mask2 = inputs <= 0
        output1 = mask1 * inputs
        output1 = 1 / (1 + np.exp(-output1)) * mask1
        output2 = mask2 * inputs
        output2 = np.exp(output2) / (1 + np.exp(output2))  * mask2
        outputs = output1 + output2

        self.outputs = outputs
        return outputs

    def backward(self, grads):
        mid = self.outputs * (1 - self.outputs)
        grads = np.multiply(grads, mid)
        return grads


class Softmax(Layer):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.name = 'Softmax'
        self.outputs = None
        self.inputs = None

    def forward(self, inputs):
        b = np.max(inputs, axis=1)
        b = np.reshape(b, [b.shape[0],1])
        x = inputs - b
        sums = np.sum(np.exp(x), axis=1)
        sums = np.reshape(sums, [sums.shape[0],1])
        outputs = np.exp(x) / sums
        self.outputs = outputs
        self.inputs = inputs
        return outputs

    def backward(self, grads):
        grads = None
        return grads


class LeakyReLU(Layer):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.name = 'LeakyReLU'
        self.outputs = None
        self.inputs = None

    def forward(self, inputs):
        outputs = None
        mask1 = inputs > 0
        mask2 = inputs <= 0
        output1 = mask1 * inputs
        output2 = mask2 * inputs * 0.1
        outputs = output1 + output2
        self.outputs = outputs
        return outputs

    def backward(self, grads):
        mask1 = self.outputs > 0
        mask2 = self.outputs <= 0
        output1 = mask1 * grads
        output2 = mask2 * grads * 0.1
        return grads* (output1 + output2)


class MLPClassifier(object):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.act_fn1 = LeakyReLU()
        self.fc2 = Linear(hidden_dim, output_dim)
        self.act_fn2 = Softmax()

        self.layers = [self.fc1,self.act_fn1,self.fc2,self.act_fn2]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        z1 = self.fc1(X)  # (N, hidden_dim)
        a1 = self.act_fn1(z1) # (N, hidden_dim)
        z2 = self.fc2(a1) # (N,output_dim)
        a2 = self.act_fn2(z2) # (N, output_dim)
        return a2
    
    def backward(self, loss_grad_a2):
        grad_l2 = self.fc2.backward(loss_grad_a2) # ()
        grad_a1 = self.act_fn1.backward(grad_l2)
        grad_l1 = self.fc1.backward(grad_a1)
