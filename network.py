import numpy as np

class NeuralNetwork:
    def __init__(self, n, hidden_layers=1, hidden_size=10, classes=2):
        self.n = n
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.classes = classes

        first_layer_weights = np.random.random_sample((hidden_size, n+1)) * 2 - 1
        last_layer_weights = np.random.random_sample((classes, hidden_size+1)) * 2 - 1
        self.weights = [first_layer_weights]
        for i in range(hidden_layers - 1):
            layer_weights = np.random.random_sample((hidden_size, hidden_size+1)) * 2 - 1
            self.weights.append(layer_weights)
        self.weights.append(last_layer_weights)

        first_layer = np.zeros((n,))
        self.layers = [first_layer]
        for i in range(hidden_layers):
            layer = np.zeros((hidden_size,))
            self.layers.append(layer)
        last_layer = np.zeros((classes,))
        self.layers.append(last_layer)
        

    def forward_propagation(self, x):
        a = x
        # repeat for each layer
        for i, weight in enumerate(self.weights):
            self.layers[i] = a.copy()
            # Add bias
            a = np.insert(a, 0, 1)
            a = self.sigmoid(weight @ a)
        self.layers[-1]  = a.copy()


    def backward_propagation(self, x, y):
        pass

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, x, y):
        # TODO: add regularization
        m = x.shape[0]
        j = 0
        for i in range(m):
            self.forward_propagation(x[i])
            j -= 1 / m * (y[i].dot(np.log(self.layers[-1])) + (1 - y[i]).dot(np.log(1 - self.layers[-1]))).sum()
        return j

    def gradient_check(self, x, y):
        pass

    def train(self, x, y):
        pass

    def predict(self, x):
        pass

n = NeuralNetwork(1)
print(n.cost(np.array([[2], [3], [4]]), np.array([[0,1], [1,0], [0,1]])))