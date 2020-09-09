import numpy as np

class NeuralNetwork:
    EPSILON = 1e-6
    ALPHA = 0.5
    LAMBDA = 0.001
    def __init__(self, n, alpha=ALPHA, epsilon=EPSILON, _lambda=LAMBDA,
                 hidden_layers=1, hidden_size=10, classes=2):
        self.n = n
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.classes = classes

        self.alpha = alpha
        self.epsilon = epsilon
        self._lambda = _lambda
        self.mean = 0
        self.range = 1

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
    
    def normalize(self, x):
        self.mean = x.mean(axis=0)
        self.range = x.max(axis=0) - x.min(axis=0)
        return (x - self.mean) / self.range

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
        Delta = [np.zeros((weight.shape)) for weight in self.weights]
        m = x.shape[0]
        # TODO: rewrite this algorithm without the outer loop
        for i in range(m):
            self.forward_propagation(x[i])
            delta = self.layers[-1] - y[i]
            for l in range(len(self.layers) - 2, -1, -1):
                biased = np.insert(self.layers[l], 0, 1)
                Delta[l] += delta.reshape((delta.shape[0], 1)) @ biased.reshape((1, biased.shape[0]))
                delta = (self.weights[l].T @ delta)[1:,] * (self.layers[l] * (1 - self.layers[l]))

        derivatives = []
        for i, d in enumerate(Delta):
            derivative = 1 / m * d
            lambdas = np.array([0] + [self._lambda] * (derivative.shape[1] - 1))
            derivative += lambdas * self.weights[i]
            derivatives.append(derivative)
        return derivatives

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, x, y):
        # DONE: add regularization
        m = x.shape[0]
        j = 0
        for i in range(m):
            self.forward_propagation(x[i])
            j -= 1 / m * (y[i].dot(np.log(self.layers[-1])) + (1 - y[i]).dot(np.log(1 - self.layers[-1]))).sum()
            j += self._lambda / (2 * m) * sum([(weight[:,1:] ** 2).sum() for weight in self.weights])
        return j

    def gradient_check(self, x, y):
        delta = []
        epsilon = 1e-4
        for weight in self.weights:
            layer_delta = np.zeros(weight.shape)
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    weight[i, j] = weight[i, j] + epsilon
                    cost_after = self.cost(x, y)
                    weight[i, j] = weight[i, j] - 2 * epsilon
                    cost_before = self.cost(x, y)
                    weight[i, j] = weight[i, j] + epsilon
                    partial = (cost_after - cost_before) / (2 * epsilon)
                    layer_delta[i, j] = partial
            delta.append(layer_delta)
        return delta

    def train(self, x, y):
        x = self.normalize(x)
        last_cost = 0
        while True:
            new_cost = self.cost(x, y)
            if abs(new_cost - last_cost) < self.epsilon:
                break
            last_cost = new_cost

            derivatives = self.backward_propagation(x, y)
            for i, derivative in enumerate(derivatives):
                self.weights[i] -= self.alpha * derivative

    def predict(self, x):
        self.forward_propagation((x - self.mean) / self.range)
        return self.layers[-1]


n = NeuralNetwork(3, hidden_layers=3, hidden_size=10, classes=3)
x = np.array([[2, 1, 4], [3, 2, 6], [4, 7, 6]])
y = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
check = n.gradient_check(x, y) 
actual = n.backward_propagation(x, y)
error = sum([abs(check[i] - actual[i]).sum() for i in range(len(check))])
if error < 1e-6:
    print("Correct")
else:
    print("Value is off")
n.train(x, y)
print(n.predict(np.array([2, 1, 4])))
print(n.predict(np.array([3, 2, 6])))
print(n.predict(np.array([4, 7, 6])))