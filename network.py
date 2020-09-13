import numpy as np

class NeuralNetwork:
    EPSILON = 1e-6
    ALPHA = 1
    LAMBDA = 0.001
    def __init__(self, inputs, alpha=ALPHA, epsilon=EPSILON, _lambda=LAMBDA,
                 hidden_layers=1, hidden_size=10, outputs=2):
        # Initializing network parameters
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.outputs = outputs

        # Initilaizing learning parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self._lambda = _lambda
        self.mean = 0
        self.range = 1

        # Initializing weights with random natural distribution
        first_layer_weights = np.random.randn(hidden_size, inputs + 1)
        # self.weights is a list of matrices, a matrix per layer
        self.weights = [first_layer_weights]
        for i in range(hidden_layers - 1):
            layer_weights = np.random.randn(hidden_size, hidden_size + 1)
            self.weights.append(layer_weights)
        last_layer_weights = np.random.randn(outputs, hidden_size + 1)
        self.weights.append(last_layer_weights)

        # Initializing layers with zeros
        first_layer = np.zeros((1, inputs))
        # self.layers is a list of matrices, a matrix per layer
        # In one matrix, each row correspinds to a test case
        self.layers = [first_layer]
        for i in range(hidden_layers):
            layer = np.zeros((1, hidden_size))
            self.layers.append(layer)
        last_layer = np.zeros((1, outputs))
        self.layers.append(last_layer)
    
    def normalize(self, x):
        self.mean = x.mean(axis=0)
        self.range = x.max(axis=0) - x.min(axis=0)
        return (x - self.mean) / self.range

    def forward_propagation(self, x):
        # Initialize first layer, Assume the number of inputs matches the network
        assert x.shape[1] == self.inputs
        self.layers[0] = x
        a = x

        # Repeat for each layer
        for i, weight in enumerate(self.weights):
            # Add bias
            a = np.insert(a, 0, 1, axis=1)
            # Evaluate layer
            a = self.sigmoid(a @ weight.T)
            self.layers[i + 1] = a

    def backward_propagation(self, x, y):
        # Initialize Delta matrices
        Delta = [np.zeros((weight.shape)) for weight in self.weights]
        m = x.shape[0]
        
        # Populate layers
        self.forward_propagation(x)

        # Calculate last layer's delta
        delta = self.layers[-1] - y

        # Reverse layers and weights, omit the last layer
        layers = self.layers[-2: :-1]
        weights = self.weights[-1: :-1]

        # Repeat for each layer
        for i, layer in enumerate(layers):
            # Acummelate Delta += delta(l+1) * a(l)
            biased = np.insert(layer, 0, 1, axis=1)
            Delta[-1 - i] += delta.T @ biased

            # Calculate new delta(l) to use on next iteration
            # Omits the bias weights
            # delta(l) = (delta(l + 1) * theta(l)) .* sigmoid'(z)
            delta = (delta @ weights[i])[:,1:] * (layers[i] * (1 - layers[i]))

        # Add regularization
        derivatives = []
        for i, d in enumerate(Delta):
            # Create regularization vector [0, lambda, lambda, lambda, ...]
            lambdas = np.array([0] + [self._lambda] * (d.shape[1] - 1))

            # D = 1 / m * Delta + lambda / m * theta
            derivative = 1 / m * d + lambdas / m * self.weights[i]
            derivatives.append(derivative)

        return derivatives

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, x, y):
        m = x.shape[0]
        # Populate layer with data
        self.forward_propagation(x)

        # Calculate cost function with regularization
        j = -1 / m * ((y * np.log(self.layers[-1])).sum() + ((1 - y) * np.log(1 - self.layers[-1])).sum())
        j += self._lambda / (2 * m) * sum([(weight[:, 1:] ** 2).sum() for weight in self.weights])
        return j

    def gradient_check(self, x, y):
        delta = []
        epsilon = 1e-4

        for weight in self.weights:
            layer_delta = np.zeros(weight.shape)
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    # For each layer, row and column
                    # Calculate j(theta(i,j,l) + epsilon) - j(theta(i,j,l) - epsilon)
                    #           / (2 * epsilon)
                    weight[i, j] = weight[i, j] + epsilon
                    cost_after = self.cost(x, y)

                    weight[i, j] = weight[i, j] - 2 * epsilon
                    cost_before = self.cost(x, y)
                    
                    partial = (cost_after - cost_before) / (2 * epsilon)
                    layer_delta[i, j] = partial

                    # Restore weights to the original value
                    weight[i, j] = weight[i, j] + epsilon
                    
            delta.append(layer_delta)

        return delta

    def train(self, x, y):
        x = self.normalize(x)
        last_cost = 0
        while True:
            new_cost = self.cost(x, y)
            # Divergence condition
            if abs(new_cost - last_cost) < self.epsilon:
                break
            last_cost = new_cost

            derivatives = self.backward_propagation(x, y)
            for i, derivative in enumerate(derivatives):
                self.weights[i] -= self.alpha * derivative

    def predict(self, x):
        self.forward_propagation((x - self.mean) / self.range)
        return self.layers[-1]

if __name__ == "__main__":
    # np.random.seed(1)
    n = NeuralNetwork(3, hidden_layers=5, hidden_size=10, outputs=4)
    # print("weights:")
    # print(*n.weights, sep="\n")
    # print()
    x = np.array([[2, 3 , 4], [1, 5, 2], [5, 6, 3], [4, 7, 8]])
    y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # print(f"cost {n.cost(x, y)}")

    check = n.gradient_check(x, y)
    actual = n.backward_propagation(x, y)
    # print(*check, sep='\n')
    # print(*actual, sep='\n')
    error = sum([abs(check[i] - actual[i]).sum() for i in range(len(check))])
    if error < 1e-6:
        print("Correct")
    else:
        print("Value is off")
    # n.train(x, y)
    # print(n.predict(np.array([2, 1, 4])))
    # print(n.predict(np.array([3, 2, 6])))
    # print(n.predict(np.array([4, 7, 6])))