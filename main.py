import numpy as np
import pandas as pd
from network import NeuralNetwork

def main():
    data = pd.read_csv('Iris.csv')
    mapping = {
        'Iris-setosa': np.array([1, 0, 0]),
        'Iris-versicolor': np.array([0, 1, 0]),
        'Iris-virginica': np.array([0, 0, 1])
    }
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    x = data.iloc[:, :4].values
    y = np.array([mapping[d] for d in data['label'].values])

    train_x, train_y, test_x, test_y = split_data(x, y, ratio=0.4, seed=1)
    nn = NeuralNetwork(x.shape[1], outputs=y.shape[1], hidden_layers=3,
                       hidden_size=6, _lambda=0.01)
    nn.train(train_x, train_y)
    predictions = nn.predict(test_x)
    predictions = np.array([mapping[classes[val]] for val in np.argmax(predictions, axis=1)])
    print((predictions != test_y).sum())

def split_data(x, y, ratio=0.3, seed=None):
    train_count = int(ratio * x.shape[0])
    combined = list(zip(x, y))
    if seed:
        np.random.seed(seed)
    np.random.shuffle(combined)
    train, test = combined[:train_count], combined[train_count:]
    train_x = np.array(list(zip(*train))[0])
    train_y = np.array(list(zip(*train))[1])
    test_x = np.array(list(zip(*test))[0])
    test_y = np.array(list(zip(*test))[1])
    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    main()