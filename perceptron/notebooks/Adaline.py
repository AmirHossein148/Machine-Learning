import numpy as np
from sklearn.metrics import accuracy_score
class Adaline:
    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def weighting(self, x):
        return np.dot(x, self.weights) + self.bias

    def activation(self, z):
        return z

    def predict(self, inputs):
        X = np.array(inputs, dtype=float)
        predictions = []
        for x in X:
            pred = self.activation(self.weighting(x))  
            predictions.append(1 if pred >= 0 else -1)
        return np.array(predictions)

    def fit(self, inputs, outputs, learning_rate=0.01, epochs=50):
        X = np.array(inputs, dtype=float)
        y = np.array(outputs)

        self.weights = np.random.uniform(-0.01, 0.01, size=X.shape[1])
        self.bias = 0.0

        for epoch in range(epochs):
            for i in range(len(X)):
                x_i = X[i]
                pred = self.activation(self.weighting(x_i))
                error = y[i] - pred
                self.weights += learning_rate * error * x_i
                self.bias += learning_rate * error

