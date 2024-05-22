class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # 1. Initialize weights and bias
        self.weights = [0] * num_features
        self.bias = 0

        # Ensure y consists of 1 or -1
        y_ = [1 if i > 0 else -1 for i in y]

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (sum(x_i * self.weights) + self.bias) >= 0
                if not condition:
                    update = self.learning_rate * y_[idx]
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        y_pred = []
        for x_i in X:
            y_pred.append(1 if sum(x_i * self.weights) + self.bias >= 0 else 0)
        return y_pred


# Toy dataset: OR gate
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1]

# Train the Perceptron
clf = Perceptron(learning_rate=0.01, n_iterations=1000)
clf.fit(X, y)

# Predict
predictions = clf.predict(X)
print(predictions)  # Should approximate the OR gate: [0, 1, 1, 1]
