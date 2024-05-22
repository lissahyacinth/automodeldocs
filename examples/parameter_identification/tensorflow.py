import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load dataset
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

# One-hot encoding
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)


# Function to create a Keras model
def create_model(optimizer="adam", neurons=10):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(y_onehot.shape[1], activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


# Create a Keras classifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10, verbose=0)

# Define hyperparameters grid to search
param_grid = {
    "optimizer": ["SGD", "Adam"],
    "neurons": [10, 20, 30],
    "batch_size": [5, 10, 20],
    "epochs": [10, 50],
}

# Grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
y_pred = grid_result.predict(X_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
print("Test accuracy:", accuracy)
