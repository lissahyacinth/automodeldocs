import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Load dataset (using Boston housing data as an example)
data = load_boston()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Define hyperparameters grid to search
param_grid = {
    "num_leaves": [31, 50, 100],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [20, 40],
    "min_child_samples": [5, 10, 20],
    "boosting_type": ["gbdt", "dart"],
    "objective": ["regression"],
    "metric": ["l2"],
}

# Create a LightGBM regressor
lgbm = lgb.LGBMRegressor()

# Create the grid search with 5-fold cross-validation
grid_search = GridSearchCV(lgbm, param_grid, cv=5, scoring="neg_mean_squared_error")

grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters found: ", grid_search.best_params_)

# Predict and evaluate
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE: ", mse)
