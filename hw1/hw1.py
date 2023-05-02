###### Your ID ######
# ID1: 311146021
# ID2: 807724
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = (y - y.mean(axis=0)) / (y.max(axis=0) - y.min(axis=0))

    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.
    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    X = np.vstack((np.ones(X.shape[0]), X.transpose())).transpose()

    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    J = 0  # We use J for the cost.
    hypothesis = X @ theta
    m = len(y)
    J = np.sum((hypothesis - y) ** 2) / (2 * m)

    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    theta = theta.copy()
    J_history = []
    m = len(y)
    for i in range(num_iters):
        prediction = X @ theta
        theta = theta - (alpha / m) * ((prediction - y) @ X)
        J_history.append(compute_cost(X, y, theta.copy()))

    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    pinv_theta = []
    A = X.T.dot(X)
    U, D, V = np.linalg.svd(A)

    D_plus = np.zeros((A.shape[0], A.shape[1])).T
    D_plus[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))

    A_plus = V.T.dot(D_plus).dot(U.T)
    pinv_X = A_plus @ X.T
    pinv_theta = pinv_X @ y

    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration

    m = len(y)
    for i in range(num_iters):
        prediction = X @ theta
        theta = theta - (alpha / m) * ((prediction - y) @ X)
        J_history.append(compute_cost(X, y, theta.copy()))
        if (i > 1) and ((J_history[-2] - J_history[-1]) <= 1e-8):
            break

    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    np.random.seed(42)
    temp_theta = np.random.random(size=2)
    for i in alphas:
        np.random.seed(42)
        _, new_cost = efficient_gradient_descent(X_train, y_train, temp_theta, i, iterations)
        alpha_dict[i] = new_cost[-1]

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []

    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)
    features = set(range(1, X_train.shape[1]))
    for _ in range(5):
        min_cost, best_f = float('inf'), None
        for i in features:
            curr_features = selected_features + [i]
            np.random.seed(42)
            theta = np.random.random(size=len(curr_features))
            temp_cost = compute_cost(X_val[:, curr_features], y_val,
                                     efficient_gradient_descent(X_train[:, curr_features], y_train, theta, best_alpha,
                                                                iterations)[0])
            if temp_cost < min_cost:
                min_cost, best_f = temp_cost, i
        features.remove(best_f)
        selected_features.append(best_f)
    selected_features = [f - 1 for f in selected_features]

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()

    n_features = df.shape[1]
    # Loop over all pairs of features
    for i in range(n_features):
        for j in range(i, n_features):   # Multiply the two features together
            new_feature = np.multiply(df_poly.iloc[:, i], df_poly.iloc[:, j])

            # Add a column for the new feature
            feature_name = f"{df.columns[i]}*{df.columns[j]}"
            df_poly[feature_name] = new_feature

    return df_poly
