import numpy as np
from planar_utils import sigmoid

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (number of features, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- the size of hidden layer
    
    Returns:
    n_x -- the size of input layer
    n_h -- the size of hidden layer
    n_y -- the size of output layer
    """
    n_x = X.shape[0]
    n_h = 4 # default size of hidden layer is 4
    n_y = Y.shape[0]
    
    return(n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    n_x -- the size of input layer
    n_h -- the size of hidden layer
    n_y -- the size of output layer
    
    Returns:
    params -- dictionary containing:
                W1 -- weight matrix of shape (n_h, n_x)
                b1 -- bias vector of shape (n_h, 1)
                W2 -- weight matrix of shape (n_y, n_h)
                b2 -- bias vector of shape (n_y, 1)
    """
    # np.random.seed(2)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return params


def forward_propagation(X, params):
    """
    Arguments:
    X -- input dataset of shape (number of features, number of examples)
    params -- dictionary containing:
                W1 -- weight matrix of shape (n_h, n_x)
                b1 -- bias vector of shape (n_h, 1)
                W2 -- weight matrix of shape (n_y, n_h)
                b2 -- bias vector of shape (n_y, 1)
                
    Returns:
    A2 -- sigmoid output of the second activation
    cache -- dictionary containing:
                Z1 -- (n_h, number of examples)
                A1 -- (n_h, number of examples)
                Z2 -- (n_y, number of examples)
                A2 -- (n_y, number of examples)
    """
    # Retrieve parameters from params
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    # Calculate Z1, A1, Z2 and A2
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1) # Use tanh as the first activation function
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    
    return A2, cache


def compute_cost(A2, Y, params):
    """
    Compute the cross-entropy cost.
    
    Arguments:
    A2 -- sigmoid output of the second activation
    Y -- labels of shape (1, number of examples)
    params -- dictionary containing:
                W1 -- weight matrix of shape (n_h, n_x)
                b1 -- bias vector of shape (n_h, 1)
                W2 -- weight matrix of shape (n_y, n_h)
                b2 -- bias vector of shape (n_y, 1)
    
    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    
    logprobs1 = -np.dot(Y, np.log(A2).T)
    logprobs2 = -np.dot(1-Y, np.log(1-A2).T)
    cost = 1/m * (logprobs1 + logprobs2)
    
    cost = np.asscalar(cost)
    return cost


def backward_propagation(params, cache, X, Y):
    """
    Arguments:
    params -- dictionary containing:
                W1 -- weight matrix of shape (n_h, n_x)
                b1 -- bias vector of shape (n_h, 1)
                W2 -- weight matrix of shape (n_y, n_h)
                b2 -- bias vector of shape (n_y, 1)
    cache -- dictionary containing:
                Z1 -- (n_h, number of examples)
                A1 -- (n_h, number of examples)
                Z2 -- (n_y, number of examples)
                A2 -- (n_y, number of examples)
    X -- input dataset of shape (number of features, number of examples)
    Y -- labels of shape (1, number of examples)
    
    Returns:
    grads -- dictionary containing the gradients dW1, db1, dW2 and db2
    """
    m = X.shape[1]
    
    W1 = params["W1"]
    W2 = params["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    
    return grads


def update_parameters(params, grads, learning_rate = 1.2):
    """
    Arguments:
    params -- dictionary containing:
                W1 -- weight matrix of shape (n_h, n_x)
                b1 -- bias vector of shape (n_h, 1)
                W2 -- weight matrix of shape (n_y, n_h)
                b2 -- bias vector of shape (n_y, 1)
    grads -- dictionary containing the gradients dW1, db1, dW2 and db2
    learning_rate -- learning rate
    
    Returns:
    params -- dictionary contraining updated parameters
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return params


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    params -- parameters learnt by the model, containing W1, b1, W2 and b2
    X -- input data of size (n_x, m)
    
    Returns:
    predictions -- vector of predictions of the model (red: 0 / blue: 1)
    """
    A2, cache = forward_propagation(X, parameters)

    predictions = (A2 > 0.5)
    
    return predictions