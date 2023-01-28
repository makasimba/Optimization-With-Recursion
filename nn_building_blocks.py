import numpy as np
import matplotlib.pyplot as plt

g = np.random.default_rng(20)


def initialize_parameters(layers):
    """
    Arguments:
    layers -- iterable with the dimensions of each layer of the NN.

    Returns:
    parameters -- python dictionary containing model parameters "W1", "b1", ..., "WL", "bL"
    """
    parameters = {}
    L = len(layers)

    for l in range(1, L):
        n_l, n_p = layers[l], layers[l-1]
        parameters['W%d' % l] = g.normal(
            size=(n_l, n_p), scale=1.0/np.sqrt(n_p))
        parameters['b%d' % l] = np.zeros((n_l, 1))
    return parameters


def relu(Z):
    return np.maximum(0, Z)


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def forward_propagate(A, W, b, g):
    """
    Implements forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    g -- a non-linear activation function

    Returns:
    A -- activation for next layer 
    """
    Z = W.dot(A) + b
    return g(Z), Z


def compute_cost(AL, Y, m):
    return (-1 / m) * np.sum((Y * np.log(AL)) + ((1-Y) * np.log(1-AL)))


def sigmoid_derivative(Z):
    A = sigmoid(Z)
    return A * (1 - A)


def relu_derivative(Z):
    return np.array(Z > 0, dtype=np.int8)


def tanh_derivative(z):
    a = np.tanh(z)
    return 1 - a ** 2


def weights_for_layer(l, parameters):
    return parameters['W' + str(l)], parameters['b' + str(l)]


def get_activation_function_for_layer(l, hyperparameters):
    activation_functions = hyperparameters['activation_functions']

    if l == hyperparameters['L']:
        g = activation_functions['sigmoid']
    else:
        g = activation_functions['relu']

    return g


def back_propagate(dA_next, A_prev, Z, W, l, L, m, parameters):
    derivative = sigmoid_derivative if l == L else relu_derivative
    g_prime = derivative(Z)

    dZ = dA_next * g_prime
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    parameters['dW' + str(l)] = dW
    parameters['db' + str(l)] = db

    if l != 1:
        return np.dot(W.T, dZ)


def propagate(A_prev, Y, l, parameters, hyperparameters):

    L = hyperparameters.get('L')
    m = np.shape(Y)[1]

    if l == L+1:
        parameters['J'] = compute_cost(A_prev, Y, m)
        dA = np.divide(-Y, A_prev) + np.divide(1-Y, 1-A_prev)
        return dA
    else:
        W, b = weights_for_layer(l, parameters)
        g = get_activation_function_for_layer(l, hyperparameters)
        A, Z = forward_propagate(A_prev, W, b, g)
        dA = propagate(A, Y, l+1, parameters, hyperparameters)
    return back_propagate(dA, A_prev, Z, W, l, L, m, parameters)


def update(parameters, hyperparameters):
    L = hyperparameters['L']
    alpha = hyperparameters['learning_rate']
    for l in range(1, L+1):
        parameters['W' + str(l)] -= alpha * parameters['dW' + str(l)]
        parameters['b' + str(l)] -= alpha * parameters['db' + str(l)]


def optimize(A, Y, parameters, hyperparameters):
    costs = list()
    for t in range(hyperparameters.get('n_iterations')):
        propagate(A, Y, 1, parameters, hyperparameters)
        update(parameters, hyperparameters)
        if t % 100 == 0:
            costs.append(parameters.get('J'))
    return parameters, costs


def plot_cost(costs):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title('J')
    plt.show()


def nn_model(A, Y, hyperparameters):
    parameters = initialize_parameters(hyperparameters.get('nn_structure'))
    parameters, costs = optimize(A, Y, parameters, hyperparameters)
    #plot_cost(costs, hyperparameters.get('learning_rate'))
    return parameters


def predict(parameters, hyperparameters, A, Y):
    L = hyperparameters.get('L')
    w_and_b = [(parameters.get('W%d' % l), parameters.get('b%d' % l))
               for l in range(1, L+1)]
    *hidden_layer_w_and_b, last_layer_w_and_b = w_and_b

    for W, b in hidden_layer_w_and_b:
        A = relu(np.dot(W, A) + b)

    W, b = last_layer_w_and_b
    y_hat = sigmoid(W.dot(A) + b)

    pred = y_hat > 0.5
    accuracy = np.mean(pred == Y)

    return y_hat, pred, accuracy


# how to define the hyperparameters for the model
hyperparameters = {
    'learning_rate': 0.002,
    'nn_structure': (2, 4, 4, 1),
    'n_iterations': 5_000,
    'activation_functions': {'relu': relu, 'sigmoid': sigmoid},
    'L': 3,
    'learning_rate': 0.1,
}
