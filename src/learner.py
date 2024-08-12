import numpy as np
import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def binary_accuracy(Y, Y_hat, theta=0.5):
    """
    Calculates accuracy for using right side of .5.
    
    Parameters
    ----------
    Y: numpy.ndarray
        The targets.
    Y_hat: numpy.ndarray
        These are the predicted values.
    theta: float
        This value is the threshold to decide binary accuracy per unit on the output layer.
    
    Returns
    -------
    numpy.ndarray
        The accuracy for each example. This can be used as loss (but we will just use it as accuracy).
    """
   # Convert predictions to binary by checking if they are greater than 0.5
    preds = tf.cast(Y_hat > theta, tf.float32)
    
    # Compare predictions to the true values
    correct = tf.equal(preds, Y)
    
    # Calculate the accuracy as the mean of correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    return accuracy




def batch_accuracy(Y, Y_hat, theta=0.5, dichotomous=True):
    """
    Calculates accuracy for each example quickly.
    
    Parameters
    ----------
    Y: numpy.ndarray
        The targets.
    Y_hat: numpy.ndarray
        These are the predicted values.
    theta: float
        This value is the threshold to decide binary accuracy per unit on the output layer.
    dichotomous: bool
        Should the accuracies be dichotomous (default) or continuous (as the mean over units)
    
    Returns
    -------
    numpy.ndarray
        The accuracy for each example.
    
    
    """
    # Threshold predictions
    preds = (Y_hat > theta).astype(int)
    
    if dichotomous:
        accuracies = np.all(preds == Y, axis=1)
    if not dichotomous:
        accuracies = np.mean(preds == Y, axis=1)

    return accuracies




def learner(X, Y, seed, hidden, optimizer=None):

    tf.random.set_seed(seed)

    if optimizer is None:
        optimizer = 'adam' # Adam(learning_rate=learning_rate) # was originally .1

    model = Sequential()
    model.add(Dense(hidden, input_shape=(X.shape[1], )))
    model.add(Dense(Y.shape[1], activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[binary_accuracy, 'mse'])
    
    return model


