
import math
import numpy as np

def scale(x, K):
    return K*math.log(x)


def subset_kidwords(subset, words, X, Y, remove_null_columns = True):

    """Generate a subset of X, Y patterns from kidwords given a list of words.

    
    Parameters
    ----------
    subset : list
        A list of words that you'd like to subset from parameter words in order to return X, Y patterns.
    words : list
        The words from which you'd like to subset X, Y patterns. The ith word in words must correspond
        to the ith pattern in X and Y along axis 0.
    X : array
        A 2D array representing the X patterns (orthography). The ith pattern along axis 0 must correspond
        to the ith word in words for things to work properly.
    Y : array
        A 2D array representing the Y patterns (phonology). The ith pattern along axis 0 must correspond
        to the ith word in words for things to work properly.        

    
    Notes
    ----_
    words, X, and Y are all ordered in the same manner such that the ith element in one corresponds to
    the ith element in each of the others. Therefore len(words) == X.shape[0] == Y.shape[0]

    Returns
    -------
    tuple
        Three elements are returned: the first is the the subset words reordered such that the index of
        each word matches its corresponding element (axis 0) in the reteurned X and Y patterns. The second 
        is the subset of X patterns (based on subset parameter) and the third is the subset of Y patterns, 
        based on the subset parameter.

    """

    indices = [i for i, e in enumerate(words) if e in subset]

    A = X[indices]
    B = Y[indices]
    
    if remove_null_columns:
        non_zero_a = np.any(X != 0, axis=0)
        A = X[:, non_zero_a]

        non_zero_b = np.any(Y != 0, axis=0)
        B = Y[:, non_zero_b]

    return [words[i] for i in indices], A, B


