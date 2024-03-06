""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 

  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
# this function should return a dataset (X and Y) that will work  		  	   		 	   			  		 			     			  	 
# better for linear regression than decision trees  		  	   		 	   			  		 			     			  	 
def best_4_lin_reg(seed=1489683273):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   			  		 			     			  	 
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	   			  		 			     			  	 
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param seed: The random seed for your data generation.  		  	   		 	   			  		 			     			  	 
    :type seed: int  		  	   		 	   			  		 			     			  	 
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   			  		 			     			  	 
    :rtype: numpy.ndarray  		  	   		 	   			  		 			     			  	 
    """
    # 2 to 10 X columns
    # 10 to 1000 rows
    np.random.seed(seed)
    ROWS = 800
    COLS = 6

    # Linear regression is best suited for datasets with a linear relationship so we will generate such data here
    # An example equation to be used will look like this:
    # Y = c1 * x1 + c2 * x2 + c3 * x3 + ... + cn * xn
    # and we might add a few other terms to create relationships between features
    # eg: + (c1+c3) * x1x3 + c5* x5x6

    X = np.random.random(size=(ROWS, COLS))
    Y = np.zeros((ROWS,))
    coefficients = np.random.random(COLS)
    # for each row, multiply the column val by the coefficient and sum them
    for i in range(ROWS):
        Y[i] = np.sum(X[i, :] * coefficients)

    return X, Y

def best_4_dt(seed=1489683273):
    """
    Returns data that performs significantly better with DTLearner than LinRegLearner.
    The data set should include from 2 to 10 columns in X, and one column in Y.
    The data should contain from 10 (minimum) to 1000 (maximum) rows.

    :param seed: The random seed for your data generation.
    :type seed: int
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.
    :rtype: numpy.ndarray
    """
    np.random.seed(seed)
    # for DT, we create data with non linear relationships
    # we will use different relationships like quadratic, cosine, logarithm, and exponential on features

    # for this exercise, we'll be raising each column to a power that corresponds to the column index.
        # to is added to ignore lower powers. Adding 1 will work as well.
    ROWS, COLS = 600, 7
    X = np.random.uniform(1, 10, size=(ROWS, COLS))
    Y = np.zeros((ROWS,))

    for row in range(ROWS):
        yval = 0
        for col in range(COLS):
            colVal = X[row][col]
            yval += colVal ** (col+2)

        Y[row] = yval/(10**COLS)    # normalize data by dividing by a large number. This, if removed will still work but with very high X values and RMSE.
    return X, Y

  		  	   		 	   			  		 			     			  	 
def author():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
    :rtype: str  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    return "syusuff3"
  		  	   		 	   			  		 			     			  	 

if __name__ == "__main__":
    # Whenever the seed is the same you must return exactly the same data set. Different seeds must result in different data sets.
    """
    Each dataset must include no fewer than 2 and no more than 10 features (or “X”) columns.
    The dataset must contain 1 target (or “Y”) column. The Y column must contain real numbers.
    Y values may not be hard-coded and must be generated by the X value.
    Each dataset must contain no fewer than 10 and no more than 1000 examples (i.e., rows).
    While you are free to determine these sizes, they may not vary between generated datasets.
    """
    pass
