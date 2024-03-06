""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		 	   			  		 			     			  	 
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
  		  	   		 	   			  		 			     			  	 
import warnings  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
class DTLearner(object):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		 	   			  		 			     			  	 
    your own correct DTLearner from Project 3.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		 	   			  		 			     			  	 
    :type leaf_size: int  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.treeArray = np.empty(
            (0, 4))  # save the tree in a matrix with 4 cols. [nodeId, FactorIdx, leftIdx, rightIdx]
        self.leaf_size = leaf_size

    def author(self):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
        :rtype: str  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        return "syusuff3"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        # to train data in a DT Learner, we need to build a tree with nodes
        # these nodes contain decisions on certain values
        data_y_reshaped = data_y.reshape(-1, 1)
        data = np.hstack((data_x, data_y_reshaped))
        self.treeArray = self.build_tree(data)

    def get_best_feature_index(self, data):
        data_x, data_y = data[:, 0:-1], data[:, -1]
        # below gives a matrix of correlations of each variable (all x's and y) with one another
        corr_arr = np.corrcoef(data_x, data_y, rowvar=False)
        # we exclude the correlation of y to other variables.
        y_correlations = corr_arr[:-1, -1]
        high_corr_idx = np.argmax(np.abs(y_correlations))

        return high_corr_idx

    def build_tree(self, data):
        # treeArray
        ydata = data[:, -1]
        if ydata.size == 0:
            return np.empty((0, 4))
        if data.shape[0] <= self.leaf_size:
            # single row so we're at the root. return the y val
            meanVal = np.mean(ydata)
            return np.array([['leaf', float(meanVal), np.nan, np.nan]])

        if len(np.unique(data[:, -1])) == 1:
            # all y values are the same so we don't need to explore further
            return np.array([['leaf', float(data[0, -1]), np.nan, np.nan]])
        else:
            # let's determine the best feature to split on
            # we choose a feature that's mostly correlated to y, using the median
            # using axis = 0, we calculate the median for each x COLUMN except the y coln

            # selecting the split index (using correlation). Find the index that's mostly correlated
            splitValIdx = self.get_best_feature_index(data)
            splitVal = np.median(data[:, splitValIdx])
            # grab all rows in data whose values in the splitValIdx column <= splitVal
            left_data = data[data[:, splitValIdx] <= splitVal]
            right_data = data[data[:, splitValIdx] > splitVal]

            if left_data.shape[0] == data.shape[0]:
                # check if the x values at the column are the same and just return mean(y)
                if len(np.unique(data[:, splitValIdx])) == 1:
                    meanVal = np.mean(ydata)
                    return np.array([['leaf', float(meanVal), np.nan, np.nan]])
                else:
                    splitVal = np.min(data[:, splitValIdx])
                    left_data = data[data[:, splitValIdx] <= splitVal]
                    right_data = data[data[:, splitValIdx] > splitVal]

            leftTree = self.build_tree(left_data)
            rightTree = self.build_tree(right_data)
            root = np.array([[float(splitValIdx), float(splitVal), 1, leftTree.shape[0] + 1]])

            # merge each 1D array (4,) into a big 2D array
            result = np.vstack((root, leftTree, rightTree))
            return result

    def predict(self, xvals):
        currIdx = 0
        while True:
            if self.treeArray[currIdx][0] == "leaf":
                return self.treeArray[currIdx][1]
            factorIdx = int(float(self.treeArray[currIdx][0]))
            if xvals[factorIdx] <= float(self.treeArray[currIdx][1]):
                # go to the left child
                currIdx += int(float(self.treeArray[currIdx][2]))
            else:
                # go to the right child
                currIdx += int(float(self.treeArray[currIdx][3]))

    def query(self, points):
        # for each point,
        result = np.empty((points.shape[0],))
        for i in range(points.shape[0]):
            y = self.predict(points[i])
            result[i] = y

        return result


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")  		  	   		 	   			  		 			     			  	 
