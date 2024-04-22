import numpy as np


def author():
    return 'syusuff3'
def get_mode(arr):
    unique, counts = np.unique(arr, return_counts=True)
    idx = np.argmax(counts)
    return unique[idx]


# this learner is used for classification data so we use mode instead of mean or median
class RTLearner(object):
    """
    Random Tree Learner
    """
    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.treeArray = np.empty((0,4))     # save the tree in a matrix with 4 cols. [nodeId, FactorIdx, leftIdx, rightIdx]
        self.leaf_size = leaf_size

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
        data_y_reshaped = data_y.reshape(-1,1)
        data = np.hstack((data_x, data_y_reshaped))
        self.treeArray = self.build_tree(data)

    def build_tree(self, data):
        # treeArray
        ydata = data[:, -1]
        if ydata.size == 0:
            return np.empty((0, 4))
        if data.shape[0] <= self.leaf_size:
            # single row so we're at the root. return the y val
            mode = get_mode(ydata)
            # single row so we're at the root. return the y val
            return np.array([['leaf', float(mode), np.nan, np.nan]])

        if len(np.unique(data[:, -1])) == 1:
            # all y values are the same so we don't need to explore further
            return np.array([['leaf', float(data[0, -1]), np.nan, np.nan]])
        else:
            # let's determine the best feature to split on
            # we choose a random feature from the list of features
            # splitVal will be the median of that feature

            # generate a random colum from 0 to columCount-1 (excluding the y column)
            randColIdx = np.random.randint(0, data.shape[1]-1)
            splitVal = np.median(data[:, randColIdx])

            # grab all rows in data whose values in the splitValIdx column <= splitVal
            left_data = data[data[:, randColIdx] <= splitVal]
            right_data = data[data[:, randColIdx] > splitVal]

            if left_data.shape[0] == data.shape[0]:
                # check if the x values at the column are the same and just return mean(y)
                if len(np.unique(data[:, randColIdx])) == 1:
                    mode = get_mode(ydata)
                    return np.array([['leaf', float(mode), np.nan, np.nan]])
                else:
                    splitVal = np.min(data[:, randColIdx])
                    left_data = data[data[:, randColIdx] <= splitVal]
                    right_data = data[data[:, randColIdx] > splitVal]

            leftTree = self.build_tree(left_data)
            rightTree = self.build_tree(right_data)

            root = np.array([[float(randColIdx), float(splitVal), 1, leftTree.shape[0]+1]])
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
