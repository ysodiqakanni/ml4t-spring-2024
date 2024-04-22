""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Bag Learner - An ensemble of learners		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def author():
    return 'syusuff3'

def get_mode(arr):
    unique, counts = np.unique(arr, return_counts=True)
    idx = np.argmax(counts)
    return unique[idx]

class BagLearner(object):
    """  		  	   		 	   			  		 			     			  	 
    This is a Bootstrap Aggregation Learner.		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 

        # create n instances or bags of the learner. eg 10 DT instances
        # grab random data n times to train each learner instance
        # on predict, use each learner instance to run the prediction and take the mean of all
        self.learners = []
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))
  		  	   		 	   			  		 			     			  	 
    def add_evidence(self, data_x, data_y):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Add training data to learner  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner  		  	   		 	   			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		 	   			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	   			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		 	   			  		 			     			  	 
        """
        for i in range(len(self.learners)):
            # select random features with replacement
            randomIndices = np.random.randint(0, data_x.shape[0], len(data_y))
            randomX = data_x[randomIndices]
            randomY = data_y[randomIndices]
            self.learners[i].add_evidence(randomX, randomY)
  		  	   		 	   			  		 			     			  	 
    def query(self, points):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	   			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		 	   			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		 	   			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 

        # querying an array of points gives an array of y vals
        # for each bag (learner instance), we run the query and add the results to a bigger array
        # then we take the mean
        result = np.empty((points.shape[0],0))
        for i in range(len(self.learners)):
            qrez = self.learners[i].query(points)
            qrez = qrez.reshape(-1,1)  # change qrez from 1D to 2D array
            result = np.hstack((result, qrez))
        # let's get the mean row by row
        #meanResult = np.mean(result, axis=1)
        modeResult = np.apply_along_axis(get_mode, axis=1, arr=result)
        return modeResult

  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		 	   			  		 			     			  	 
