import numpy as np
import BagLearner as blr
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = np.array([blr.BagLearner(learner=lrl.LinRegLearner, bags=20, verbose=self.verbose) for i in range(20)])
    def author(self):
        return "syusuff3"
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        result = np.empty((points.shape[0], 0))
        for learner in self.learners:
            rez = learner.query(points)
            rez = rez.reshape(-1,1)
            result = np.hstack((result, rez))
        return np.mean(result, axis=1)