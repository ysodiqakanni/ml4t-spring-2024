""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import math  		  	   		 	   			  		 			     			  	 
import sys  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt
import datetime
  		  	   		 	   			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bgl
import InsaneLearner as ilr


def get_istanbuldata(file_name):
    inf = open(file_name)
    data = np.array(
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
    )
    return data

def get_wine_data():
    file_name = 'Data/winequality-red.csv' #
    inf = open(file_name)
    data = np.array(
        [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    )
    return data


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 903953477

def run_experiment_1():
    # run several DTLearner experiments with varying leaf sizes
    leaf_sizes = np.array(list(range(0,70, 2)))
    rmse_insample = np.array([])
    rmse_outsample = np.array([])

    for leaf_size in leaf_sizes:
        learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse_ismpl = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmse_insample = np.append(rmse_insample, rmse_ismpl)

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse_osmpl = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        rmse_outsample = np.append(rmse_outsample, rmse_osmpl)

    plt.figure()
    plt.plot(leaf_sizes, rmse_insample, label='In sample')
    plt.plot(leaf_sizes, rmse_outsample, label='Out sample')
    label_and_title_plot(plt, "Decision Tree regression with varying leaf sizes", "Leaf size", "RMSE", "Exp1DtRMSE")

def run_experiment_2():
    # choose a fixed number of bags, run a bagLearner with DTLearner and vary the leaf size
    leaf_sizes = np.array(list(range(0,70, 2)))
    rmse_insample = np.array([])
    rmse_outsample = np.array([])

    for leaf_size in leaf_sizes:
        learner = bgl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20)
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse_ismpl = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmse_insample = np.append(rmse_insample, rmse_ismpl)

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse_osmpl = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        rmse_outsample = np.append(rmse_outsample, rmse_osmpl)

    plt.figure()
    plt.plot(leaf_sizes, rmse_insample, label='In sample')
    plt.plot(leaf_sizes, rmse_outsample, label='Out sample')
    label_and_title_plot(plt, "20 Bags of Decision Tree learners", "Leaf size", "RMSE", "Exp2BaggingDt")

def run_experiment_3():
    leaf_sizes = np.array(list(range(0,50)))
    dt_time_to_train = exp3_dt_learner(leaf_sizes, "DT")
    rt_time_to_train = exp3_dt_learner(leaf_sizes, "RT")
    plt.figure()
    plt.plot(leaf_sizes, dt_time_to_train, label='DT learner')
    plt.plot(leaf_sizes, rt_time_to_train, label='RT learner')
    label_and_title_plot(plt, "DT vs RT Leaner Time to train", "Leaf size", "Time(milliseconds)", "Exp3DtRtTrainTime")
def exp3_dt_learner(leaf_sizes, type="DT"):
    # compare DT and RT learners using two metrics:
    # record performance (time to train and MAE)
    dt_time_to_train = np.array([])
    mean_absolute_error_insample = np.array([])
    mean_absolute_error_outsample = np.array([])
    # create a temporary default learner
    learner = dtl.DTLearner()
    for leaf_size in leaf_sizes:
        if type == "DT":
            # dt learner
            learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
        else:
            learner = rtl.RTLearner(leaf_size=leaf_size, verbose=False)
        time_start = datetime.datetime.now()
        learner.add_evidence(train_x, train_y)  # train it
        time_end = datetime.datetime.now()
        time_elapsed = (time_end-time_start).microseconds/1000
        dt_time_to_train = np.append(dt_time_to_train, time_elapsed)
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        mae = (np.absolute(train_y - pred_y)).sum() / train_y.shape[0]
        mean_absolute_error_insample = np.append(mean_absolute_error_insample, mae)

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        mae = (np.absolute(test_y - pred_y)).sum() / test_y.shape[0]
        mean_absolute_error_outsample = np.append(mean_absolute_error_outsample, mae)

    plt.figure()
    plt.plot(leaf_sizes, mean_absolute_error_insample, label='In sample MAE')
    plt.plot(leaf_sizes, mean_absolute_error_outsample, label='Out of sample MAE')
    plot_title = "Decision tree regression MAE" if type == "DT" else "Random tree regression MAE"
    file_name = "Exp3DtMae" if type == "DT" else "Exp3RtMae"
    label_and_title_plot(plt, plot_title, "Leaf size", "MAE", file_name)

    return dt_time_to_train

def label_and_title_plot(plt, title, xlabel, ylabel, fileName):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig("{}.png".format(str(fileName)))

if __name__ == "__main__":
    if len(sys.argv) != 2:  		  	   		 	   			  		 			     			  	 
        #print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    test_data_file = sys.argv[1]
    data = get_istanbuldata(test_data_file)
    np.random.seed(gtid())

    # compute how much of the data is training and testing  		  	   		 	   			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		 	   			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		 	   			  		 			     			  	 

    # get the row indices
    row_indices = np.arange(data.shape[0])
    np.random.shuffle(row_indices)   # to shuffle the indices

    train_x = data[row_indices[:train_rows], 0:-1]
    train_y = data[row_indices[:train_rows], -1]
    test_x = data[row_indices[train_rows:], 0:-1]
    test_y = data[row_indices[train_rows:], -1]

    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
