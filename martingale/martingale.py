""""""  		  	   		 	   			  		 			     			  	 
"""Assess a betting strategy.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
  		  	   		 	   			  		 			     			  	 
Student Name: Sodiq Yusuff 		  	   		 	   			  		 			     			  	 
GT User ID: syusuff3		  	   		 	   			  		 			     			  	 
GT ID: 903953477 		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def author():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
    :rtype: str  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    return "syusuff3"
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def gtid():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT ID of the student  		  	   		 	   			  		 			     			  	 
    :rtype: int  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    return 903953477  # replace with your GT ID number
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		 	   			  		 			     			  	 
    :type win_prob: float  		  	   		 	   			  		 			     			  	 
    :return: The result of the spin.  		  	   		 	   			  		 			     			  	 
    :rtype: bool  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    result = False
    rdn = np.random.random()
    if rdn <= win_prob:
        result = True  		  	   		 	   			  		 			     			  	 
    return result
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 

def run_single_realistic_episode(win_prob, bankroll):
    """
    User cannot spend than the bankroll
    """
    episode_length = 1000
    max_win_allowed = 80
    episode_winnings = 0
    episode_count = 0
    winnings = np.array([0])
    last_bet_amt = 0
    has_sufficient_balance = True   # track the bankroll
    while episode_winnings < max_win_allowed and episode_count < episode_length and episode_winnings > -bankroll:
        won = False
        bet_amount = 1
        while not won and episode_count < episode_length:
            won = get_spin_result(win_prob)
            last_bet_amt = bet_amount
            if won == True:
                # there's a win
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount = bet_amount * 2


            winnings = np.append(winnings, episode_winnings)
            episode_count += 1
            # before moving to the next iteration, let's check if there's a sufficient balance to bet with
            amount_at_hand = bankroll + episode_winnings
            if amount_at_hand < bet_amount:
                # print(f"insufficient balance! balance: {episode_winnings}, betAmt: {bet_amount}, atHand= {amount_at_hand}")
                break

    # if winnings[-1] <0:
    #     print(f"lost at {winnings[-1]}. Last bet amt={last_bet_amt}")
    # else:
    #     print(f"win at {winnings[-1]}. Last bet amt={last_bet_amt}")
    # fill winning list with 80s to make 1000 points per instruction.
    winnings = np.append(winnings, [winnings[-1]]*(episode_length+1 - len(winnings)))
    return winnings

def run_single_episode(win_prob):
    """
    An episode consists of 1000 successive bets. In this experiment, we assume that the user has
    an unlimited budget to bet with
    """
    episode_length = 1000
    max_win_allowed = 80        # stop once you win $80
    episode_winnings = 0
    episode_count = 0
    winnings = np.array([0])    # adding a zero for convenience. Just to ensure all data starts from zero
    while episode_winnings < max_win_allowed and episode_count < episode_length:
        won = False
        bet_amount = 1
        while not won and episode_count < episode_length:
            won = get_spin_result(win_prob)
            if won == True:
                # there's a win
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount = bet_amount * 2     # double the bet amount anytime you lose

            winnings = np.append(winnings, episode_winnings)
            episode_count += 1

    # fill winning list with 80s to make 1000 points per instruction.
    # basically if the win = $80 at the 300th spin, fill forward $80 for the remaining 700 spins
    winnings = np.append(winnings, [max_win_allowed]*(episode_length+1 - len(winnings)))
    return winnings

def experiment2(win_prob):
    # run 1000 times
    bankroll = 256  # $256 -> max allowed to spend
    episode_winnings = np.array([])
    lastValuePerEpisode = np.array([])
    for i in range(1000):
        episode_array = run_single_realistic_episode(win_prob, bankroll)
        lastValuePerEpisode = np.append(lastValuePerEpisode, episode_array[-1])
        if i == 0:
            episode_winnings = episode_array
        else:
            episode_winnings = np.vstack((episode_winnings, episode_array))

    #print(f"shape of array: {episode_winnings.shape}")
    mean_values = episode_winnings.mean(axis=0)  # compute column-wise mean values
    standard_deviations = episode_winnings.std(axis=0)
    line_above_std = mean_values + standard_deviations
    line_below_std = mean_values - standard_deviations

    print_to_file(f"""
        For the {lastValuePerEpisode.size} spins in Experiment2,
        \n $80 occurred {np.count_nonzero(lastValuePerEpisode == 80)} times and $-256 occurred {np.count_nonzero(lastValuePerEpisode == -256)} times
        \n Other values are: {lastValuePerEpisode[(lastValuePerEpisode != 80) & (lastValuePerEpisode != -256)]}
    """)
    plot_data([(mean_values, "Mean"), (line_above_std, "Mean + Std Deviation"), (line_below_std, "Mean - Std Deviation")],
              "Figure 4 - Experiment2 Mean and Std Deviations", "Figure4", "Spin round", "Winning($)")

    # check the max and min values and the count of the last value
    print_to_file(f"""
    Experiment 2: The upper and lower standard deviation from the mean: 
            \nThe upper std deviation reached a maximum at {line_above_std.max()}
            \nThe upper std deviation line was stable at {line_above_std[-1]} value for the last {np.count_nonzero(line_above_std == line_above_std[-1])} spins

            \nThe lower std deviation reached a minimum at {line_below_std.min()}
            \nThe lower std deviation line was stable at {line_below_std[-1]} value for the last {np.count_nonzero(line_below_std == line_below_std[-1])} spins
        """)

    # get the medians for FIGURE 5
    median_values = np.median(episode_winnings, axis=0)
    median_plus_std = median_values + standard_deviations
    median_minus_std = median_values - standard_deviations
    plot_data([(median_values, "Median"), (median_plus_std, "Median + Std Deviation"), (median_minus_std, "Median - Std Deviation")],
              "Figure 5 - Experiment2 Median and Std Deviations","Figure5", "Spin round", "Winning($)")

def print_to_file(content):
    f = open("p1_results.txt", 'a')
    f.write(content)
    f.close()
def plot_figure1(win_prob):
    # FIGURE 1 for 10 EPISODES
    ten_episode_winnings = np.array([])
    lastEpisodeVals = []
    minimumWin = 80;
    # running 10 episodes
    for i in range(10):
        episode_array = run_single_episode(win_prob)
        lastEpisodeVals.append(episode_array[-1])
        minimumWin = min(minimumWin, min(episode_array))
        if i == 0:      # for the first round
            ten_episode_winnings = (episode_array, f"Episode {i+1}")
        else:
            ten_episode_winnings = np.vstack((ten_episode_winnings, (episode_array, f"Episode {i+1}")))

    print_to_file(f"Experiment 1 - Amount won for 10 episodes \n{lastEpisodeVals} \nMaximum loss was ${minimumWin}")
    plot_data(ten_episode_winnings, "Figure 1 - Running 10 episodes", "Figure1", "Spin round", "Winning($)")


def plot_figure2And3(win_prob):
    # run for 1000 rounds
    # calculate the mean of each spin round
    # FIGURE 2 for 1000 EPISODES
    episode_winnings = np.array([])
    minimumWin = 80
    # running 1000 episodes
    for i in range(1000):
        episode_array = run_single_episode(win_prob)
        minimumWin = min(minimumWin, min(episode_array))
        if episode_winnings.size == 0:
            episode_winnings = episode_array
        else:
            episode_winnings = np.vstack((episode_winnings, episode_array))

    mean_values = episode_winnings.mean(axis=0)     # compute column-wise mean values
    standard_deviations = episode_winnings.std(axis=0)
    line_above_std = mean_values + standard_deviations
    line_below_std = mean_values - standard_deviations

    print_to_file(f"\nExperiment 1 - Maximum loss in 1000 episodes ${minimumWin}")

    # check the max and min values and the count of the last value
    print_to_file(f"""
        The upper and lower standard deviation from the mean: 
        \nThe upper std deviation reached a maximum at {line_above_std.max()}
        \nThe upper std deviation line was stable at {line_above_std[-1]} value for the last {np.count_nonzero(line_above_std == line_above_std[-1])} spins
        
        \nThe lower std deviation reached a minimum at {line_below_std.min()}
        \nThe lower std deviation line was stable at {line_below_std[-1]} value for the last {np.count_nonzero(line_below_std == line_below_std[-1])} spins
    """)


    plot_data([(mean_values, "Mean"), (line_above_std, "Mean + Std Deviation"), (line_below_std, "Mean - Std Deviation")],
              "Figure 2 - Mean and Std Deviations over 1000 Episodes", "Figure2", "Spin round", "Winning($)")

    # USING THE SAME DATA, we plot the median
    plot_figure3(episode_winnings, standard_deviations)

def plot_figure3(np_array, standard_deviations):
    median_values = np.median(np_array, axis=0)     # compute column-wise median values
    line_above_std = median_values + standard_deviations
    line_below_std = median_values - standard_deviations

    plot_data([(median_values, "Median"), (line_above_std, "Median + Std Deviation"), (line_below_std, "Median - Std Deviation")],
              "Figure 3 - Median and Std Deviations over 1000 Episodes", "Figure3", "Spin round", "Winning($)")


def plot_data(data_array, title, filename, xlabel, ylabel, xmin = 0, xmax=300, ymin=-256, ymax=100):
    # begin a new figure to avoid overlapping charts.
    plt.figure()

    for data in data_array:
        plt.plot(data[0], label=data[1])

    plt.title(title)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig("images/{}.png".format(str(filename)))


def personal_test():
    r1 = run_single_episode(0.474)
    r2 = run_single_episode(0.474)
    r3 = run_single_episode(0.474)
    rss = np.array([r1,r2,r3])
    mean1 = np.mean(rss, axis=0)
    std = np.std(rss, axis=0)
    up = mean1 + std
    lw = mean1 - std
    #plt.plot(r1)
    #plt.plot(mean1)
    plt.plot(up)
    plt.plot(lw)
    plt.show()
    return
    # create a sample 3x10 array. 3 rows, 10 cols
    arr1 = np.random.random((1, 5))
    arr2 = np.random.random((3, 5))
    arr3 = np.array([1,2,3,4])
    arr4 = np.array([(1,2,3,4,5), (6,7,8,9,10), (11,12,13,14,15)])
    mean = np.mean(arr4, axis=0)
    print(mean)
    plt.plot(mean)
    plt.show()
    # print(arr1)
    # print("array 2 below")
    # print(arr2)
    # print(f"array sizes {arr1.shape} and {arr2.shape} and {arr3.shape}")
    # print(arr3.shape[0])
    # ndarr = np.array([0])
    # ndarr = np.append(ndarr, 2)
    # ndarr = np.append(ndarr, 3)
    # ndarr = np.append(ndarr, 4)
    # ndarr = np.append(ndarr, 5)
    # print(f"array is {ndarr} and shape is {ndarr.shape}")
    # ndarr = np.append(ndarr, [7,8,9,10])
    # print(f"array is {ndarr} and shape is {ndarr.shape} len = {len(ndarr)}")
    # ndarr = np.append(ndarr, [80]*5)
    # print(f"array is {ndarr} and shape is {ndarr.shape} len = {len(ndarr)} and shape = {len(ndarr)}")

# test total amount of loss possible in 1000 spins
def test_code():
    """  		  	   		 	   			  		 			     			  	 
    Method to test your code  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    win_prob = 0.474 # 18/38 the probability of a win, since there are 18 blacks
    np.random.seed(gtid())  # do this only once
    # add your code here to implement the experiments

    plot_figure1(win_prob)
    plot_figure2And3(win_prob)
    experiment2(win_prob)


  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    test_code()  		  	   		 	   			  		 			     			  	 
