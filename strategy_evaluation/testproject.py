import ManualStrategy as ms
import experiment1 as exp1
# to initialize and run all necessary files for the report

def run_experiment1():
    # just call experiment1.py file
    in1, out1 = exp1.run_experiment()
    #in2, out2 = exp1.run_experiment()

    #isequal1 = in1.values == in2.values
    #isequal2 = out1.values == out2.values
    return

if __name__ == "__main__":
    # run manual strategy and generate charts for insample and outsample
    manual_strategy = ms.ManualStrategy(impact=0.005, commission=9.95)
    manual_strategy.generate_charts("JPM")
    run_experiment1()