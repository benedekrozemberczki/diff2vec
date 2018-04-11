import argparse
from texttable import Texttable
import numpy as np

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Facebook Restaurants network.
    The default hyperparameters give a high quality representation already without grid search.
    """

    parser = argparse.ArgumentParser(description = "Run diffusion2vec.")

    parser.add_argument('--input',
                        nargs = '?',
                        default = './data/restaurant_edges.csv',
	                help = 'Input graph path')

    parser.add_argument('--output',
                        nargs = '?',
                        default = './output/restaurant.out',
	                help = 'Embeddings path')

    parser.add_argument('--dimensions',
                        type = int,
                        default = 128,
	                help = 'Number of dimensions. Default is 128.')

    parser.add_argument('--vertex-set-cardinality',
                        type = int,
                        default = 40,
	                help = 'Length of diffusion per source is 2*cardianlity-1. Default is 40.')

    parser.add_argument('--num-diffusions',
                        type = int,
                        default = 10,
	                help = 'Number of diffusions per source. Default is 10.')

    parser.add_argument('--window-size',
                        type = int,
                        default = 10,
                    	help = 'Context size for optimization. Default is 10.')

    parser.add_argument('--iter',
                        default = 1,
                        type = int,
                        help = 'Number of epochs in ASGD. Default is 1.')

    parser.add_argument('--workers',
                        type = int,
                        default = 4,
	                help = 'Number of cores. Default is 4.')

    parser.add_argument('--alpha',
                        type = float,
                        default = 0.025,
	                help = 'Initial learning rate. Default is 0.025.')
    
    return parser.parse_args()

def generation_tab_printer(read_times, generation_times):
    """
    Function to print the time logs in a nice tabular format.
    """
    t = Texttable() 
    t.add_rows([["Metric","Value"],
                ["Mean graph read time:", np.mean(read_times)],
                ["Standard deviation of read time.",np.std(read_times)]]) 
    print t.draw() 
    t = Texttable()
    t.add_rows([["Metric","Value"],
                ["Mean sequence generation time:", np.mean(generation_times)],
                ["Standard deviation of generation time.",np.std(generation_times)]])
    print t.draw() 

def result_processing(results):
    walk_results = map(lambda x: x[0],results)
    read_time_results = map(lambda x: x[1],results)
    generation_time_results = map(lambda x: x[2],results)
    generation_tab_printer(read_time_results, generation_time_results)
    walk_results = [walk for walks in walk_results for walk in walks]
    return walk_results
