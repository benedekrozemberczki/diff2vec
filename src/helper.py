import argparse

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
                        default = 80,
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
    
    parser.add_argument('--type',
                        type = str,
                        default = "eulerian",
                        help = 'Traceback type. Default is Eulerian.')
    
    return parser.parse_args()
