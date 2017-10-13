from subgraphcomponents import SubGraphComponents
from helper import parameter_parser
from joblib import Parallel, delayed
from gensim.models import Word2Vec
from gensim.models.word2vec import logger, FAST_VERSION
import logging
import numpy.distutils.system_info as sysinfo
import scipy; scipy.show_config()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sysinfo.get_info('atlas')
     
def create_features(seeding, edge_list_path, vertex_set_cardinality, traceback_type):
    
    sub_graphs = SubGraphComponents(edge_list_path, seeding)
    sub_graphs.separate_subcomponents()
    sub_graphs.print_graph_generation_statistics()
    sub_graphs.single_feature_generation_run(vertex_set_cardinality, traceback_type)
    sub_graphs.print_path_generation_statistics()
    paths = sub_graphs.get_path_descriptions()
    
    return paths

def run_parallel_feature_creation(edge_list_path,  vertex_set_cardinality, number_of_replicates, num_of_workers, traceback_type):
    number_of_cores = num_of_workers
    results = Parallel(n_jobs = number_of_cores)(delayed(create_features)(i, edge_list_path, vertex_set_cardinality, traceback_type) for i in range(0,number_of_replicates))
    return results
    
def learn_embeddings(walks, args):
    model = Word2Vec(walks, size = args.dimensions, window = args.window_size, min_count = 1, sg = 1, workers = args.workers, iter = args.iter, alpha = args.alpha)
    model.wv.save_word2vec_format(args.output)


def main(args):
    walks = run_parallel_feature_creation(args.input,  args.vertex_set_cardinality, args.num_diffusions, args.workers, args.type)
    walks = [w for walk in walks for w in walk]
    learn_embeddings(walks, args)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
