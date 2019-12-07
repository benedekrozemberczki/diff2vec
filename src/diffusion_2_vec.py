"""Diff2Vec model."""

import time
import logging
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.models import Word2Vec, Doc2Vec
import numpy.distutils.system_info as sysinfo
from subgraphcomponents import SubGraphComponents
from gensim.models.word2vec import logger, FAST_VERSION
from helper import parameter_parser, result_processing
from helper import process_non_pooled_model_data, argument_printer

sysinfo.get_info("atlas")
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def create_features(seeding, edge_list_path, vertex_set_cardinality):
    """
    Creating a single feature for every node.
    :param seeding: Random seed.
    :param edge_list_path:  Path to edge list csv.
    :param vertex_set_cardinality: Number of diffusions per node.
    :return: Sequences and measurements.
    """
    sub_graphs = SubGraphComponents(edge_list_path, seeding, vertex_set_cardinality)
    return sub_graphs.paths, sub_graphs.read_time, sub_graphs.generation_time, sub_graphs.counts

def run_parallel_feature_creation(edge_list_path,
                                  vertex_set_card,
                                  replicates,
                                  workers):
    """
    Creating linear node sequences for every node multiple times in a parallel fashion
    :param edge_list_path: Path to edge list csv.
    :param vertex_set_card: Number of diffusions per node.
    :param replicates: Number of unique nodes per diffusion.
    :param workers: Number of cores used.
    :return walk_results: List of 3-length tuples with sequences and performance measurements.
    :return counts: Number of nodes.
    """
    results = Parallel(n_jobs=workers)(delayed(create_features)(i, edge_list_path, vertex_set_card) for i in tqdm(range(replicates)))
    walk_results, counts = result_processing(results)
    return walk_results, counts

def learn_pooled_embeddings(walks, counts, args):
    """
    Method to learn an embedding given the sequences and arguments.
    :param walks: Linear vertex sequences.
    :param counts: Number of nodes.
    :param args: Arguments.
    """
    model = Word2Vec(walks,
                     size=args.dimensions,
                     window=args.window_size,
                     min_count=1,
                     sg=1,
                     workers=args.workers,
                     iter=args.iter,
                     alpha=args.alpha)

    save_embedding(args, model, counts)

def learn_non_pooled_embeddings(walks, counts, args):
    """
    Method to learn an embedding given the sequences and arguments.
    :param walks: Linear vertex sequences.
    :param counts: Number of nodes.
    :param args: Arguments.
    """
    walks = process_non_pooled_model_data(walks, counts, args)
    model = Doc2Vec(walks,
                    size=args.dimensions,
                    window=0,
                    dm=0,
                    alpha=args.alpha,
                    iter=args.iter,
                    workers=args.workers)

    save_embedding(args, model, counts)

def save_embedding(args, model, counts):
    """
    Function to save the embedding.
    :param args: Arguments object.
    :param model: The embedding model object.
    :param counts: Number of nodes.
    """
    out = []
    for node in range(1, counts[0]):
        if args.model == "non-pooled":
            out.append([int(node)-1] + list(model.docvecs[node]))
        else:
            out.append([int(node)-1] + list(model.wv[str(node-1)]))
    columns = ["node"] + ["x_"+str(x) for x in range(args.dimensions)]
    out = pd.DataFrame(out, columns=columns)
    out = out.sort_values(["node"])
    out.to_csv(args.output, index=None)

def main(args):
    """
    Main method for creating sequences and learning the embedding.
    :param args: Arguments object.
    """
    argument_printer(args)
    print("\n----------\nFeature extraction starts.\n----------\n\n")
    walks, counts = run_parallel_feature_creation(args.input,
                                                  args.vertex_set_cardinality,
                                                  args.num_diffusions,
                                                  args.workers)
    print("\n----------\nLearning starts.\n----------\n")

    if args.model == "non-pooled":
        learn_non_pooled_embeddings(walks, counts, args)
    else:
        learn_pooled_embeddings(walks, counts, args)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
