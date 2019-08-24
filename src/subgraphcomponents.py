import time
import random
import numpy as np
import pandas as pd
import networkx as nx
from diffusiontrees import EulerianDiffuser

class SubGraphComponents:
    """
    Methods separate the original graph and run diffusion on each node in the subgraphs.
    """
    
    def __init__(self, edge_list_path, seeding, vertex_set_cardinality):
        """
        Initializing the object with the main parameters.
        :param edge_list_path: Path to the csv with edges.
        :param seeding: Random seed.
        :param vertex_set_cardinality: Number of unique nodes per tree.
        """
        self.seed = seeding
        self.vertex_set_cardinality = vertex_set_cardinality
        self.read_start_time = time.time()
        self.graph = nx.from_edgelist(pd.read_csv(edge_list_path, index_col = None).values.tolist())
        self.counts = len(self.graph.nodes()) + 1
        self.separate_subcomponents()
        self.single_feature_generation_run()

    def separate_subcomponents(self):
        """
        Finding the connected components.
        """
        self.graph = sorted(nx.connected_component_subgraphs(self.graph), key = len, reverse = True)
        self.read_time = time.time()-self.read_start_time
        
    def single_feature_generation_run(self):
        """
        Running a round of diffusions and measuring the sequence generation performance.
        """
        random.seed(self.seed)
        self.generation_start_time = time.time()
        self.paths = {}

        for sub_graph in self.graph:
            current_cardinality = len(sub_graph.nodes())
            if current_cardinality < self.vertex_set_cardinality:
                self.vertex_set_cardinality = current_cardinality
            diffuser = EulerianDiffuser(sub_graph, self.vertex_set_cardinality)
            self.paths.update(diffuser.diffusions)
        self.paths = [v for k,v in self.paths.items()]
        self.generation_time = time.time() - self.generation_start_time
