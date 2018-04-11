from diffusiontrees import EulerianDiffuser
import pandas as pd
import networkx as nx
import random
import numpy as np
import time

class SubGraphComponents:
    """
    Methods separate the original graph and run diffusion on each node in the subgraphs.
    """
    
    def __init__(self, edge_list_path, seeding, vertex_set_cardinality):
        
        self.seed = seeding
        self.vertex_set_cardinality = vertex_set_cardinality
        self.read_start_time = time.time()
        self.graph = nx.from_edgelist(pd.read_csv(edge_list_path, index_col = None).values.tolist())
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
        Running a round of diffusions.
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
        self.paths = self.paths.values()
        self.generation_time = time.time() - self.generation_start_time
