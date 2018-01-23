from diffusiontrees import EulerianDiffusionTree, EndPointDiffusionTree
import pandas as pd
import networkx as nx
import time
import random
import numpy as np

class SubGraphComponents:
    
    def __init__(self, edge_list_path, seeding):
        
        self.seed = seeding
        self.start_time = time.time()
        self.graph = nx.from_edgelist(pd.read_csv(edge_list_path, index_col = None).values.tolist())
 
    def separate_subcomponents(self):
        
        self.graph = sorted(nx.connected_component_subgraphs(self.graph), key = len, reverse = True)
        
    def print_graph_generation_statistics(self):       
        print("The graph generation at run " + str(self.seed) + " took: " + str(round(time.time() - self.start_time, 3)) + " seconds.\n") 
        
    def single_feature_generation_run(self, vertex_set_cardinality, traceback_type):
        
        random.seed(self.seed)
        
        self.start_time = time.time()
        
        self.paths = {}

        for sub_graph in self.graph:
 
            nodes = sub_graph.nodes()
            random.shuffle(nodes)
            
            current_cardinality = len(nodes)
            
            if current_cardinality < vertex_set_cardinality:
                vertex_set_cardinality = current_cardinality
            for node in nodes:
                tree = EulerianDiffusionTree(node)
                tree.run_diffusion_process(sub_graph, vertex_set_cardinality)
                path_description = tree.create_path_description(sub_graph)
                self.paths[node] = map(lambda x: str(x), path_description)
                
        self.paths = self.paths.values()
                
    def print_path_generation_statistics(self):
        print("The sequence generation took: " + str(time.time() - self.start_time))
        print("Average sequence length is: " + str(np.mean(map(lambda x: len(x), self.paths))))
        
    def get_path_descriptions(self):
        return self.paths
