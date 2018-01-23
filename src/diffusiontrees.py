import random
import networkx as nx
from collections import Counter

class EulerianDiffusionTree:
    
    def __init__(self, node):

        """
        Initializing a diffusion tree.

        :param node: Source of diffusion.
        """
        
        self.start_node = node
        self.infected = [node]
        self.infected_set = set(self.infected)
        self.sub_graph = nx.DiGraph()
        self.sub_graph.add_node(node)
        self.infected_counter = 1
        
    def run_diffusion_process(self, graph, number_of_nodes):

        """
        Creating a diffusion tree from the start node on G with a given vertex set size.
        The tree itself is stored in an NX graph object.

        :param graph: Original graph of interest.
        :param number_of_nodes: Cardinality of vertex set.
        """
        
        while self.infected_counter < number_of_nodes:
            
            end_point = random.sample(self.infected, 1)[0]
            sample = random.sample(graph.neighbors(end_point), 1)[0]

            if sample not in self.infected:
                
                self.infected_counter = self.infected_counter + 1
                self.infected_set = self.infected_set.union([sample])
                self.infected = self.infected + [sample]
                self.sub_graph.add_edges_from([(end_point, sample), (sample, end_point)])
                
                if self.infected_counter == number_of_nodes:
                    break
                
    def create_path_description(self, graph):

        """
        Creating a random Eulerian walk on the diffusion tree.

        :param graph: Original graph of interest.
        """
        
        self.euler = [u for u,v in nx.eulerian_circuit(self.sub_graph, self.start_node)]
        if len(self.euler) == 0 :
            self.euler = [u for u,v in nx.eulerian_circuit(graph, self.start_node)]
        return self.euler

class EndPointDiffusionTree:

    def __init__(self, node):

        """
        Initializing a diffusion tree.

        :param node: Source of diffusion.
        """

        self.needed_nodes = [node]
        self.diffusion_set =  [self.needed_nodes]
        self.start_node = node
        self.infected = set(self.needed_nodes)
        
    def run_diffusion_process(self, graph, number_of_nodes):

        """
        Creating a diffusion tree from the start node on G with a given vertex set size.
        The tree itself is stored in a list of lists.

        :param graph: Original graph of interest.
        :param number_of_nodes: Cardinality of vertex set.
        """

        while len(self.infected) < number_of_nodes:
            diffusion_set_to_be_added = []
            end_point = self.diffusion_set[0][-1]
            sample = random.sample(graph.neighbors(end_point), 1)[0]
            if sample not in self.infected:
                diffusion_set_to_be_added = diffusion_set_to_be_added + [self.diffusion_set[0] + [sample]]
                self.infected = self.infected.union([sample])
                self.needed_nodes = self.needed_nodes + [end_point, sample]
                if len(self.infected) == number_of_nodes:
                    break
                self.diffusion_set = self.diffusion_set + diffusion_set_to_be_added
            random.shuffle(self.diffusion_set)
        
    def transform_infected_nodes(self):

        """
        Creating a set of nodes with degree equal to 1.  
        """

        self.needed_nodes = Counter(self.needed_nodes)
        self.needed_nodes = [needed_node for needed_node in self.needed_nodes if self.needed_nodes[needed_node] == 1]
        self.infected = set(self.needed_nodes)
        
    
    def filter_path_chunks(self):

        """
        Filtering the diffusion paths that end in a vertex with degree equal to 1.
        """

        self.diffusion_set = filter(lambda x: (x[0] == self.start_node) and (x[-1] in self.infected) and len(x) > 1, self.diffusion_set)
    
    def create_path_description(self):

        """
        Creating an endpoint traceback on the diffusion tree.
        """      

        self.transform_infected_nodes()

        if len(self.diffusion_set) > 1:

            self.paths = []
            self.filter_path_chunks()

            for path in self.diffusion_set:

                reverse_path = path[::-1]
                out_path = path[1::] + reverse_path[1:len(path)-1]

                if len(out_path) > 1:

                    self.paths = self.paths + out_path
        else:

            self.paths = self.diffusion_set
    
        self.paths = [self.start_node] + self.paths

        return self.paths
