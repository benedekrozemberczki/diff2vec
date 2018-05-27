import random
import networkx as nx

class EulerianDiffuser:
    """
    Class to make diffusions for a given graph.
    """
    def __init__(self, graph, number_of_nodes):
        """
        Initializing a diffusion object.
        :param graph: Graph of interest.
        :param number_of_nodes: Cardinality of vertex set.
        """
        self.graph = graph
        self.number_of_nodes = number_of_nodes
        self.nodes = graph.nodes()
        self.run_diffusions()

    def run_diffusion_process(self, node):
        """
        Generating a diffusion tree from a given source node and linearizing it with an Eulerian tour.
        :param node: Source of diffusion.
        :return euler: Eulerian linear node sequence. 
        """
        infected = [node]
        sub_graph = nx.DiGraph()
        sub_graph.add_node(node)
        infected_counter = 1
        
        while infected_counter < self.number_of_nodes:
            end_point = random.sample(infected, 1)[0]
            sample = random.sample(self.graph.neighbors(end_point), 1)[0]
            if sample not in infected:   
                infected_counter = infected_counter + 1
                infected = infected + [sample]
                sub_graph.add_edges_from([(end_point, sample), (sample, end_point)])
                if infected_counter == self.number_of_nodes:
                    break
        euler = [str(u) for u,v in nx.eulerian_circuit(sub_graph, start_node)]
        if len(euler) == 0:
            euler = [str(u) for u,v in nx.eulerian_circuit(graph, start_node)]
        return euler

    def run_diffusions(self):
        """
        Running diffusions from every node.
        """
        self.diffusions = {node: self.run_diffusion_process(node) for node in self.nodes}
