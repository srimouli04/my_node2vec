from node2vec_parser import parm_parser
from graph_algos import node2vec

#main function where all intialization and triggering happens 
def main(args):
    '''
	Pipeline for representational learning for all nodes in a graph.
	'''
    graph_nodes = node2vec(args)
    graph_nodes.read_graph()
    graph_nodes.prep_trans_prob()
    walks = graph_nodes.simulate_walks(args.num_walks,args.walk_length)
    graph_nodes.generate_node2vec_embeddings(walks)
    


if __name__ == "__main__":
	args = parm_parser()
	main(args)