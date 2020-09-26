from graph_algos import nd2vec
from n2v_parser import nd2vec_parser
from utilities import read_graph

#main function where all intialization and triggering happens 
def main(args):
    '''
	Pipeline for representational learning for all nodes in a graph.
	'''
    parsed_input_graph  = read_graph(args.input,args.weighted,args.directed)
    graph_nodes = nd2vec(args,parsed_input_graph)
    graph_nodes.prep_trans_prob()
    walks = graph_nodes.simulate_walks(args.num_walks,args.walk_length)
    graph_nodes.generate_nd2vec_embeddings(walks)
    


if __name__ == "__main__":
	args = nd2vec_parser()
	main(args)