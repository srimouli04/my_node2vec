from Src.graph_algos import nd2vec
from Src.n2v_parser import nd2vec_parser
from Src.utilities import read_graph,tab_printer

#main function where all intialization and triggering happens 
def nd2vec_main(args):
    '''
	Pipeline for representational learning for all nodes in a graph.
	'''
    tab_printer(args)
    parsed_input_graph  = read_graph(args.input,args.weighted,args.directed)
    graph_nodes = nd2vec(args,parsed_input_graph)
    graph_nodes.prep_trans_prob()
    walks = graph_nodes.simulate_walks(args.num_walks,args.walk_length)
    graph_nodes.generate_nd2vec_embeddings(walks)
    


if __name__ == "__main__":
	args = nd2vec_parser()
	nd2vec_main(args)
