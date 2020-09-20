import networkx as nx
from tqdm import tqdm
from gensim.models import word2vec 

def read_graph(self,args):
    
    '''Reads input by default its fifa '''
    
    if args.weighted:
        Graph_obj = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),),create_using=nx.DiGraph())
    else:
        Graph_obj = nx.read_edgelist(args.input,nodetype=int,create_using=nx.DiGraph())
        print('Edge weighting \n')
        for edge in tqdm(Graph_obj.edges()):
            Graph_obj[edge[0]][edge[1]]['weight'] = 1.0
    
    if not args.directed:
        Graph_obj = Graph_obj.to_undirected()
    
    return Graph_obj
    
         