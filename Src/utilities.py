import networkx as nx
import numpy as np
from texttable import Texttable
from tqdm import tqdm



def read_graph(input_file,is_weighted,is_directed):
    '''It reads the graph and prepares it for further processing. By default karate club graph is read'''
    
    if is_weighted:
        Graph_obj = nx.read_edgelist(input_file, nodetype=int, data=(('weight',float),),create_using=nx.DiGraph())
    else:
        Graph_obj = nx.read_edgelist(input_file,nodetype=int,create_using=nx.DiGraph())
        print('\n Edge weighting \n')
        for edge in tqdm(Graph_obj.edges()):
            Graph_obj[edge[0]][edge[1]]['weight'] = 1.0
    
    if not is_directed:
        Graph_obj = Graph_obj.to_directed()
    
    return Graph_obj
    
def alias_setup(probs):
    
    ''' This is an old but efficient technique to create uniform probability distribution and then we calculate
        samples based on the uniform distribution. This is not necessary, for smaller graphs, but when the 
        graph size increases this method is an efficient approach to in deciding the probabilities'''
        
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
	        larger.append(kk)
         
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
	        smaller.append(large)
        else:
	        larger.append(large)

    return J,q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
 
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]   
    
def tab_printer(args):

    print('\n Arguments Passed : \n ')
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())          