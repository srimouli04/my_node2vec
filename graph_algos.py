import numpy as np
import random
from tqdm import tqdm
import networkx as nx
from utilities import read_graph
from gensim.models import Word2Vec 

''' This module concentrates on building the second order Random walks needed for Node2vec alogorithm to work'''
'''This Class describes the Node2vec implementation'''
class node2vec:
    def __init__(self,G,p,q):
        '''
        Paramters
        g - graph object
        p - Parameter controlling BFS
        q - Parameter controlling DFS
        num_walks - number of walks
        walk_length - Walk length
        '''
        self.G = G
        self.nodes = nx.nodes
        print('Edge weightining \n')
        for edge in tqdm(self.G.edges()):
            self.G[edge[0]][edge[1]]['weight'] = 1.0
            self.G[edge[1]][edge[0]]['weight'] = 1.0
        self.p = p
        self.q = q
        #self.prep_trans_prob()
        #self.simulatewalks()
        
    def node2vecwalk(self,walk_length,start_node):
        '''As part of this function we are trying to generate random walks given a node
        we'll be using alias sampling here inorder to improve the efficiency of algorithm to 
        when calculating the transitional probabilities to be O(1), I will explain it in detail 
        in alias_setup procedure'''
        
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        
        walk = [start_node]
        
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbours(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0],alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev,cur)[0]],alias_edges[(prev,cur)[1]])]
                    walk.append(next)
            else:
                break
        return walk
    
    def simulate_walks(self,num_walks,walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('walk-iteration')
        for walk_iter in tqdm(range(num_walks)):
            random.shuffle(nodes)
            print(str(walk_iter+1)), '/',str(num_walks)
            for node in tqdm(nodes):
                walks.append(self.node2vecwalk(walk_length=walk_length,start_node=node))
        return walks
        
    def get_alias_edge(self,src,dst):
        '''
        Get alias edge  for a given edge
        '''
        
        G = self.G
        p=self.p
        q=self.q
        
        unnormalised_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalised_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr,src):
                unnormalised_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalised_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalised_probs)
        return unnormalised_probs
    
        
    def generate_node2vec_embeddings(self,walks,args):
        self.walks = walks
        walks = [map(str,walk) for walk in walks]
        model = Word2Vec(walks,size=args.dimensions,window=args.window_size,min_count=0,sg=1,workers=args.workes
                     ,iter = args.iter)
        model.save_word2vec_format(args.output)
        return 

    def prep_trans_prob(self):
        
        '''
        Calculating the transitional probabilities before hand
        '''

        G = self.G
        is_directed = self.is_directed
        
        alias_nodes ={}
        for node in G.nodes():
            unnormalised_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalised_probs)
            normalised_probs = [float(u_prob)/norm_const for u_prob in unnormalised_probs]
            alias_nodes[node] = alias_setup(normalised_probs)
            
        alias_edges = {}
        
        if is_directed:
            for edge in G.edges():
                alias_edge[edge] = self.get_alias_edge(edge[0],edge[1])
        else:
            for edge in G.edges():
                alias_edge[edge] = self.get_alias_edge(edge[0],edge[1])
                alias_edge[(edge[1],edge[0])] = self.get_alias_edge(edge[1],edge[0])     
                
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        
        return
    
    def alias_setup(self,probs):
        
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
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

        return J, q

    def alias_draw(self,J, q):
        '''
        Draw sample from a non-uniform discrete distribution using alias sampling.
        '''
        K = len(J)

        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]   