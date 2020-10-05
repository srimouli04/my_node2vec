import numpy as np
import random
from tqdm import tqdm
import networkx as nx
from utilities import alias_setup,alias_draw
from gensim.models import Word2Vec 

''' This module concentrates on building the second order Random walks needed for Node2vec alogorithm to work'''
'''This Class describes the Node2vec implementation'''
class nd2vec:
    def __init__(self,args,G):
        '''
        Paramters
        args - all the arguments 
        G - Graph object
        '''
        
        self.G = G
        self.args = args
        self.is_directed = args.directed
        self.p = args.p
        self.q = args.q
        self.workers = args.workers
        
    def nd2vec_wk(self,walk_length,start_node):
        '''As part of this function we are trying to generate random walks given a node
        we'll be using alias sampling here inorder to improve the efficiency of algorithm to 
        when calculating the transitional probabilities to be O(1), I will explain it in detail 
        in alias_setup procedure'''
        
        G = self.G
        alias_nodes = self.alias_nodes
        weighted_edge = self.weighted_edge
        
        walk = [start_node]
        
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0],alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(weighted_edge[(prev, cur)][0], weighted_edge[(prev, cur)][1])]
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
                walks.append(self.nd2vec_wk(walk_length=walk_length,start_node=node))
        return walks
        
    def get_weighted_edge(self,src,dest):
        '''
        Get alias edge  for a given edge
        '''
        
        G = self.G
        p=self.p
        q=self.q
        
        unnormalised_probs = []
        for dest_nbr in sorted(G.neighbors(dest)):
            if dest_nbr == src:
                unnormalised_probs.append(G[dest][dest_nbr]['weight']/p)
            elif G.has_edge(dest_nbr,src):
                unnormalised_probs.append(G[dest][dest_nbr]['weight'])
            else:
                unnormalised_probs.append(G[dest][dest_nbr]['weight']/q)
        norm_const = sum(unnormalised_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalised_probs]
        
        return alias_setup(normalized_probs)

    
    def generate_nd2vec_embeddings(self,walks):
        
        walks = list(map(str,(walk for walk in walks)))
        model = Word2Vec(walks,size=self.args.dimensions,window=self.args.window_size,min_count=0,sg=1,workers=self.args.workers
                     ,iter = self.args.iter)
        model.wv.save_word2vec_format(self.args.output)
        return 

    def prep_trans_prob(self):
        
        '''
        Calculating the transitional probabilities
        '''

        G = self.G
        is_directed = self.is_directed
        
        alias_nodes ={}
        for node in G.nodes():
            unnormalised_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalised_probs)
            normalised_probs = [float(u_prob)/norm_const for u_prob in unnormalised_probs]
            alias_nodes[node] = alias_setup(normalised_probs)
            
        weighted_edge = {}
        
        if is_directed:
            for edge in G.edges():
                weighted_edge[edge] = self.get_weighted_edge(edge[0],edge[1])
        else:
            for edge in G.edges():
                weighted_edge[edge] = self.get_weighted_edge(edge[0],edge[1])
                weighted_edge[(edge[1],edge[0])] = self.get_weighted_edge(edge[1],edge[0])     
                
        self.alias_nodes = alias_nodes
        self.weighted_edge = weighted_edge
        
        return


