
# coding: utf-8

# In[ ]:

import argparse

import pandas as pd
import numpy as np
import random
import networkx as nx
import bridgeness
from numpy.random import choice, uniform

from multiprocessing import Pool, TimeoutError
import time
import os

from joblib import Memory
from collections import defaultdict

import sys
#sys.argv = ['foo'] #to make parseargs work in ipython too


# In[ ]:

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--graph", type=str, help="Graph filename", default='graph2.txt.csv')
parser.add_argument("--num_steps", type=int, help="Number of steps", default=10000)
parser.add_argument("--check_every" , type=int, help="Check for convergence every N steps", default=100)
parser.add_argument("--phi" , type=float, help="Phi marameter", default=100)
parser.add_argument("--mu" , type=float, help="Mu parameter", default=0.2)
parser.add_argument("--repeats" , type=int, help="Number of parallel runs", default=4)
parser.add_argument("--conv_thr" , type=float, help="Convergence threshold", default=1.0e-6)

args = parser.parse_args()
print(args)


# In[ ]:

FN = args.graph
data = pd.read_csv(FN, delimiter = " ")

G = nx.Graph()

for i,s in data.iterrows():
    u = s[0]
    v = s[1]

    G.add_edge(u,v)


# In[ ]:

memory = Memory(cachedir='./cached_bri', verbose=0)

@memory.cache
def cached_bridgeness(G):
    print("Computing centrality metric (cache miss), please be patient.")
    return bridgeness.bridgeness_centrality(G)

bri = cached_bridgeness(G)
#bri = bridgeness.bridgeness_centrality(G)
#e_bri = bridgeness.edge_bridgeness_centrality(G)
#bet = bridgeness.betweenness_centrality(G)


# In[ ]:

phi = args.phi
mu = args.mu
#phi = 100
#mu = 0.2

#lab0 = 81  #82 in matlab
#lab1 = 190 #191 in matlab
labs = ((81, 0), (190, 1)) #seed nodes: community

adj = nx.to_numpy_matrix(G)
bri_vals = list(bri.values())
quality = (1 / np.exp( np.multiply( bri_vals, phi )))
diag = np.diagflat(quality) #k: v = node, bridgeness
_tmp = np.matmul(diag, adj)
_tmp_sum_on_rows = np.sum(_tmp, axis=0)
_tmp_sum_on_rows_recip = np.reciprocal(_tmp_sum_on_rows)
norm = np.diagflat( _tmp_sum_on_rows_recip )

T = np.matmul( _tmp, norm )

num_nodes = len(G.nodes())
alpha = 2.0/(2.0 + mu)


# In[ ]:

steps = args.num_steps
convcheckfreq = args.check_every
conv_thr = args.conv_thr #1e-06

repeatz = args.repeats

print("Alpha = %.3f" %(alpha))

def rw(sn): #signed node
    n0 = sn
    print("RW from signed node %i started." % n0)
    rw_visits = np.zeros( num_nodes )
    last_rw_visits = np.copy(rw_visits)

    nodes = range(num_nodes)

    for s0 in range(1,steps+1):

        #start from
        #go back to labeled node? prob = 1-alpha; prob to trans = alpha
        if (uniform() < alpha): #
            trans_probs = T[:,n0].view(np.ndarray).flatten() 
            trans_to = int( choice(nodes, 1, p = trans_probs) )
            n0 = trans_to
        else:
            n0 = sn

        rw_visits[n0] += 1

        if (s0 % convcheckfreq == 0):
            diff_rw_visits = ((rw_visits/s0 - last_rw_visits/ (s0 - convcheckfreq) )**2).sum()
            print("[sn %d] At step %d, diff %.10f" %(sn, s0, diff_rw_visits) )
            if ((diff_rw_visits) < conv_thr):# and np.all(rw_visits != last_rw_visits)):
                print("Converged at <%.10f" % conv_thr)
                break
            last_rw_visits = np.copy(rw_visits)

    return rw_visits/s0, s0

#output
seed_nodes_seq = []
communities_seq = []

for _sn, _comm in labs*repeatz:
    seed_nodes_seq.append(_sn)
    communities_seq.append(_comm)

if __name__ == '__main__':
    with Pool() as pool:
        res = pool.map(rw, seed_nodes_seq)

#consolidate results
all_prob_distr = defaultdict(list) #k = community : v = seednode, steps, probdistr

for r, (prob, steps) in enumerate(res):
    print("RW %d (seed node %d, comm %d) run for %d steps" % (r, seed_nodes_seq[r], communities_seq[r], steps))
    all_prob_distr[communities_seq[r]].append((seed_nodes_seq[r], steps, prob))
    
#save as numpy, just in case
with open('probs.npy', 'wb') as f:
    np.save(f, all_prob_distr)


# In[ ]:

comms_keys = all_prob_distr.keys()

#list of lists: [community][repetition]
probs_vectors_by_comm_by_repeat = { ck: [all_prob_distr[ck][rep][2] for rep in range(repeatz)] for ck in comms_keys}
probs_weights_by_comm_by_repeat = { ck: [all_prob_distr[ck][rep][1] for rep in range(repeatz)] for ck in comms_keys}

#average by community (need to weight by steps before convergence? I'd say no!)
avg_probs = {ck: np.average(probs_vectors_by_comm_by_repeat[ck], axis = 0) for ck in comms_keys}
                            #weights = probs_weights_by_comm_by_repeat[ck]) for ck in comms_keys}


# In[ ]:

OUT_FN = 'RW_avg_probs.csv'
pd.DataFrame(avg_probs).to_csv(OUT_FN)
print("Averaged RW probabilities dumped to file %s." % OUT_FN)

