"""
Betweenness centrality measures.
"""
#    Copyright (C) 2004-2011 by 
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    Bridgeness extensions Matteo Morini <matteo.morini@ens-lyon.fr>
#    All rights reserved.
#    BSD license.
import heapq
import networkx as nx
import random
__author__ = """Aric Hagberg (hagberg@lanl.gov), Matteo Morini (matteo.morini@ens-lyon.fr)"""

__all__ = ['betweenness_centrality',
           'bridgeness_centrality',
           'edge_betweenness_centrality',
           'edge_bridgeness_centrality']

global_distances_dict={}
global_degrees_dict={}


def betweenness_centrality(G, k=None, normalized=True, weight=None, 
                           endpoints=False, 
                           seed=None):

    betweenness=dict.fromkeys(G,0.0) # b[v]=0 for v in G
    
    if k is None:
        nodes = G
    else:
        random.seed(seed)
        nodes = random.sample(G.nodes(), k)
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S,P,sigma=_single_source_shortest_path_basic(G,s)
        else:  # use Dijkstra's algorithm
            S,P,sigma=_single_source_dijkstra_path_basic(G,s,weight)
        # accumulation
        if endpoints:
            betweenness=_accumulate_endpoints(betweenness,S,P,sigma,s)
        else:
            betweenness=_accumulate_basic(betweenness,S,P,sigma,s)
    # rescaling
    betweenness=_rescale(betweenness, len(G),
                         normalized=normalized,
                         directed=G.is_directed(),
                         k=k)
    return betweenness


def bridgeness_centrality(G, k=None, normalized=True, weight=None, 
                           endpoints=False, 
                           seed=None):

    """
    Compute the shortest-path bridgeness EXN centrality for nodes.
    """

    global global_distances_dict

    bridgeness=dict.fromkeys(G,0.0) # b[v]=0 for v in G

    if k is None:
        nodes = G
    else:
        random.seed(seed)
        nodes = random.sample(G.nodes(), k)
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S,P,sigma=_single_source_shortest_path_basic(G,s)
        else:  # use Dijkstra's algorithm
            S,P,sigma=_single_source_dijkstra_path_basic(G,s,weight)
        # accumulation
        if endpoints:
            bridgeness=_accumulate_endpoints(bridgeness,S,P,sigma,s)
        else:
            bridgeness=_accumulate_basic_bridge(bridgeness,S,P,sigma,s)
    # rescaling
    bridgeness=_rescale(bridgeness, len(G),
                         normalized=normalized,
                         directed=G.is_directed(),
                         k=k)
    return bridgeness



def edge_betweenness_centrality(G,normalized=True,weight=None):
 
    betweenness=dict.fromkeys(G,0.0) # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    betweenness.update(dict.fromkeys(G.edges(),0.0))
    for s in G:
        # single source shortest paths
        if weight is None:  # use BFS
            S,P,sigma=_single_source_shortest_path_basic(G,s)
        else:  # use Dijkstra's algorithm
            S,P,sigma=_single_source_dijkstra_path_basic(G,s,weight)
        # accumulation
        betweenness=_accumulate_edges(betweenness,S,P,sigma,s)
    # rescaling
    for n in G: # remove nodes to only return edges 
        del betweenness[n]
    betweenness=_rescale_e(betweenness, len(G),
                           normalized=normalized,
                           directed=G.is_directed())
    return betweenness

def edge_bridgeness_centrality(G,normalized=True,weight=None):
 
    bridgeness=dict.fromkeys(G,0.0) # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    bridgeness.update(dict.fromkeys(G.edges(),0.0))
    for s in G:
        # single source shortest paths
        if weight is None:  # use BFS
            S,P,sigma=_single_source_shortest_path_basic(G,s)
        else:  # use Dijkstra's algorithm
            S,P,sigma=_single_source_dijkstra_path_basic(G,s,weight)
        # accumulation
        bridgeness=_accumulate_edges_bridge(bridgeness,S,P,sigma,s)
    # rescaling
    for n in G: # remove nodes to only return edges 
        del bridgeness[n]
    bridgeness=_rescale_e(bridgeness, len(G),
                           normalized=normalized,
                           directed=G.is_directed())
    return bridgeness



# helpers for betweenness centrality

def _single_source_shortest_path_basic(G,s):

    global global_distances_dict
    global global_degrees_dict
    
    #for n in G.nodes():
    global_distances_dict = dict(dict(nx.shortest_path_length(G)))
    global_degrees_dict = nx.degree(G)

    
    S=[]
    P={}
    for v in G:
        P[v]=[]
    sigma=dict.fromkeys(G,0.0)    # sigma[v]=0 for v in G
    D={}
    sigma[s]=1.0
    D[s]=0
    Q=[s]
    while Q:   # use BFS to find shortest paths
        v=Q.pop(0)
        S.append(v) #########
        Dv=D[v]
        sigmav=sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w]=Dv+1
            if D[w]==Dv+1:   # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v) # predecessors 
                
    #global_distances_dict.setdefault(s, D) #store distances for node s 
    return S,P,sigma #also distances, needed for bridgeness



def _single_source_dijkstra_path_basic(G,s,weight='weight'):
    # modified from Eppstein
    S=[]
    P={}
    for v in G:
        P[v]=[]
    sigma=dict.fromkeys(G,0.0)    # sigma[v]=0 for v in G
    D={}
    sigma[s]=1.0
    push=heapq.heappush
    pop=heapq.heappop
    seen = {s:0}
    Q=[]   # use Q as heap with (distance,node id) tuples
    push(Q,(0,s,s))
    while Q:
        (dist,pred,v)=pop(Q)
        if v in D:
            continue # already searched this node.
        sigma[v] += sigma[pred] # count paths
        S.append(v)
        D[v] = dist
        for w,edgedata in G[v].items():
            vw_dist = dist + edgedata.get(weight,1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q,(vw_dist,v,w))
                sigma[w]=0.0
                P[w]=[v]
            elif vw_dist==seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    return S,P,sigma,D

def _accumulate_basic(betweenness,S,P,sigma,s):
    delta=dict.fromkeys(S,0)
    while S:
        w=S.pop()
        coeff=(1.0+delta[w])/sigma[w]
        for v in P[w]:
            delta[v] += sigma[v]*coeff
        if w != s:
            betweenness[w]+=delta[w]
    return betweenness

def _accumulate_basic_bridge(bridgeness,S,P,sigma,s):

    global global_distances_dict    
    
    delta=dict.fromkeys(S,0)
    delta_bri=dict.fromkeys(S,0)
    while S:
        w=S.pop()
        coeff=(1.0+delta[w])/sigma[w]
        coeff_bri=(delta[w])/sigma[w] #1.0 removed
        
        for v in P[w]:
            #print "sigmav[",v,"] ", sigma[v], " sigmaw[",w,"] ", sigma[w], " deltav[",v,"] ", delta[v]
            delta[v] += sigma[v]*coeff
            
            delta_bri[v] += sigma[v]*coeff_bri

        #if w != s: #no no, must be if d[w_index]>1
        #print(s, w, dict(dict(global_distances_dict)[s])[w], " apart")
        #print( dict( dict(global_distances_dict)[s] ) [w], s,w )
        if ( global_distances_dict[s][w] > 1 ): #no no, must be if d[w_index]>1 --> dist(s,w)>1
            bridgeness[w]+=delta_bri[w]
    
    return bridgeness

def _accumulate_endpoints(betweenness,S,P,sigma,s):
    betweenness[s]+=len(S)-1
    delta=dict.fromkeys(S,0)
    while S:
        w=S.pop()
        coeff=(1.0+delta[w])/sigma[w]
        for v in P[w]:
            delta[v] += sigma[v]*coeff
        if w != s:
            betweenness[w] += delta[w]+1
    return betweenness

def _accumulate_edges(betweenness,S,P,sigma,s):
    delta=dict.fromkeys(S,0)
    while S:
        w=S.pop()
        coeff=(1.0+delta[w])/sigma[w]
        for v in P[w]:
            c=sigma[v]*coeff
            if (v,w) not in betweenness:
                betweenness[(w,v)]+=c
            else:
                betweenness[(v,w)]+=c
            delta[v]+=c
        if w != s:
            betweenness[w]+=delta[w]
    return betweenness

def _accumulate_edges_bridge(bridgeness,S,P,sigma,s):

    global global_distances_dict
    global global_degrees_dict
    
    delta=dict.fromkeys(S,0)
    while S:
        w=S.pop()
        coeff=(1.0+delta[w])/sigma[w]

        counter = 0        
        for s1 in S:
            #print "...s1 =", s1, " w= ", w, "P= ", P, "P[s1]= ", P[s1]
            #if (w == P[s1]):
            if (w in P[s1]):
                #print "---- s1 =", s1, " w= ", w
                counter += 1
                
        #print "W = ", w, " counter = ", counter
        
        #print "degree w = ", global_degrees_dict[w]
        coeff_bri=(delta[w]-global_degrees_dict[w]+counter)/sigma[w]                
        
        for v in P[w]:
            #print "EDGE-v/w", v, w
            c=sigma[v]*coeff

            c_bri=sigma[v]*coeff_bri

            if(w,s) not in global_distances_dict:
                d1 = global_distances_dict[s][w]
            else:
                d1 = global_distances_dict[w][s] 
                    
            #print "D1 === ", d1
            if (d1 > 2) & (c_bri > 0.0) :
                #print "EDGE + c_bri ", c_bri

                if (v,w) not in bridgeness:
                    bridgeness[(w,v)]+=c_bri
                else:
                    bridgeness[(v,w)]+=c_bri

            delta[v]+=c
            #print "V=",v,"delta[v]", delta[v], "c_bri",c_bri
            #print "-------------------------------"
            
        if w != s:
        #if global_distances_dict[s][w] > 2:
            bridgeness[w]+=delta[w]
            
    return bridgeness

def _rescale(betweenness,n,normalized,directed=False,k=None):
    if normalized is True:
        if n <=2:
            scale=None  # no normalization b=0 for all nodes
        else:
            scale=1.0/((n-1)*(n-2))
    else: # rescale by 2 for undirected graphs
        if not directed:
            scale=1.0/2.0
        else:
            scale=None
    if scale is not None:
        if k is not None:
            scale=scale*n/k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness

def _rescale_e(betweenness,n,normalized,directed=False):
    if normalized is True:
        if n <=1:
            scale=None  # no normalization b=0 for all nodes
        else:
            scale=1.0/(n*(n-1))
    else: # rescale by 2 for undirected graphs
        if not directed:
            scale=1.0/2.0
        else:
            scale=None
    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness
