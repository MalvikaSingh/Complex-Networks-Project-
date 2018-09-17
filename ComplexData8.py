import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
import StringIO
import matplotlib.pyplot as plt
import random
import numpy as np
from operator import itemgetter

def efficiency(G):
    #print "Number of nodes"
    N= G.number_of_nodes() 
    #print N
   
    E=0.0;
    length=nx.all_pairs_dijkstra_path_length(G)
    #print length

    for key,val in length.iteritems():
        for key2,val2 in val.iteritems():
            if val2>0:
                #print "in if"
                E=E+(1.0/val2);
 
    print "efficiency of network"
    E=E/((N)*(N-1))
    print E

def updateEdgeWeight(G,i):
    #print "Update edge weight"
    l=nx.get_edge_attributes(G, 'weight')
    #print l

    for key1,key2 in l:
        #print key1
        #print key2
        #print l[(key1,key2)]
        x=key1
        y=key2
        if (x==i) or (y==i):
            G.edge[x][y]['weight']=(l[(key1,key2)]*G.node[i]['threshold'])/G.node[i]['betweenness_centrality']
    #nx.set_edge_attributes(G, 'weight', (l.get('weight')*G.node[i]['threshold'])/G.node[i]['betweenness_centrality'])
    


def removeNode(x,G):
    
    print "Removing node="+str(x)

    for i in G.nodes():
        if G.has_edge(i,x):
            G.remove_edge(i,x) 
        if G.has_edge(x,i):
            G.remove_edge(x,i) 
    
    betweennessCentralityNodes(G) 

    H=G.copy()

    for i in G.nodes():
        if G.node[i]['threshold']<G.node[i]['betweenness_centrality']:
            updateEdgeWeight(G,i)
    
    print "G: Before mitigation"
    efficiency(G);

    #Mitigation on graph H:
    #Mitigation
    load_rem_arr=[0.3,0.85,0.95]
    for rem in load_rem_arr:
        H_temp=H.copy()
        print "rem="+str(rem)
        for i in H_temp.nodes():
            H_temp.node[i]['betweenness_centrality']=H_temp.node[i]['betweenness_centrality']*rem
        for i in H_temp.nodes():
            if H_temp.node[i]['threshold']<H_temp.node[i]['betweenness_centrality']:
                updateEdgeWeight(H_temp,i)
        print "H: After mitigation"
        efficiency(H_temp);

def printAdjacencyMatrix(G):
    #Adjacency Matrix
    A=nx.adjacency_matrix(G);
    print(A.todense())

def printLenofComponnets(G):
    #Length of componenets
    print "Componenets:"
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        print len(c) 

def betweennessCentralityNodes(G):
    #   Betweenness centrality of a node is the fraction of all shortest paths that pass through that node.
    nodes_bc= nx.betweenness_centrality(G);
    nx.set_node_attributes(G, 'betweenness_centrality', nodes_bc)
    
   # print "Betwenness centrality"

    #for k, v in nodes_bc.iteritems():
     #   print k, v
    
    
    return nodes_bc

def checkThreshold(G):
    for i in G.nodes():
        if G.node[i]['betweenness_centrality']>G.node[i]['threshold']:
            removeNode(i,G);
            checkThreshold(G);
            return;

def drawGraph(G):

    nx.draw_random(G,node_size=1)
    plt.savefig("/Users/SmritiSharma/Desktop/Projects/Networks/Graph.png") # save as png
    plt.show() # display


def degreeDistribution(G):
    
    #Degree Distribution
    degs = defaultdict(int)
    for i in G.degree().values(): degs[i]+=1
    items = sorted ( degs.items () )
    x, y = np.array(items).T
    y = [float(i) / sum(y) for i in y]
    plt.plot(x, y, 'bo')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['Degree'])
    plt.xlabel('$K$', fontsize = 20)
    plt.ylabel('$P_K$', fontsize = 20)
    plt.title('$Degree\,Distribution$', fontsize = 20)
    plt.show()
    '''
    #Or
    
    #Degree Distribution
    in_degrees = G.degree() # dictionary node:degree
    in_values = sorted(set(in_degrees.values()))
    in_hist = [in_degrees.values().count(x) for x in in_values]
    plt.figure()
    plt.plot(in_values,in_hist,'ro-') # in-degree
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('US Powergrid Degree Distribution')
    plt.savefig('D:\Sem7\ComplexNetworks\Project\Powergrid\opsahl-powergrid\hartford_degree_distribution.png')
    plt.close()
    '''

    




'''Code'''
#Taake input
raw = []
with open('/Users/SmritiSharma/Desktop/Projects/Networks/data.txt','r') as f:
    for line in f:
        raw.append(line)
  #  print raw

g = nx.parse_edgelist(raw, nodetype = int)

'''
print "No of nodes="
print g.number_of_nodes() # also g.order()
print "No of edges="
print g.number_of_edges() # also g.size()
'''

#Efficiency of the network


#Crucitii:Fail a node
#Assign initial load
bc=betweennessCentralityNodes(g)
#Assign initial weight=1 for all edges
nx.set_edge_attributes(g, 'weight', 1)
#print nx.get_edge_attributes(g,'weight')

#efficiency(g);


#for load_based_removal
alpha_arr=[1.1]

#betweenness_dict = nx.betweenness_centrality(g)  # Run betweenness centrality
        # Assign each to an attribute in your network
#nx.set_node_attributes(g, 'betweenness', betweenness_dict)

sorted_betweenness = sorted(bc.items(), key=itemgetter(1), reverse=True)
        #print("Top 20 nodes by betweenness centrality:")
the_list = []
for key,val in sorted_betweenness[:20]:
    the_list.append(key) #Extract 1st row node number
    

for alpha in alpha_arr:
#alpha is the tolearance parameter
    print "alpha="+str(alpha)
    #Threshhold capacity for each node=t
    t=bc;
    for key in t:
        t[key] = t[key] * alpha;

    nx.set_node_attributes(g, 'threshold', t)

    #print "Threshold after"
    #for i in g.nodes():
     #   print g.node[i]['threshold']

    N=g.number_of_nodes() # also g.order()
    arr=[1,2,3]
    
    
    for i in arr:
    #node to be failed=x
        print "Remove node iter="+str(i)
        #x= random.randint(1,N)#To be chosen randomly
        x = random.choice(the_list)  # To be chosen randomly
        Graph_removedNode=g.copy()
        removeNode(x, Graph_removedNode);
'''
#Average degree of graph
#Clustering coefficient for each node
#print(nx.clustering(G))
#print(nx.average_clustering(G))

#printAdjacencyMatrix(g)

#Print edges
for e in G.edges():
    print e
print G.nodes()

print G


'''