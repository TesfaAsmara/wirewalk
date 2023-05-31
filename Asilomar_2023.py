# Create the following folders to store the experiment results for each of the BlogCatalog, Protein-Protein Interaction, and Wikipedia datasets using the following commands:

# ```
# mkdir data
# mkdir blogcatalog
# mkdir Homo_sapiens
# mkdir POS
# ```

# Download the BlogCatalog, Protein-Protein Interaction, and Wikipedia datasets from https://snap.stanford.edu/node2vec/ into the data folder we just created using the following commands:

# ```
# cd data/
# wget http://socialcomputing.asu.edu/datasets/BlogCatalog3
# wget https://snap.stanford.edu/node2vec/Homo_sapiens.mat
# wget https://snap.stanford.edu/node2vec/POS.mat
# ```

# You will also need an environment with the following installed

# - Python3
# - [scikit-learn](https://scikit-learn.org/stable/install.html)
# - [Networkx](https://networkx.org/documentation/stable/install.html)
# - [Numpy](https://numpy.org/install/)
# - [Gensim](https://pypi.org/project/gensim/)
# - [Joblib](https://joblib.readthedocs.io/en/latest/installing.html)
# - [editdistance](https://pypi.org/project/editdistance/)
# - [node2vec](https://github.com/eliorc/node2vec)

# You can then change the `dataset` variable in the following code to one of `blogcatalog`, `Homo_sapiens`, or `POS` and change the `funcs` list to contain the transformations that you would like results on.

import scipy.io
import networkx as nx
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec
import json
import os
from joblib import Parallel, delayed
import time
import math
import editdistance
import warnings

# Load the MATLAB file into a Python object
dataset = "blogcatalog"
mat_data = scipy.io.loadmat(f'data/{dataset}.mat')

# Extract the adjacency matrix from the object
adj_matrix = mat_data['network']

group_labels = mat_data['group']

# Get the index of the maximum value for each row
labels = group_labels.argmax(axis=1).A1

# Convert the adjacency matrix to a NetworkX graph
G = nx.from_numpy_array(adj_matrix)

nx.set_edge_attributes(G, nx.get_edge_attributes(G, "weight"), "capacity")

# Print some basic information about the graph
print(nx.info(G))  


def max_flow(G, current_node, destination):
    if current_node != destination:
        ss_weight, _ = nx.maximum_flow(G, current_node, destination)
    else:
        ss_weight = 0
    return ss_weight

def min_cost_max_flow(G, current_node, destination):
    if current_node != destination:
        mincostFlow = nx.max_flow_min_cost(G, current_node, destination)
        ss_weight = nx.cost_of_flow(G, mincostFlow)
    else:
        ss_weight = 0
    return ss_weight 

def jaccard_coefficient(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        denominator = len(curr)+len(dest)
        if denominator == 0:
            ss_weight = 0
        else:
            numerator = len(set(curr+dest))
            ss_weight = numerator/denominator
    else:
        ss_weight = 0
    return ss_weight 

def adamic_adar(G, current_node, destination, num_jobs=-1):
    def calc_weight(node):
        denominator = math.log10(len(list(G.neighbors(node))))
        if denominator == 0:
            return 0
        else:
            return 1 / denominator
        
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        intersection = set(curr+dest)
        
        adamic_adar_indices = sum(Parallel(n_jobs=num_jobs)(
            delayed(calc_weight)(node) for node in intersection
        ))
        
        ss_weight = adamic_adar_indices
    else:
        ss_weight = 0
    return ss_weight

def common_neighbors(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        ss_weight = len(set(curr+dest))
    else:
        ss_weight = 0
    return ss_weight 

def lhn_index(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        denominator = len(curr)*len(dest)
        if denominator == 0:
            ss_weight = 0
        else:
            numerator = len(set(curr+dest))
            ss_weight = numerator/denominator
    else:
        ss_weight = 0
    return ss_weight 

def preferential_attachment(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        ss_weight = len(curr) * len(dest)
    else:
        ss_weight = 0
    return ss_weight 

def hub_promoted(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        denominator = min(len(curr),len(dest))
        if denominator == 0:
            ss_weight = 0
        else:
            numerator = len(set(curr+dest))
            ss_weight = numerator/denominator
    else:
        ss_weight = 0
    return ss_weight 

def hub_depressed(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        denominator = max(len(curr),len(dest))
        if denominator == 0:
            ss_weight = 0
        else:
            numerator = len(set(curr+dest))
            ss_weight = numerator/denominator
    else:
        ss_weight = 0
    return ss_weight 

def salton_index(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        denominator = math.sqrt(len(curr)*len(dest))
        if denominator == 0:
            ss_weight = 0
        else:
            numerator = len(set(curr+dest))
            ss_weight = numerator/denominator
    else:
        ss_weight = 0
    return ss_weight 

def sorenson_index(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        denominator = len(curr)+len(dest)
        if denominator == 0:
            ss_weight = 0
        else:
            numerator = 2*len(set(curr+dest))
            ss_weight = numerator/denominator
    else:
        ss_weight = 0
    return ss_weight 

def resource_allocation(G, current_node, destination, num_jobs=-1):
    def calc_weight(node):
        denominator = len(list(G.neighbors(node)))
        if denominator == 0:
            return 0
        else:
            return 1 / denominator
        
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        intersection = set(curr+dest)
        
        resource_allocation_indices = sum(Parallel(n_jobs=num_jobs)(
            delayed(calc_weight)(node) for node in intersection
        ))
        
        ss_weight = resource_allocation_indices
    else:
        ss_weight = 0
    return ss_weight

def levenshtein(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        ss_weight = editdistance.eval(curr, dest)
    else:
        ss_weight = 0
    return ss_weight

def tversky(G, current_node, destination):
    if current_node != destination:
        curr = list(G.neighbors(current_node))
        dest = list(G.neighbors(destination))
        a = set(curr)
        b = set(dest)
        union = a.union(b)
        a_compl = a.difference(b)
        b_compl = b.difference(a)
        denominator = len(union) + len(curr)*len(a_compl)+len(dest)*len(b_compl)
        if denominator == 0:
            ss_weight = 0
        else:
            numerator = len(a.intersection(b))
            ss_weight = numerator/denominator
    else:
        ss_weight = 0
    return ss_weight

def speed_up(G, num_workers, transition_matrix_function):
    nodes = G.nodes
    # Split the nodes into chunks for each worker
    node_chunks = np.array_split(nodes, num_workers)

    # Use joblib to parallelize the calculation of ss_weight
    ss_weights = Parallel(n_jobs=num_workers)(
        delayed(transition_matrix_function)(G, current_node, destination)
        for chunk in node_chunks
        for current_node in chunk
        for destination in nodes
    )

    # Reshape the ss_weights list into a matrix
    ss_weights_matrix1 = np.reshape(ss_weights, (len(nodes), len(nodes)))
    # ss_weights_matrix1 = ss_weights_matrix1/ss_weights_matrix1.sum(axis=1, keepdims=True)


    # check if row sums are zero
    row_sums = ss_weights_matrix1.sum(axis=1)
    zero_rows = np.where(row_sums == 0)[0]

    # set all elements in zero rows to zero, except for diagonal element
    ss_weights_matrix1[zero_rows, :] = 0
    for i in zero_rows:
        ss_weights_matrix1[i, i] = 1

    row_sums = ss_weights_matrix1.sum(axis=1)
    # normalize matrix by row sums
    ss_weights_matrix1 = np.divide(ss_weights_matrix1, row_sums[:, np.newaxis])
    return ss_weights_matrix1

def sparse_speed_up(G, k, num_workers, transition_matrix_function):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        A = nx.adjacency_matrix(G)
    A.data = np.ones_like(A.data)
    S = A.copy()
    for i in range(k):
        S = A + np.multiply(S,A)

#     print("Sparse Matrix Shape:")
#     print(A.shape, S.shape, A.nnz, S.nnz)

    ss_weights_matrix1 = np.zeros(S.shape)
    for src_idx in range(S.shape[0]):
        for dest_idx in range(S.shape[0]):
            if S[src_idx, dest_idx] > 0:
                ss_weights_matrix1[src_idx, dest_idx] = transition_matrix_function(G, src_idx, dest_idx)

    # check if row sums are zero
    row_sums = ss_weights_matrix1.sum(axis=1)
    zero_rows = np.where(row_sums == 0)[0]

    # set all elements in zero rows to zero, except for diagonal element
    ss_weights_matrix1[zero_rows, :] = 0
    for i in zero_rows:
        ss_weights_matrix1[i, i] = 1

    # normalize matrix by row sums
    row_sums = ss_weights_matrix1.sum(axis=1)
    ss_weights_matrix1 = np.divide(ss_weights_matrix1, row_sums[:, np.newaxis])

    return ss_weights_matrix1

def generate_walks(graph, num_walks, walk_length, transition_probs, num_jobs=-1):
    walks = []
    nodes = list(graph.nodes())

    # Convert the transition probabilities to a dictionary of dictionaries for faster access
    probs = {}
    for i, node_i in enumerate(nodes):
        probs[node_i] = {}
        for j, node_j in enumerate(nodes):
            probs[node_i][node_j] = transition_probs[i][j]

    def generate_walks_for_node(node):
        node_walks = []
        for walk in range(num_walks):
            walk_list = [node]
            for step in range(walk_length - 1):
                neighbors = list(probs[walk_list[-1]].keys())
                probabilities = list(probs[walk_list[-1]].values())
                next_node = np.random.choice(neighbors, p=probabilities)
                walk_list.append(next_node)
            node_walks.append(walk_list)
        return node_walks

    node_walks_list = Parallel(n_jobs=num_jobs)(
        delayed(generate_walks_for_node)(node) for node in nodes)

    for node_walks in node_walks_list:
        walks += node_walks

    return walks

def cluster_scoring(emb, y_true, num_folds=2):
    #Implementing cross validation
    kf = StratifiedKFold(n_splits=num_folds, random_state=None)

    f1_macro_score = []
    f1_micro_score =[]
    fm_score = []
    acc_score = []
    
    for train_index , test_index in kf.split(emb, y_true):
        X_train , X_test = emb[train_index,:], emb[test_index,:]
        y_train , y_test = y_true[train_index], y_true[test_index]

        #model = KMedoids(n_clusters=len(np.unique(y_train)), metric='euclidean').fit(X_train)
        # Create an instance of the logistic regression classifier with L2 regularization
        model = LogisticRegression(penalty='l2', multi_class='ovr', solver='liblinear').fit(X_train, y_train)

        # Compute clustering metrics on test set
        test_labels = model.predict(X_test)
        f1macro_score_test = f1_score(y_test, test_labels, average='macro')
        f1micro_score_test = f1_score(y_test, test_labels, average="micro")
        fm_score_test = fowlkes_mallows_score(y_test, test_labels)
        acc_score_test = accuracy_score(y_test, test_labels, normalize=True)
        
        
        f1_macro_score.append(f1macro_score_test)
        f1_micro_score.append(f1micro_score_test)
        fm_score.append(fm_score_test)
        acc_score.append(acc_score_test)

        
    
    avg_f1_macro_score = sum(f1_macro_score)/num_folds
    avg_f1_micro_score = sum(f1_micro_score)/num_folds
    avg_fm_score = sum(fm_score)/num_folds
    avg_acc_score = sum(acc_score)/num_folds

    
    return 'Avg avg_f1_macro_score : {}'.format(avg_f1_macro_score), \
            'Avg avg_f1_micro_score : {}'.format(avg_f1_micro_score), \
            'Avg fowlkes_mallows_score : {}'.format(avg_fm_score), \
            'Avg avg_acc_score : {}'.format(avg_acc_score)

def getnet(G,func,dataset,labels,num_walks=10, walk_length=80, num_workers=4, window=10,dimension=128, num_folds=2):
    start_time = time.time()
    entry = {}
    func_name = str(func).split()[1]
    print(f"Computing {func_name} transition matrix")
    ss_weights_matrix = speed_up(G,num_workers,func)
    entry["transition_matrix"] = ss_weights_matrix.tolist()
    print(f"Computing {func_name} walks")
    walks = generate_walks(G, num_walks=num_walks, walk_length=walk_length, transition_probs = ss_weights_matrix)
    print(f"Computing {func_name} model")
    model = Word2Vec(walks, window=window, workers=num_workers, vector_size=dimension)
    emb=model.wv[[i for i in model.wv.key_to_index]]
    model.save(f"{dataset}/my_{func_name}_model")
    print(f"Computing {func_name} results")
    results = cluster_scoring(emb,labels,num_folds=num_folds)
    entry["results"] = results
    end_time = time.time()
    time_elapsed = end_time - start_time
    entry["time_elapsed"] = time_elapsed
    with open(f"{dataset}/{func_name}_results.txt", "w") as f:
        json.dump(entry, f)
    return entry

def sparse_getnet(G,k,func,dataset,labels,num_walks=10, walk_length=80, num_workers=4, window=10,dimension=128, num_folds=2):
    start_time = time.time()
    entry = {}
    func_name = str(func).split()[1]
    print(f"Computing {func_name}{k} transition matrix")
    ss_weights_matrix = sparse_speed_up(G,k,num_workers,func)
    entry["transition_matrix"] = ss_weights_matrix.tolist()
    print(f"Computing {func_name}{k} walks")
    walks = generate_walks(G, num_walks=num_walks, walk_length=walk_length, transition_probs = ss_weights_matrix)
    print(f"Computing {func_name}{k} model")
    model = Word2Vec(walks, window=window, workers=num_workers, vector_size=dimension)
    emb=model.wv[[i for i in model.wv.key_to_index]]
    model.save(f"{dataset}/my_{func_name}{k}_model")
    print(f"Computing {func_name}{k} results")
    results = cluster_scoring(emb,labels,num_folds=num_folds)
    entry["results"] = results
    end_time = time.time()
    time_elapsed = end_time - start_time
    entry["time_elapsed"] = time_elapsed
    with open(f"{dataset}/{func_name}{k}_results.txt", "w") as f:
        json.dump(entry, f)
    return entry

def n2vec(G, labels,dimensions=128, p=1, q=1, walk_length=80, num_walks=10, window=10, title=''):
    from node2vec import Node2Vec
    entry = {}
    node2vec_model = Node2Vec(G, dimensions=dimensions, p=p, q=q, walk_length=walk_length, num_walks=num_walks, workers=4)
    model = node2vec_model.fit(window=window)
    emb=model.wv[[i for i in model.wv.key_to_index]]
    results = cluster_scoring(emb,labels,num_folds=10)
    entry["results"] = results
    return entry

print(f"Running node2vec")
result = n2vec(G,labels)
with open(f"node2vec_experiment.txt", "w") as f:
    json.dump(result, f)

num_workers = os.cpu_count()
funcs = [adamic_adar, resource_allocation, levenshtein, tversky]
params = []
for func in funcs:
    param = (G,func,dataset,labels, 10, 80, num_workers, 10, 128, 10)
    params.append(param)

results = Parallel(n_jobs=num_workers)(delayed(getnet)(*param) for param in params)