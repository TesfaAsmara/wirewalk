# wirewalk

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

# References

[Wire Before You Walk. T. Asmara, D. Bhaskar, I. Adelstein, S.
Krishnaswamy, M. Perlmutter. In proceedings for Asilomar
2023.](http://tesfaasmara.com/wirewalk.pdf)

# Asilomar 2023 Proceedings

To reproduce the results, please see the instructions in the
[Asilomar_2023.py](https://github.com/TesfaAsmara/wirewalk/blob/main/Asilomar_2023.py)
file.

## Install

``` sh
pip install wirewalk
```

### Prerequisites

You will need

- Python3
- [Networkx](https://networkx.org/documentation/stable/install.html)
- [Numpy](https://numpy.org/install/)
- [Gensim](https://pypi.org/project/gensim/)
- [editdistance](https://pypi.org/project/editdistance/)

I highly recommend installing an
[Anaconda](https://www.anaconda.com/distribution/#download-section)
environment. Future versions of WireWalk will be available on PyPI and
conda.

## How to use

``` python
import networkx as nx
from wirewalk.core import WireWalk, jaccard_coefficient, max_flow

# Create a graph
graph = nx.fast_gnp_random_graph(n=10, p=0.5)

# Instantiate a WireWalk object
wireWalk = WireWalk(graph, dimensions = 128, window = 10, walk_length = 80, num_walks = 10, workers = 1)

# Compute transition probabilities using jaccard coefficient transformation, generate walks, and embed nodes
model = wireWalk.fit(jaccard_coefficient)

# **MAX_FLOW and MIN_COST_MAX_FLOW ONLY WORK WITH GIVEN capacity**
# If weight exists, then  
# nx.set_edge_attributes(graph, nx.get_edge_attributes(graph, "weight"), "capacity").
# Otherwise,
nx.set_edge_attributes(graph, 1, "capacity")
model = wireWalk.fit(max_flow)
```
