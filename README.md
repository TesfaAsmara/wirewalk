# WireWalk
Python3 Implementation of the WireWalk Algorithm by [Tesfa Asmara, et. al.](tesfaasmara.com). Wire Before You Walk. T. Asmara, D. Bhaskar, I. Adelstein, S. Krishnaswamy, M. Perlmutter. In proceedings for Asilomar 2023.
# Asilomar 2023 Proceedings

To replicate the results, please see the [Asilomar_2023.ipynb](https://github.com/TesfaAsmara/wirewalk/blob/main/Asilomar_2023.ipynb) file.

## Install
Currently, WireWalk can only be installed from source.

### Prerequisites

You will need

- Python3
- Networkx
- Numpy
- Gensim
- editdistance

I highly recommend installing an
[Anaconda](https://www.anaconda.com/distribution/#download-section)
environment. Future versions of WireWalk will be available on PyPI and
conda.
## How to use

``` python
import networkx as nx
from wirewalk.core import WireWalk
from wirewalk.functions import jaccard_coefficient

# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)

# Instantiate a WireWalk object
wireWalk = WireWalk(graph, dimensions = 128, window = 10, walk_length = 80, num_walks = 10, workers = 1)

# Precompute probabilities using preferential attachment, generate walks, and embed nodes
model = wireWalk.fit(jaccard_coefficient)
```
