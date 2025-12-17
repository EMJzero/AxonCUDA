# Setup

Installed CUDA (driverless) from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local.
Did first `wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run`.
Then `sh cuda_<version>_linux.run --silent --toolkit --override --installpath=$HOME/cuda`.

## Requirements

Tested with:
- Nvidia drivers: 550.78
- CUDA Version: 12.4 (A100)
- GCC 13.4.0

# Useful Commands

- `nvcc filename.cu -o filename -run`: compile and run;
- `nsys profile --stats=true ./filename`: profile;

- `make`
- `make run`
- `make clean`

# Algorithmic Complexity Legend

An hypergraph (hgraph) is defined as $H_g = (V, E, w_e)$, where $V = \{0, 1, \dots N - 1\}$ is the set of vertices, or nodes, $E = \{(s, D) | s \in V, D \subseteq V\}$ is the set of hyperedges (hedges), each with a source $s$ and one or more destinations $D$ ($D = \emptyset$) is a degenerate, admissible case), and $w_e : E \rightarrow \mathbb{R}$ is the weight of each hedge.

For convenience, we discuss algorithmic complexity using the following variables w.r.t. a generic hgraph:
- $n$ : number of hgraph nodes, $|V|$
- $e$ : number of hgraph hedges, $|E|$
- $d$ : average cardinality of each hedge, $(\sum_{(s, D) \in E} 1 + |D|) / |E|$
- $h$ : average connections per node, $(\sum_{v \in V} |\{(s, D) \in E | v = s \vee v \in D\}|) / |V| = e \cdot d / n$

# Some Terminology

- *in-isolation*: gain calculated for a move applied by itself
- *in-sequence*: gain for a move calculated assuming all higher-gain moves already applied