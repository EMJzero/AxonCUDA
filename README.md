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

A partitioning of $H_g$ is defined as a function $\rho(V) \rightarrow P$, where $P$ is the set of partitions.

For convenience, we discuss algorithmic complexity using the following variables w.r.t. a generic hgraph:
- $n$ : number of hgraph nodes, $|V|$
- $e$ : number of hgraph hedges, $|E|$
- $d$ : average cardinality of each hedge, $(\sum_{(s, D) \in E} 1 + |D|) / |E|$
- $h$ : average connections per node (node degree), $(\sum_{v \in V} |\{(s, D) \in E | v = s \vee v \in D\}|) / |V| = e \cdot d / n$
- $p$ : number of achieved partitions, $|P|$

# Hypergraph Partitioning Metrics

Let $G = (N, E, \omega)$ be a hypergraph with $N$ nodes and $E \subseteq \mathcal{P}(N)$ hyperedges each with weight $\omega(e)$, where $\omega : E \rightarrow \mathbb{R}$.
Let $P \subseteq \mathcal{P}(N)$ a partition of $N$.
For a hyperedge $e \in E$, define its **connectivity** as
$$
\lambda(e) = \left|\{\, i \mid e \cap V_i \neq \emptyset \,\}\right|.
$$
Then a partition's quality metrics are:

**Cut Size (Hyperedge Cut)**  
Counts the number of hyperedges that span more than one block:
$$
\text{CutSize} = \sum_{e \in E} \omega(e) \cdot \mathbf{1}\{\lambda(e) > 1\}.
$$

---

**km1 (Connectivity–1)**  
Sums the excess connectivity of each hyperedge:
$$
\text{km1} = \sum_{e \in E} \omega(e) \cdot \bigl(\lambda(e) - 1\bigr).
$$

---

**SOED (Sum of External Degrees)**  
Weights excess connectivity by hyperedge size:
$$
\text{SOED} = \sum_{e \in E} \omega(e) \cdot |e| \cdot \bigl(\lambda(e) - 1\bigr),
$$
where $|e|$ is the number of nodes in hyperedge $e$.


# Some Terminology

- *in-isolation*: gain calculated for a move applied by itself
- *in-sequence*: gain for a move calculated assuming all higher-gain moves already applied

# Working HyperGraphs

| **hypergraph** | **hardware** | **status** | score [cut cost] | time [ms] | notes |
| --- | --- | --- | --- | --- | --- |
| 8k-model | loihi64 | <code style="color : lime">ok</code> | 10123.654 | 376.632 |  |
| 64k-model | loihi64 | <code style="color : lime">ok</code> | 45651.293 | 2604.534 |  |
| 256k-model | loihi84 | <code style="color : lime">ok</code> | 53971.926 | 24197.279 |  |
| 1M-model | loihi84 | <code style="color : lime">ok</code> | 164824.750 | 358181.033 | as of commit 99500e3 |
| 16M-model | loihi84 | <code style="color : red">ko</code> |  |  |  |
| LenNet | loihi64 | <code style="color : lime">ok</code> | 3631.116 | 878.774 |  |
| VGG11 | loihi84 | <code style="color : lime">ok</code> | 71695.305 | 302039.327 |  |
| AlexNet | loihi84 | <code style="color : lime">ok</code> | 20610.857 | 344111.681 |  |
| MobileNet | loihi84 | <code style="color : red">ko</code> |  |  |  |
| Allen V1 | loihi84 | <code style="color : red">ko</code> |  |  |  |
| 16k-rand | loihi64 | <code style="color : lime">ok</code> | 90674.516 | 530.093 |  |
| 64k-rand | loihi64 | <code style="color : lime">ok</code> | 709580.000 | 2246.531 |  |
| 256k-rand | loihi64 | <code style="color : lime">ok</code> | 4108240.750 | 16451.505 |  |

> All "ko"s are out-of-memory instances...