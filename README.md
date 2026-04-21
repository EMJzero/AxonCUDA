# AxonCUDA

A hypergraph partitioning tool designed to leverage the massive parallelism of a GPU.
Entirely developed in CUDA, every algorithm is parallel in nodes and hyperedge pins, with at most one sequential iteration over incidence sets when visiting neighborhoods.

Present algorithms originally targeted a specific set of partitioning constraints:
- maximum nodes per partition
- maximum distinct inbound hyperedges per partition

However, it has also been generalized to support $k$-way balanced partitioning.<br>
The primary optimization objective is the connectivity (or $\lambda - 1$ metric).

## Requirements

Tested with:
- Nvidia drivers: 550.78
- CUDA Version: 12.4
- GCC 13.4.0
- GPUs:
  - GH200
  - A100-SXM4-40GB
  - RTX 2080ti
  - A6000

## Setup

Ensure you have CUDA version >12.4 and drivers version >550.54 available.
You can set both them up as follows ([source](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)):
```sh
# download cuda
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
# set it up
sh cuda_<version>_linux.run --silent --toolkit --override --installpath=$HOME/cuda
# put a link to cuda in this project's path
ln -s $HOME/cuda/include cuda-include
```

If you intend to use AxonCUDA for $k$-way partitioning, Mt-KaHyPar is required to produce the initial partitioning.
Hence, please setup Mt-KaHyPar following the instructions [here](https://github.com/kahypar/mt-kahypar.git) regarding the Debian package release. In brief:
```sh
# download the package
wget https://github.com/kahypar/mt-kahypar/releases/download/v1.5.3/mtkahypar_1.5.3_amd64.deb
# extract it
dpkg -x mtkahypar_1.5.3_amd64.deb mtkahypar
# add mtkahypar to path
export PATH="$HOME/mtkahypar/usr/bin:$PATH"
```
Then, make sure `mtkahypar` is a valid executable in your environment.

Build this project:
```sh
# optional: override target compute capability (ex. with 'sm_80')
sed -i 's/native/sm_80/' makefile
# straight build
make clean && make
```


## Usage

Refer to the help menu `./hgraph_gpu.exe -h` for detailed usage instructions.

AxonCUDA supports two partitioning modes.
Inbound constrained minimal partitioning, that attempts to construct the fewest partitions under size and inbound set size constraints per-partition.
K-way balanced partitioning, with a fixed number of partitions and a constraint only on partition size.
Both modes optimize for minimum connectivity, but cut-net and SOED also improve as a side-effect (see [metrics](#hypergraph-partitioning-metrics)).

Examples of typical invocations are as follows:
```sh
# partition a SNN with predefined constraints
./hgraph_gpu.exe -r hgraphs/some_snn.snn -c loihi84
# partition a SNN with custom constraints and algorithm-specific options
./hgraph_gpu.exe -r hgraphs/some_snn.snn -m 128 1024 4096 -cnc 4 -rfr 32
# partition an hypergraph under k-way balanced constraints
./hgraph_gpu.exe -r hgraphs/some_hgr.hgr -k 2 0.03
```

Partitioning settings:
- `-r <hgraph>`: path to the hypergraph to partition;
- `-c <name>`: choose a named constraints set, among hard-coded ones, for inbound constrained minimal partitioning;
- `-m <size> <inbound> <p-cnt>`: set partitions size, inbound size, and count constraints for inbound constrained minimal partitioning;
- `-k <k> <ε>`: set partitions count and balance parameter for k-way balanced partitioning;

> Unless `-k` is present, the partitioning mode defaults to inbound constrained minimal partitioning.

Recommended options:
- `-dtc`: speedup loading time by building and deduplicating initial incidence sets directly on the GPU;
- `-ipm`: speedup refinement by reducing the number of initial partitions through greedy merging;

Helpful options:
- `-smh <lvl>`: increasing 'lvl' can prevent going out of memory when there are many coarsening levels or simply many nodes/nets;
- `-om <mult>`: increase if you an assert for full hash tables triggers during initial neighbors construction;

## Procuring Hypergraphs

Our set of benchmark hypergraphs derived from Spiking Neural Networks (SNNs) is available [here (Zenodo)](https://zenodo.org/records/19194881).

Other relevant benchmarks are the ISPD98 suite and SAT14 suite, that we suggest downloading from [here (Zenodo)](https://zenodo.org/records/30176).<br>
Their size is however relatively small to represent a challenge on GPU, hence we provide the `scale_hgr.py` slop-script to multiply their number of nodes and pins by a user-defined amount (e.g. 16x).

AxonCUDA supports hypergraphs both in the `.hgr` and in its own `.snn` and `.axh` binary formats.<br>
File format details can be found in comments near the end of [`hgraph.hpp`](./includes/hgraph.hpp).

Or you could just go under [`hgraphs`](./includes/hgraphs) and run:
```sh
# fetch all hypergraphs (expected space required: 33GB of SNNs + 1GB of ISPD98)
./procure_hgraphs.sh
```

# Some Terminology

## Algorithmic Complexity Legend

An hypergraph (hgraph) is defined as $G = (V, E, \omega)$, where $V = \{0, 1, \dots N - 1\}$ is the set of vertices, or nodes, and $E = \{(s, D) | s \in V, D \subseteq V\}$ is the set of hyperedges (hedges)
Each hyperedge with a set of sources $src(e)$ and destinations $dst(d)$ (with, in general, no cycles), and a weight given by $\omega : E \rightarrow \mathbb{R}$.

A partitioning of $G$ is defined as a function $\rho(V) \rightarrow P$, where $P$ is the set of partitions.

Let $\Omega$ be the maximum number of nodes allowed for a partition, and $\Delta$ its maximum number of distinct inbound hyperedges.
Formally, our constraints imply:
```math
\begin{aligned}
\forall p \in P, \: |p| &\leq \Omega \\
|\textstyle\bigcup_{n \in p} in(n)| &\leq \Delta
\end{aligned}
```

For convenience, we discuss algorithmic complexity using the following variables w.r.t. a generic hgraph:
- $n$ : number of hgraph nodes, $|V|$
- $e$ : number of hgraph hedges, $|E|$
- $d$ : average cardinality of each hedge, $\sum_{e \in E} |e| / |E|$
- $h$ : average connections per node (node degree), $(\sum_{v \in V} |\{e \in E | v \in e\}|) / |V| = e \cdot d / n$
- $p$ : number of achieved partitions, $|P|$

## Hypergraph Partitioning Metrics

Let $G = (N, E, \omega)$ be a hypergraph with $N$ nodes and $E \subseteq \mathcal{P}(N)$ hyperedges each with weight $\omega(e)$, where $\omega : E \rightarrow \mathbb{R}$.
Let $P \subseteq \mathcal{P}(N)$ a partition of $N$.
For a hyperedge $e \in E$, define its **connectivity** as
```math
\lambda(e) = \left|\{\, i \mid e \cap V_i \neq \emptyset \,\}\right|.
```
Then a partition's quality metrics are:

**Cut Size (Cut Hyperedges)**
Counts the number of hyperedges that span more than one block:
```math
\text{CutSize} = \sum_{e \in E} \omega(e) \cdot \mathbf{1}\{\lambda(e) > 1\}.
```

---

**Connectivity (km1)**
Sums the excess connectivity of each hyperedge:
```math
\text{Conn} = \sum_{e \in E} \omega(e) \cdot \bigl(\lambda(e) - 1\bigr).
```

---

**SOED (Sum of External Degrees)**
Weights excess connectivity by hyperedge size:
```math
\text{SOED} = \sum_{e \in E} \omega(e) \cdot |e| \cdot \bigl(\lambda(e) - 1\bigr),
```
<!--where $|e|$ is the number of nodes in hyperedge $e$.-->

## Speaking of Gains

We refer to refinement gains in two ways:
- *in-isolation*: gain calculated for a move applied by itself
- *in-sequence*: gain for a move calculated assuming all higher-gain moves already applied


# Development Status

## Working HyperGraphs

| **hypergraph** | **constraints** | **status** | connectivity | time [ms] | notes |
| --- | --- | --- | --- | --- | --- |
| 8k-model | loihi64 | <code style="color : lime">ok</code> | 5241.233 | 534.197 |  |
| 64k-model | loihi64 | <code style="color : lime">ok</code> | 40403.949 | 3852.321 |  |
| 256k-model | loihi84 | <code style="color : lime">ok</code> | 42202.102 | 25124.571 |  |
| 1M-model | loihi84 | <code style="color : lime">ok</code> | 136837.625 | 232653.438 |  |
| 16M-model | loihi84 | <code style="color : red">ko</code> |  |  | OOM |
| LenNet | loihi64 | <code style="color : lime">ok</code> | 2716.121 | 508.468 |  |
| VGG11 | loihi84 | <code style="color : lime">ok</code> | 51220.008 | 90740.923 |  |
| AlexNet | loihi84 | <code style="color : lime">ok</code> | 15634.667 | 124028.297 |  |
| MobileNet | loihi84 | <code style="color : red">ko</code> | 3335512.000 | 372353.156 | requires `-dtc`, `-smh 12`, `-ipm` |
| Allen V1 | loihi84 | <code style="color : lime">ok</code> | 6220.164 | 42259.551 | suggested `-cnc 16` |
| 16k-rand | loihi64 | <code style="color : lime">ok</code> | 72359.555 | 902.281 |  |
| 64k-rand | loihi64 | <code style="color : lime">ok</code> | 629264.000 | 2923.482 |  |
| 256k-rand | loihi64 | <code style="color : lime">ok</code> | 3747706.750 | 24485.122 |  |

> All results reported under the "default" configuration.
> All "ko"s are out-of-memory instances...

## Suggested Configurations

| **name** | max candidates (`-cnc`) | refinement repeats (`-rfr`) | oversized multiplier (`-om`) |
| --- | --- | --- | --- |
| default | 4 | 16 | 1.2 |
| profile | 16 | 3 | 1.5 |
| quality | 2 | 32 | 2.5 |

# Opsies

<img src="static/tehepero.png" width="250px" padding-left="15px" align="right"/>

Well, this implementation isn't bulletproof, there are a few knobs that require tuning upon targeting very large hypergraphs.
All problems usually manifest as asserts being triggered:
- `idx < extra_path_size + path_size` in `grouping_kernel`:
  - try increasing `PATH_SIZE`, but keep in mind that it should fit in a thread's registers for performance reasons...
  - check how many repeats the kernel is running, e.g. `NOTE: grouping kernel required [...] repeats=42 ...`, the number of repeats is actively shrinking the available path-length, hence try multiplying `PATH_SIZE` by the same amount...
  - if even with `PATH_SIZE = 1024` the problem persist, the likely suspect is an asymmetric neighbors histogram from `candidates_kernel`, run with `-v 3` for more info. That said, bugs aside, the only reasonable cause is an overflow due to `FIXED_POINT_SCALE`, try lowering it...
- `GM hash-set full!` in any `apply_X` or `neighbors` kernel, means oversized segments for deduplication were not large enough, increase `-om <mul>` from the CLI...
- `invalid partitioning returned` in k-way mode after initial Mt-KaHyPar solution means no valid initial partitioning likely existed, try raising `KWAY_INIT_UPPER_THREASHOLD`...
- in case of an unexpected `invalid partitining` under known valid constraints:
  - inspect the total hyperedges weight (printed after the hypergraph is loaded), and if it is close to the `uint32 / FIXED_POINT_SCALE` limit, the cause is likely an overflow after applying `FIXED_POINT_SCALE`, try lowering it...
  - otherwise, make sure the hypergraph has less than $2^{32}$ nodes or hyperedges, as the code is currently hardwired to identify those with 32 bits...
- too much host RAM usage: add the `-dtc` flag if your device has more VRAM than the host has RAM! Also recommended for a good speedup...

> After any modification, remember to recompile (`make`)!

All the mentioned constants can be found either in [`defines.cuh`](./headers/defines.cuh) or in the offending kernel's header file under [`./headers`](./headers/).

# Reference

AxonCUDA is a free software provided under the MIT License.
If you use AxonCUDA in an academic setting please cite the appropriate papers.

```bibtex
@misc{AxonCUDA-prototype-AsHES,
    title={Incidence Constraints in Hypergraph Partitioning on GPU},
    author={Marco Ronzani and Cristina Silvano},
    year={2026},
    eprint={2604.14411},
    archivePrefix={arXiv},
    primaryClass={cs.DC},
    url={https://arxiv.org/abs/2604.14411},
}
```