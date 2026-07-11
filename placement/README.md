# AxonCUDA - Placement

A hypergraph placement tool designed to leverage massive parallelism and multi-start exploration on a GPU.
This is a spin-off of AxonCUDA, which uses many of the same kernel ideas and infrastructure to handle hypergraphs.
Its complexity is again that of a neighborhood iteration, running sequentially over incidence sets.

This placement tool and the parent partitioning one are meant to be used in pipeline, first partitioning nodes, then placing each partition over a lattice representing a system over which to distribute the original workload (e.g. SNN).

The algorithms assign nodes from a hypergraph to points on a 2D lattice so as to minimize the resulting hops of hyperedges between nodes over the lattice.
Two algorithms run back to back:
- initial placement, recursive bisection and binary tree folding form a 1D sequence of locally-connected nodes, then projected to 2D via the Hilbert space-filling curve;
- force-directed placement refinement, swapping pairs of neighboring nodes to reduce the total distance covered by hyperedges, converges to a local optima;
With multi-start, both steps repeat several times, possibly concurrently on the same GPU if occupancy allows it.

Pending work:
- investigation of different space-filling curves for the 1D to target topology initial placement projection
- generalization beyond the 2D lattice, assigning hypergraphs towards any graph-backed topology.

## Setup

Ensure CUDA and matching drivers are available, as per the [main README](../README.md#setup).

Build this project:
```sh
# enter the 'placement' directory
cd placement
# optional: override target compute capability (ex. with 'sm_80')
sed -i 's/native/sm_80/' makefile
# straight build
make clean && make
```

## Usage

Refer to the help menu `./hplace_gpu.exe -h` for detailed usage instructions.

Examples of typical invocations are as follows:
```sh
# place a pre-partitioned SNN with predefined constraints
./hplace_gpu.exe -r hgraphs/some_part_snn.snn -c loihi64
# place a pre-partitioned SNN with algorithm-specific options
./hplace_gpu.exe -r hgraphs/some_part_snn.snn -c loihi84 -lpr 16 -fdi 1024 -mso 64 -thr 4
```

Placement settings:
- `-r <hgraph>`: path to the hypergraph to place;
- `-c <name>`: choose a named constraints set, among hard-coded ones, for the lattice size and hop costs;

Recommended options:
- `-dtc`: speed up loading time by building and deduplicating initial incidence sets directly on the GPU;
- `-seed <num>`: set a seed for the multi-start routine;

Helpful options:
- `-sfc <curve>`: determines the space-filling curve to use for the 1D->2D projection during initial placement;
- `-ff`: skips initial placement in favor of a sequential host-side greedy ordering of nodes with "some" locality;

## Procuring Hypergraphs

Make sure you followed the instructions to procure the original pre-partitioning hypergraphs [here](../README.md#procuring-hypergraphs).
Intended benchmarks for placement are the results of partitioning such hypergraphs.

To prepare the input hypergraphs, go under [`hgraphs`](./hgraphs) and run:
```sh
# prepare all partitioned hypergraphs (expected space required: 42MB)
./build_hgraphs.sh
```

# Placement Quality Metrics

<!--Let the input hypergraph be $G = (N, E, \eta)$, with $N$ nodes, $E$ edges, $src(e)$, $dst(e)$ denoting the source (hp: one only) and destinations within each $e \in E$, and $\eta : E \rightarrow \mathbb{R}$ being the weight of each hyperedge.-->
Let the input hypergraph be $G = (N, E, \eta)$, with $N$ nodes, $E$ edges, $src(e)$ and $dst(e)$ denoting the sources and destinations within each $e \in E$, and $\eta : E \rightarrow \mathbb{R}$ being the weight of each hyperedge.
Let the target lattice be $H = \{(h_x, h_y) \in \mathbb{N}^2 : 0 \leq h_x < width, 0 \leq h_y < height\}$ and $\gamma : N \rightarrow H$ be the placement function assigning nodes to lattice points.

The costs we consider here are defined as:
<!--```math
\begin{align*}
    &\text{Energy} = \sum_{e \in E} \: \sum_{d \in dst(e)} \eta(e) (\lVert \gamma(src(e)) - \gamma(d) \rVert (E_R + E_T) + E_R) \\
    &\text{Avg. Latency} = \frac{1}{\sum_{e \in E} \eta(e) \cdot |dst(e)|} \cdot \sum_{e \in E} \: \sum_{d \in dst(e)} \eta(e) (\lVert \gamma(src(e)) - \gamma(d) \rVert(L_R + L_T) + L_R) \\
    &\text{Avg. congestion} = \frac{1}{|H|} \sum_{e \in E} \: \sum_{d \in dst(e)} \: \eta(e) \sum_{h \in Rect(\gamma(src(e)), \gamma(d))} \tau(h, \gamma(src(e)), \gamma(d))
\end{align*}
```-->
```math
\begin{align*}
    &\text{Energy} = \sum_{e \in E} \: \sum_{s \in src(e)} \sum_{d \in dst(e)} \eta(e) (\lVert \gamma(s) - \gamma(d) \rVert (E_R + E_T) + E_R) \\
    &\text{Avg. Latency} = \frac{1}{\sum_{e \in E} \eta(e) \cdot |src(e)| \cdot |dst(e)|} \cdot \sum_{e \in E} \: \sum_{s \in src(e)} \: \sum_{d \in dst(e)} \eta(e) (\lVert \gamma(s) - \gamma(d) \rVert(L_R + L_T) + L_R) \\
    &\text{Max. Latency} = \max_{e \in E} \: \max_{s \in src(e)} \: \max_{d \in dst(e)} (\lVert \gamma(s) - \gamma(d) \rVert(L_R + L_T) + L_R) \\
    &\text{Avg. congestion} = \frac{1}{|H|} \sum_{e \in E} \: \sum_{s \in src(e)} \: \sum_{d \in dst(e)} \: \eta(e) \sum_{h \in Rect(\gamma(s), \gamma(d))} \tau(h, \gamma(s), \gamma(d)) \\
    &\text{Max. congestion} = \max_{h \in H} \sum_{e \in E} \: \sum_{s \in src(e)} \: \sum_{d \in dst(e)} \eta(e) \cdot \tau(h, \gamma(s), \gamma(d))
\end{align*}
```

Where $E_R$, $E_T$, $L_R$, $L_T$ are the energy and latency, respectively, for routing and transmitting information (e.g. spikes) between lattice points when these model a distributed system.
Meanwhile, $\tau(h, h_s, h_d)$ is the probability of a spike being routed through core $h$ when going from core $h_s$ to $h_d$ (see [this function](./sources/nmhardware.cpp#L322) for details), $Rect(h_1, h_2)$ is the set of lattice points contained in the closed coordinate rectangle defined by the opposite corners $h_1$ and $h_2$, and $\lVert \cdot \rVert$ indicates the Manhattan distance on the lattice.


# Reference

AxonCUDA - Placement is free software provided under the MIT License.
If you use AxonCUDA in an academic setting, please cite the appropriate papers.

```bibtex
coming soon...
```
