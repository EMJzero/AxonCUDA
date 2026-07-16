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

Let the input hypergraph be $G = (N, E, \eta)$, with $N$ nodes, $E$ edges, $src(e)$ and $dst(e)$ denoting the sources and destinations within each $e \in E$, and $\eta : E \rightarrow \mathbb{R}$ being the weight of each hyperedge.
Let the target lattice be $H = \{(h_x, h_y) \in \mathbb{N}^2 : 0 \leq h_x < width, 0 \leq h_y < height\}$ and $\gamma : N \rightarrow H$ be the placement function assigning nodes to lattice points.

The costs we consider here are presented through two communication models between lattice nodes (aka "compute nodes").
> Refer to options `-noum` and `-nomm` to disable metrics evaluation for the unicast and multicast models respectively. The multicast model in particular requires solving several minimum Steiner trees and could require several hours to complete.

In both cases, $E_R$, $E_T$, $L_R$, $L_T$ represent the energy and latency, respectively, for routing and transmitting information (e.g. spikes) between lattice points when these model a distributed system.

### Unicast Model Metrics

In the unicast model, every source is assumed to dispatch a unique message towards every destination, and they all propagate independenlty.
This can be interpreted as an **upper bound** on the traffic volume and congestion.

```math
\begin{align*}
    &\text{Energy} = \sum_{e \in E} \: \sum_{s \in src(e)} \sum_{d \in dst(e)} \eta(e) \cdot \left( dist(\gamma(s), \gamma(d)) \cdot (E_R + E_T) + E_R \right) \\
    &\text{Avg. Latency} = \frac{1}{\sum_{e \in E} \eta(e) \cdot |src(e)| \cdot |dst(e)|} \cdot \sum_{e \in E} \: \sum_{s \in src(e)} \: \sum_{d \in dst(e)} \eta(e) \cdot \left( dist(\gamma(s), \gamma(d)) \cdot (L_R + L_T) + L_R \right) \\
    &\text{Max. Latency} = \max_{e \in E} \: \max_{s \in src(e)} \: \max_{d \in dst(e)} \left( dist(\gamma(s), \gamma(d)) \cdot (L_R + L_T) + L_R \right) \\
    &\text{Avg. congestion} = \frac{1}{|H|} \sum_{e \in E} \: \sum_{s \in src(e)} \: \sum_{d \in dst(e)} \: \eta(e) \sum_{h \in Rect(\gamma(s), \gamma(d))} \tau(h, \gamma(s), \gamma(d)) \\
    &\text{Max. congestion} = \max_{h \in H} \sum_{e \in E} \: \sum_{s \in src(e)} \: \sum_{d \in dst(e)} \eta(e) \cdot \tau(h, \gamma(s), \gamma(d))
\end{align*}
```

Meanwhile, $\tau(h, h_s, h_d)$ is the probability of a spike being routed through core $h$ when going from core $h_s$ to $h_d$ (see [this function](./sources/nmhardware.cpp#L322) for details), $Rect(h_1, h_2)$ is the set of lattice points contained in the closed coordinate rectangle defined by the opposite corners $h_1$ and $h_2$, and $dist(\cdot, \cdot)$ indicates the Manhattan distance on the lattice.

### Multicast Model Metrics

In the ideal multicast model, every source is assumed to emit a single copy of any message, with it prograssively being multicasted, forking along the way to reach all destinations.
In particular, we assume an ideal multicast fanout tree that thus follows any minimum-span Steiner tree connecting all hyperedge terminals.
This provides us with an optimistic **lower bound** on the volume of traffic and congestion.
It follows that this is better than what an actual multicast implementation typically achieves, but any improvement on this lower bound leaves less actually irreducible traffic for any real system to deal with.

Local costs on every hyperedge $e \in E$ or core $h \in H$ are threefold:
```math
\begin{align*}
    & \text{energy}(e) \;= hops(\gamma(e)) \cdot (E_R + E_T) + E_R \\
    & \text{latency}(e) \;= \!\! \max_{d \in dst(e)} \!\! dist(\gamma(src(e)), \gamma(d)) \cdot (L_R + L_T) + L_R\\
    & \text{congestion}(h) \;= \textstyle\sum_{e \in E} \: \eta(e) \cdot p\text{-}transit(h, \gamma(e))
\end{align*}
```
Where $dist : H \times H \rightarrow \mathbb{N}$ is the shortest path length on $H$, becoming the the Manhattan distance for a 2D lattice; $hops : \mathcal{P}(H) \rightarrow \mathbb{N}$ counts the lattice edges traversed by a hyperedge to reach all destinations from its source; $p\text{-}transit : H \times \mathcal{P}(H) \rightarrow [0, 1]$ is the probability of a message traversing core $h$ while it propagates between an hyperedge's terminals.

By aggregating these local costs, we define a global communication performance model:
```math
\begin{align*}
    & \text{Tot. Energy} \;=\, \textstyle\sum_{e \in E} \: \eta(e) \cdot \text{energy}(e) \\
    & \text{Avg. Latency} \;=\, \frac{1}{\sum_{e \in E}\eta(e)} \cdot \textstyle\sum_{e \in E} \eta(e) \cdot \text{latency}(e) \\
    & \text{Max. Congestion} \;=\, \textstyle\max_{h \in H} \: \text{congestion}(h)
\end{align*}
```

The definition of $hops$ and $p\text{-}transit$ depends on routing policies, and are thereby subject to the optimal multicast Steiner tree model, formally:
```math
\begin{align*}
    & T_H(hs) = \{ t \subseteq H : hs \subseteq t \text{ and } t \text{ spans a tree in } H \} \\
    & T_H^\star(hs) = \arg\min_{t \in T_H(hs)} |t| \\
    & hops(hs) = |t^\star| - 1 \text{\; for any \;} t^\star \!\in T_H^\star(hs) \\
    & p\text{-}transit(h, hs) = \frac{|\{t \in T_H^\star(hs) : h \in t\}|}{|T_H^\star(hs)|}
\end{align*}
```
Where $T_H(hs)$ is the set of core sets that support a tree spanning terminals $hs$ over $H$.
Thus, any $t^\star \!\in T_H^\star(hs)$ encodes a minimum-size Steiner tree containing $|t^\star|-1$ links.
Terminal roles as sources or destinations do not alter the $T_H^\star(hs)$ set.
Trees are represented by their core sets; edge variations that induce identical traversals are disregarded, as congestion is modeled lattice point.

# Reference

AxonCUDA - Placement is free software provided under the MIT License.
If you use AxonCUDA in an academic setting, please cite the appropriate papers.

```bibtex
coming soon...
```
