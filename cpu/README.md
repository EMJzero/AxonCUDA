# AxonCUDA - CPU (OpenMP)

A multithreaded CPU implementation of the very same partitioning algorithm as the CUDA version one level up.
It exists to measure the GPU's speedup on identical work: same steps, same parallelism, same outputs.

- every CUDA kernel becomes a function (same name, under [`kernels`](./kernels)) running an OpenMP loop over the same entities (nodes, hedges, groups, moves, events);
- every thrust/CUB primitive becomes a parallel CPU primitive ([`prims.hpp`](./headers/prims.hpp)): scans, stable radix sorts, compactions, reductions;
- warp- and lane-level parallelism inside an entity becomes a sequential inner loop, except where the order of a float reduction across lanes matters, then the lanes and their shuffle tree are emulated;
- host steps under [`sources`](./sources) mirror those of the CUDA version one-to-one, minus all VRAM management (chunking, spilling, oversized buffers, neighborhood size sampling).

Shared with the CUDA version: `hgraph.hpp`, `constr.hpp`/`constr.cpp`, and the program-wide constants in `headers/defines.cuh`.
Kernel-specific constants are copied in the CPU headers (marked "must match"), keep them in sync.

## Build and Usage

```sh
make clean && make
./hgraph_cpu.exe -r <hgraph> -c loihi64 -t <threads>
```

Same CLI as the CUDA version, with these differences:
- `-t <num>`: number of OpenMP threads (default: `OMP_NUM_THREADS` or all hardware threads);
- `-ppp <auto|dense|sparse>`: pins per partition representation during refinement, `auto` picks the dense matrix iff it fits in `DENSE_PPP_RAM_FRACTION` of the available RAM;
- `-ptc` (alias `-dtc`): build touching sets in parallel, rather than sequentially while preparing the hypergraph;
- `-xdp`: exact maximum weight matching during grouping (see below), NOT what the CUDA version computes;
- `-om`, `-smh`: accepted and ignored (device memory knobs).

For timing, use an otherwise idle machine and pin threads, e.g. `OMP_PROC_BIND=close OMP_PLACES=cores`.
OpenMP barriers degrade catastrophically when threads get descheduled: on a machine shared with other jobs, use fewer threads than free cores.

## Reproducibility w.r.t. the CUDA Version

The CPU version is deterministic, its output does not depend on the number of threads.

Its partitions are **bit-identical** to the CUDA version's on the incidence-constrained runs tested so far: ISPD98 ibm01-18 (`-c loihi64`, ibm01-06 also with `-np`), ISPD98 ibm18 16x (3.4M nodes, `-np`), plus `-ipm`, `-dtc`/`-ptc`, and dense vs sparse pins per partition.
To check it on a new instance, run both binaries with `-p <file>` and compare the files (the CUDA version is itself deterministic; remember its `-om` may need raising).
Divergence can only originate from:
- float reductions performed by thrust/CUB, whose association cannot be reproduced: the in-sequence gains scan (refinement), the chains' weights reduction (chaining), the total hedge score (k-way init);
- the order of coarse outbound touching sets (hash-slot order on the GPU, first-seen order here), that changes the order of float sums over touching hedges.
Integer computations (coarsening scores, events, constraint checks) and the float sums whose order is reproducible (per-lane accumulations and shuffle trees, the cascade's sequential sums) match exactly.

K-way mode additionally depends on Mt-KaHyPar, which is not deterministic: the coarse hypergraph handed to it matches the CUDA version's pin for pin, with hedge weights off by at most a couple of fixed-point units (due to the total hedge score reduction).

## Where the CPU Version Deliberately Differs

- **Grouping**: the CUDA kernel walks every node's path up its candidate tree with atomics; here the same dynamic programming runs level by level (leaves to roots, then back), each decision taken once with final inputs.
  The CUDA walks effectively give each node a gain equal to its score (each node's own walk claims its parent with `(with, wout) = (score, 0)`, which dominates, via the atomic max, the claims of walks coming from its subtree, where the sums of children's wout cancel out); the default reproduces this exactly.
  `-xdp` instead gives each node the gain of its whole subtree, `gain(v) = score(v) - max(0, best child gain)`, yielding a maximum weight matching of each tree.
- **Touching sets / neighbors / coarse hedges**: deduplicated via per-thread arrays stamped with the current item id, rather than shared+global memory hash-sets; coarse touching sets are counted from the nodes of each group rather than from the coarse hedges (same sets).
- **Candidates**: one histogram per node mapped by a per-thread node->bin array, rather than a sorted shared memory histogram processed HIST_SIZE neighbors at a time (same integer scores, same candidates).
- **Refinement events**: generated directly in rank order and compacted to valid moves, so that a stable sort by partition (and hedge) replaces the sort by (partition, (hedge,) rank); invalid moves are flagged as the dense CUDA path does.
- **Per-partition counters**: accumulated per thread and flushed once, rather than updated with atomics from every hedge.
