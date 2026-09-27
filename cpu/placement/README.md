# AxonCUDA - Placement - CPU (OpenMP)

A multithreaded CPU implementation of the very same placement algorithm as the CUDA version in [`placement`](../../placement).
Like the [CPU partitioner](../README.md), it exists to measure the GPU's speedup on identical work: same steps, same parallelism, same outputs.

- every CUDA kernel becomes a function (same name, under [`kernels`](./kernels)) running an OpenMP loop over the same batch-flat entities (nodes, hedges, events, multi-starts);
- every thrust/CUB call becomes a parallel CPU primitive, shared with the CPU partitioner ([`prims.hpp`](../headers/prims.hpp));
- warp-level parallelism inside an entity becomes a sequential inner loop, replaying lanes and shuffle trees wherever float order matters;
- host steps under [`sources`](./sources) mirror those of the CUDA version one-to-one, minus all device memory management.

Shared with the CUDA version: `topology.hpp`, `curves.hpp`, `nmhardware.hpp`/`nmhardware.cpp` (placement), `hgraph.hpp` (root).
Constants in [`defines_plc.hpp`](./headers/defines_plc.hpp) are copied from the CUDA headers, keep them in sync.

## Build and Usage

```sh
make clean && make
./hplace_cpu.exe -r <hgraph> -c loihi64 -thr <threads>
```

Same CLI as the CUDA version, with these differences:
- `-thr <num>`: number of OpenMP threads (default: `OMP_NUM_THREADS` or all hardware threads);
- `-ptc` (alias `-dtc`): build touching sets in parallel, rather than sequentially while preparing the hypergraph;
- `-mso` defaults to one multi-start per thread, rather than to what fills the GPU: pass `-mso` (and `-bs`) explicitly to compare the two.

[`hgraphs/run_plc.sh`](./hgraphs/run_plc.sh) runs the same experiments as its CUDA counterpart, reading the partitioned hypergraphs from `placement/hgraphs/part_snns` and storing results here; `THREADS=<num>` sets the thread count, `-p` adds per-phase times.

## Reproducibility w.r.t. the CUDA Version

The CPU version is deterministic, its output does not depend on the number of threads.

Its placements are **bit-identical** to the CUDA version's on all `placement/hgraphs/part_snns` inputs, with the settings of `run_plc.sh`, and further with other seeds, multi-start and batch counts, host-side touching sets, the `snak` and `zord` curves, `-cnc 2`, `-ff`, and the `tor6d` and `arb` topologies.
To check it on a new instance, run both binaries with `-s <file>` and compare the files.

What is reproduced exactly:
- the random keys of the recursive bisection: cuRAND's XORWOW host generator (one per multi-start) is copied, seeding, subsequence jumps and output order included;
- every per-node float sum over touching hedges: lanes, their shuffle tree, and the multiply-adds nvcc fuses (as explicit `fmaf`);
- the scan picking each multi-start's improving swap prefix: CUB's raking block scan is replayed step by step.

Divergence can still originate from float reductions performed by thrust/CUB, whose association is not replayed: the bisection split costs (`reduce_by_key`), the in-sequence label propagation gains (`inclusive_scan_by_key`), the subtree connection strengths and the per-attempt grades (segmented reduces).
They only matter at near-ties (a split cost improving, a subtree reversal, the winning attempt), never observed so far; the printed `whops` of each attempt differ in the last digits.

## Where the CPU Version Deliberately Differs

- **Exclusive swaps**: the CUDA kernel lets every node walk up its tree of pairs with atomics (a cooperative launch, chunked by multi-starts to fit its paths); here the same claims and pairings run level by level, as in the CPU partitioner's grouping, with the same outcome.
- **Minimum spanning trees** (grading): per-thread scratch rather than per-lane registers, so hedges are not capped at 2048 pins.
