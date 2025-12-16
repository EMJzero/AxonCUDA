New partitioning idea partly based on what Cordone said to me:
=> constructing and deconstructing, what if returning to a valid state is easier than always keeping one?

Idea:
- simple initial partitioning, like sequential
- every node proposes the partition it would like to be in: it goes over all hedges and finds the partition with the highest total weight of hedges times the number of its neighbors in that partition, thus maximizing overlap (very similar to FM), the move must be valid at least in isolation
- sync
- every node applies the move it proposed
- sync
- for every constraints-violating partition (1st parallelism level) go over all its nodes (2nd) and for each find the best partition it could move to with a valid move (in isolation)
- for every constraints-violating partition (1st parallelism level only), sort the moves proposed by lowest cost-maximum constraints gain-first, and apply them one by one after checking if they are still feasible and needed (TODO: this could be improved…)
- repeat from 2 steps back until all partitions reach a valid state, then repeat from step 2

---

Refinement improvement idea:

Instead of assuming all previous moves applied in a large sequence, consider many small non-interfering sequences! Find the highest gain moves with no partitions in common, and continue building them up first with high gain moves not in interfering with others (no neighbors moves by others), and finish with a common trail of the common moves!

---

New selection of refinement moves chain:
- order does not matter, group moves that all together result in a valid, better state
- sets of moves ranked by gain, only the highet gain one is applied -> each set's objective is to become the number 1 by fusing with others, each move starts as an independent set
- sets are sorted by the partition loosing nodes (when fusing sets, a set appears multiple times, once for every partition in it loosing nodes)
- each set goes over all other sets that loose nodes in the partitions it fills (priority to those violating constraints) and tries to find a set it can fuse with while improving gain and getting closer to being valid, or becoming "more valid" (getting further from violations)
- fuse sets, repeat until you can't anymore, pick the highest gain set and apply it

---

Distinct inbound:
If touching sets are sorted, I can use binary search for a log-n check of if an hedge is already there or not, and count the potential new ones.

Fast inbound check in candidates:
While filling the histogram, first check for size, then count how many touching of the neighbor are not in my inbound!

---

Inbound constraint:
- always inferred from touching - group size
- while coarsening, no need for extra data structures: before proposing a pair, in the loops that selects the best K pairs during the candidates kernel, check the size of the de-duplicated inbound set, add the candidate only if it passes
- at the return from the innermost grouping = partitioning, build the coarsest touching sets and store them in their own global “partitons_touching” variable
- while uncoarsening, update this “partitions_touching” after every actual move
- when finding the best possible partition for refinement, compute which size would the new partitions_touching set of the target partition have, and propose the move only if ok for the constraints
- when doing the second pass over proposed moves, applying all the ones before you, already compute the final touching set for your target partition even wrt the previous moves (dedupe), and replace the move with -1 and 0 score if you are not ok with constraints (then filter out -1s during the scan for the best subsequence)

Idea for inbound constraints:
- the pins per partition array can be used to infer the size of each partition’s touching set: a partition has exactly as many inbounds as the number of times it appears in pins per partition with a >0 count (minus its size, as always, as each node has one outbound hedge);
- once you do the “consider all previous moves applied” re-calculation of gains, you iterate each neighboring hedge’s nodes, thus if you could extend that to see “all nodes of all hedges touching my target partition” you could compute on the fly the count of pins per hedge of each of those hedges and see the new inbounds count for the partition you want to enter, and see if you fit when all previous moves are applied!
- it should also be easy to integrate in the above the check for the size of the partition!

TL;DR:
The pins per partition matrix, transposed, is the hedges per partition, with count of how many nodes of the hedge are in the partition!

Options:
- for each partition, a scan over moves, of the array (one entry per hedge, one row of the pins per partition matrix) of pins per that partition -> each step of the scan costs O(e)…
- for each partition and hedge, a scan over moves, marking only the cases where it remains zero, then reduce among hedges for each partition, you still need O(e*p) scans…

Important:
Already do the constraints checks in isolation, propose only the best moves that already satisfy the constraints by themselves!

Possible way to work with all the scans over the 3D data of partitions x moves x hedges (or nodes): store it compressed (sparse CSR) along one dimension, possible hedges! The issue is how to do the scans quickly…