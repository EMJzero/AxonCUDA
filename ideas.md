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

---

Soluzione inbound constraints check per refinement:
- traccia il numero di volte che ogni hedge è touching su ogni partition (costruiscilo una volta a fine coarsening e tienitelo aggiornato, come con i partition sizes)
- tecnica degli eventi, genera per ogni move due eventi con le inbound hedge che rimuovi (count -1) e aggiungi (count +1) for every partition -> ogni evento contiene solo un vettore di delta dei counter per ogni hedge inbound alla partizione
=> issue, tenere tutto il vettore richiederebbe tenere un int32 per hedge per evento!
  => ogni evento tiene solo la lista della hedge i cui counter va ad alterare!
  => tieni gli eventi compressi con la tecnica dei due passi! Prima esecuzione del kernel per calcolare lo spazio da allocare per gli eventi e scan per gli offset (degli eventi) a cui ogni thread scrive, secondo pass per scrivere il contenuto di ogni evento (hedge counters da incrementare/decrementare)
  => per ogni evento salva giusto una volta il verso del counter come -1 o +1 per tutte le hedge listate (ogni evento avrà quindi <lista di hedge affette> <partizione i cui counter sono da updatare> <direzione>)
- ora puoi usare, esattamente come per i size, un “event flags” kernel che confronta ogni stato as of un evento con il precedente e incrementa di 1 il contatore di partizioni invalide, finendo con una scan per trovare le mosse con 0 invalide
=> prima del kernel sulle event flags, per avere il totale dei counter per partizione, sorta gli eventi per partizione, e scrivi una custom scan by key che dentro ogni partizione accumula, in ordine di move rank / score gli update per ogni hedge counter! Questo kernel scorre, dentro gli eventi di ogni partizioni, tutti i counter delle hedges, e fa una scan per ognuno in ordine di rank!
- concludi con l’event flags kernel che conta le partizioni invalide e le isola facendo una scan

Little issue: tenere traccia dei contatori (uno per hedge) per tutte le partition!
Soluzione: tieni, per ogni partition, solo un counter per ogni hedge nel suo touching set, ed ogni volta che li aggiorni usa il doppio kernel, il primo per calcolare il size ed il secondo per ri-allocare tutti i contatori nuovi (che possono essere di più!) -> update out of place, invece che in place come per i sizes! Ma almeno risparmi spazio!

Tech for the scans: one warp per partitions (handling the event for that partition), doing the scans on shots of 32? Better: one block per partition and CUB scan at the block level!

—

Summary:
- build touching set per hedge at the coarsest level
- build events per move with the +/-1 per hedge for the event’s partition
- sort events by partition
- scan inside each partition for each hedge (both existing and new ones, first a pass to create the slots for all the same hedges in all events, then the scans per hedge)
- flag events kernel followed by a scan to count the number of invalid partitions as of every move 
- re-build with the two-step kernel the touching set per hedge


---

Idea:
As the hypergraph gets coarser, switch many kernels from one node/hedge per thread to one node/hedge per BLOCK, this way all the local memory buffers become shared memory buffers!

---

Clean inbound constraints check during refinement:
- build and track, for every partition, the number of times each hedge is inbound to it (this needs to be continuously updated while you uncoarsen -> built it once at the innermost level and update it, just like the hedge sizes), use a struct for each entry {hedgeid, count};
- for every move, generate two events for every inbound hedge, one removing it front the src, one adding it back;
- sort also those events by partition;
- do a custom prefix sum for each partition (“by key”) (one partition per block), where you do the prefix sum along each hedge that was already inbound to the partition. At the same time, count how many new hedges are added by events => keep the count of each new seen hedges via a shared memory hash-table
- find the validity of each partition as you go throught events, track the number of inbound hedges that went to 0 (count), the partition becomes invalid the moment that the number of new (distinct) inbounds in the hash-table —minus— the number of inbounds that went to 0 surpasses the constraint. When you become invalid, add +1 to the move’s validity, when you regain validity, write -1;
- find the number of valid partitions as of each move via a scan of validity;
- apply moves and do a costly (two-step, first size then write) update of the tracking of inbound counts per hedge per partition (because there could be more or less hedges than before…);

Maybe:
To rebuild the inbounds counts after you refine, maybe there is no need to re-count how many distinct inbound there are per partitions with the two-steps method, rather you can keep the count from when you compute the validity, that depends on the distinct inbounds count!

Alternative (likely slower):
Do not keep in global memory the inbound counts, re-build them in SM before every check during refinement.
This requires doing the “custom scan” from before with a grid.synch kernel, first you pay n*d to put all the inbound counts of each partition into a SM hash-table, then do as above, but now just track the size of the hash-table (from where you remove bins that go to 0) to determine whether you are valid or not, as of any event.
=> can use SM because it’s one partition per block!

Options for the scan inside each partition’s events:
- one partition per block, one warp per inbound hedge. Do the scan inside the warp. Before doing the scans, move the partition’s while inbound set (with counters) in shared memory, this way you know the amount of work to allocate to each warp
- one partition per block, one warp per event, do all the scans in parallel between warps

Likely go with option 1!
It uses the warp to go in parallel over events (nodes) and warps of a block to go in parallel over the average partition inbound cardinality!

||
||
VV

Necessary variation: must split events by partition!

Clean inbound constraints check during refinement:
- build and track, for every partition, the number of times each hedge is inbound to it (this needs to be continuously updated while you uncoarsen -> built it once at the innermost level and update it, just like the hedge sizes), use a struct for each entry {hedgeid, count};
  - build these from touching, deduplicating the touching sets of all nodes that ended up in the same group
  - touching sets are sorted => we can use a "pop from head" style as the merge in mergesort!
  - groups are at most MAX_GROUP_SIZE+1 in side! We know exactly how many workers we need per partition!
- for every move, generate two events for every inbound hedge, one removing it front the src, one adding it back;
=> each event composed of the partition it affects, the hedge involved, the originating move's rank (index), and a delta (+1 / -1 -> int8)
=> let only inbound hedges generate an event!!
- sort those events by rank;
- stable-sort those events by hedge;
- stable-sort those events by partition;
=> the resulting array will have events sorted by partition, and inside each partition sorted by hedge, and inside each hedge sorted by rank!
=> faster, sort once by the tuple (partition, hedge, rank) in lexicographical order!
- inclusive scan by key of the deltas, the key being (partition, hedge) -> we now have the total number of times each hedge appears in the inbound set as of each move (in order of rank);
=> next objective, count, as of each move (rank) how many inbound hedges has the partition (how many counters are > 0)
- create a new array of events, one event for each time the counter of an hedge in the inbound set (+ the overall inbounds per partition counter) goes from 0 to >0, the event carrying a +1 to the inbound set size, one event for each time the counter of an hedge goes from >0 to 0 carrying a -1 to the inbound set size for that partition;
=> done with one thread per event (-> the original events, one per move per hedge of the moved node), deciding just based on itself and the event before itself (just like the flags for the size events)!
=> each new event shall again carry along, like the size events, the partition it affects, its rank (index), and a delta (+1 / -1 -> int8)!
- sort the new events by the (partition, rank) tuple;
- inclusive scan by key of the deltas, the key being 'partition', this way each event represents the size of the inbound set as of the move given by its rank;
=> now we close-off exactly like for size constraints!
- one thread per new event, init. a counter for each move to 0, add +1 atomically to the counter of an event's rank for each event that makes a partition invalid (w.r.t. to the event before himself), if it makes one valid add -1 instead;
=> these counters track how many partitions are made invalid or valid as of each move
- find the valid partitions as of each move via a scan of these validity counters, those being zero after the scan represent [ranks of] valid moves;
- apply moves and do a costly (two-step, first size then write) update of the tracking of inbound counts per hedge per partition (because there could be more or less hedges than before...);

How to call these events:
- inbound hedge events
- inbound size events

Note: since touching sets are sorted, keep also the inbounds per partition sorted, and use a binary search to acquire the counter for an inbound hedge quickly!


---

WAAAAAIT A MINUTE!
Isn't the inbound counters per partition just pins per partition transposed?

pins per partition:
- for each hedge
- for each partition
- number of times a pin (node) of the hedge is in the partition

partition inbound counts:
- for each partition
- for each (inbound) hedge
- count of the hedge's occurrencies (= # of the hedge's pins/nodes in the partition)

Answer: yes, kinda! partition inbound counts is stored in a sparse form (with offsets and all), but semantically they are mirrors!

=> I can use pins-per-partition instead of computing inbound counters!
=> As extra, I just need to compute, while doing pins-per-partition, the distinct inbound count per partition
  -> can be done with a later count of the non-zero pin counters by key (= partition)!

---

BUG FIXING:

Questions:
- is the new logic for ordering touching hedges correct with the subtraction at the index?
- am I looking only at inbound hedges when creating events for each move?
- so long as you are not in your neighbors, it should not be a problem to be both the source and destination of an hedge, check if neighbors are coarsened correctly when the source is also a destination (self-cycle)!
- refinement should not care about seeing self-cycles in hedges, since it should skip the current node the hedges it visits anyway. Different story if there are duplicates in touching tho!
=> touching must prioritize inbounds, and remove duplicates from outbounds!
- check that inbound_counts is passed correctly as a recursive argument (check the arguments position!!)

TODO:
- must reshape the loops for the touching scatter! First handle (and dedupe) all inbounds of all nodes, then all outbounds!

Final solution:
- no funny business with hedges, continue deduplicating destinations wrt the source (everything as it was before)
- create another kernel to build “inbound pins per partition” by iterating over node hedges (e*d) but touching (n*h, same cost), since in touching you will use the new logic (see above) to separate inbounds from the rest
=> call it “inbound_pins_per_partition_kernel”!
=> kernel: one thread per node!

=> ultimate solution: track the separation between inbounds and outbounds in the “touching” sets only! Where you remove duplicates from “outbounds” first (the last survivor shall be an inbound)!

TODO:
- allocate once and for all pins per partition
- compute pins per partition once per level
- transform pins per partition, after refinement, by have one thread per node go fetch the node’s outbound set and substract it from the counters

Wait, but for how they are constructed, the src of hedges isn’t always the same anyway? Even if without self-cycles?
=> if so:
- verify it by moving to the host the hedges at every round and checking that srcs correspond
- write the kernel that updates counters just as a map, one thread per hedge, fetched the src and goes decrement its counter in pins per hedge
- the issue is then how to compute the inbound set sizes when src and dsts mix, and the best option if either a count-reduce per partition over pins per partition (issue: strides accesses unless they are done histogram vector by vector)

Maybe it would be faster to build pins per partition with touching, by going one block per partition, 256 threads digesting touching hedge with an hash-map in shared memory, then dumped to global with one streak of atomics?

---

Optimizing the candidates kernel...

IDEA:
- every lane picks up HIST_SIZE/WARP_SIZE neighbors and filles HIST_SIZE/WARP_SIZE
- reduce to count valid neighbors (or no? Just leave blanks there?)
- warp shuffles to make all histograms identical
=> or maybe keep using the warps like now, for neighbor inbounds?

Plan:
- shared memory bin ids (to then sort)
- local score accumulators only (one per bin)

- each thread in a warp reads a neighbor (coalesced accesses) (use an if to see if lane_id exceeds neighbors count, then don't read)
- then, in a loop, one thread at a time (up to the number that did the read = remaining number of neighbors):
  - checks the neighbor's validity, if not valid, continue
  - if valid, broadcasts the neighbor with a shuffle to other threads in the warp
  - threads proceed to check the validity w.r.t. the inbound set intersection in parallel
  - if valid, the original owner writes the entry to the histogram and decrements the remaining bins count

- sort shared memory id bins
- init local score bins to 0

- lane 0 reads hedge offset and weight, broadcasting it with a shuffle
- binary search for the histogram bin


Faster candidates kernel:
IDEA: postpone inbound size checks after computing scores!
- accept all neighbors into the histogram (except those with size limits)
- after picking the maximum among neighbors, make the whole warp compute its inbound size and check the constraints, repeating the maximum sampling if it fails

=> as of now, picking the maximum is already done with the whole warp, as is the inbound size check, so the two are compatible. This way, filling the histogram can be fully warp-parallel!
=> moreover, if you find enough valid candidates before checking every neighbor, you do fewer inbound size checks!

This could be slightly worse if you find a way to compress the histogram (fill in the empty spots left by invalid options).

Note: runtime is not deterministic because neighbors are not sorted, so the number of time the costly constraint check on the inbound set, during the candidates kernel, runs depends on which histogram chunk each neighbor is in!

---

Solution for deduping neighbors:

- host, pick 8 nodes at random, compute their maximum neighborhoods size, use double that as our reference "neighborhood_size"
- instantiate the neighbors array with neighborhood_size slots per node, each neighborhood_size section will be threated as a global hash-table
- the new neighbors kernel just does one pass, no need to first count and then write
- check if a node is in the SM hash-set (a cache, essentially), if so, skip it
- if a node is not in the SM hash-set, insert it in the global hash-table (if it is already in it, just skip it)

- finish off with a thrust map to change hash-table bins into uint32s and a pack to remove empty spaces and compute the neighbors_offsets

Maybe: make the SM hash-set a bloom filter that allows false negatives, but not positives (the opposite of the normal bloom filter)

How to use hash-sets:
- SM stays an hash-set until full
- after it becomes full, finish the current warp-level work and SORT it
- from the point the SM hash-set is sorted, access it with a binary search with key being the hash
  => this saves the cost of a full scan when the element is missing
=> experiment to do: try not using an hash function in SM too, relying on the node ids directly as the hash and key at the same time (no hash collision risk)
- treat GM as an hash-set without hash, using node ids directly (or an invertible hash function, but that woudl be a waste of compute)
- before the SM was full, write to GM only when a new value is added to SM
- after SM is full, try writing to GM whenever a value is not in SM and dedupe in GM

Alternative deduping idea:
- in both SM and GM use node ids as keys
- before doing its 32 reads, each warp increases the SM distinct count by 32, and decrements it afterwards by the number of duplicates it found (unless the increment brought distinct to less than 32 away from full, if this happens don’t undo it and just commit to the sync)
- this way, the distinct count can be used as a reliable, pessimistic, estimate of how full SM is
- whenever a warp needs to start and sees < 32 SM spaces it goes wait on a block sync
- after the sync, all threads in the block cooperate to sort SM
- threads then proceed to do a merge-sort in-place with on-the-fly deduplication of SM into GM (that is also kept sorted) (tough, but remember that SM’s size will be <= than GM, but this may still require out-of-place merging…)
- then empty SM and continue
- if GM is full, assert error
Possible issue: if the SM hash-set fill up too fast, you waste more time dumping SM into GM, than anything else...
Better:
- make SM and GM two hash sets (no hash function, use node id directly)
- dumping SM into GM is the same as having the whole block start doing bulk inserts in to the GM hash set
=> at this point just go with the other method, spilling SM on the fly

Idea:
Could try to correlate the touching set size and avg. hedge size to the neighborhood size, this way each node can be allocated a neighborhood size just based on its touching set size (already available)!

Possibility:
Go back to the count+scatter kernel for neighborhoods. Give a bit of GM to each block during the count to back-up the SM hash-set. Here use GM as hash-sets!
The scatter do it as it was before, but build a hash-set again in GM. Only that this time the GM is sized appropriately. After every negative check in shared memory, check if the value was already in GM just like before!
=> Nah, this would be just as tough, since you would still need to give "max_neighbors" initial GM size (a little less, if you remove the SM size from it and just count) to each node/block!

True solutions:
1) do this a few nodes at a time, allocate at most 1/4 max VRAM of oversized neighbors and divide the nodes in evently-sized groups that can fit in such amount of VRAM.
ISSUE: you can't resize memory allocations, you need to allocate double the amounts of neighbors by the end just to copy every deduped neighbors piece into the final, unified one array!
2) don't dedupe all at once! Like the old version, dedupe in SM only, but after SM is full, count every node not in it as a new distinct. Do the count+scatter as normal.
   Since now, after the first round, you will still have duplicates, repeat the process exactly as before, but rather than building the neighbors sets, just re-read those you created before.
   Naturally, you will construct sets such that the elements that hit SM will be first, and you know those already have no duplicates. So in the subsequent rounds you can sort the first
   "round*size_of_SM" elements of the previous set and use a binary search on them even before putting elements in SM for the current round. This way the size of SM is not anymore a limit,
   rather you are guaranteed to dedupe "size_of_SM" elements per round!
   Continue the process until you don't exceed SM's size anymore, aka you deduped everything.
ISSUE: none, really, except if SM is too small to significantly reduce the size of the first array. If that happens, you could allocate a bit (user-controllable) of GM to help SM during every round!
       Maximum memory violated? Allocate more GM! Otherwise, allocate as little as needed and rely on SM!

Final solution:
- count+scatter kernels, one node per block
- build the deduplicated neighborhood in the SM hash-set until you are about to insert in SM an element past its capacity, then stop and write the SM size as offset, also set a global flag per-node to signal that you didn't finish, plus write 1 to an identical flag shared among nodes (aka has anyone still not finished?)
=> before inserting an element in SM, run a binary search to see if its already in GM (GM starts empty tho)
- allocate in GM the size requested by each node and run the scatter that does exactly the same as the above kernel, but at last sorts SM's content and writes it to the new GM
- repeat until you have no block that wants to continue
||
Only true solution for neighbors:
- counting kernel, using 1.2x the maximum measured neighborhood size minus the SM hash table size in global memory
- since you are only counting, no need to put in GM what alreay went in SM
- now we need perfect deduplication in SM, but we can still put a cap on the probe length to avoid going over the whole table, IFF we set the same probe length both during insertion and query, because this way you know that there is no way a value can be further than "max probe length" from where the hash points, so no point in checking
- after you have the count, instantiate the right amount of memory and scatter, using SM again for early deduplication, now backing its content on GM, and GM used again as an hash-set (for this, you can reuse the current implementation of the dedupe kernel)
=> no need anymore for the costly (in terms of memory) thrust pipeline

---

OLD IDEA:
Every node can generate at most TWO gain-increasing moves, going either up or down and either left or right.
Therefore 'moves' just encodes +1/-1/0 for X first, then +1/-1/0 for Y, while 'gains' encodes their respective gain.
Gains are first computed in isolation, sorted, then each move's gain is updated assuming all moves before it being applied, for this 'moves' remain untouched (can't switch "direction"). Then the longest subsequence of gain-improving moves is applied.
Repeat until convergence.

In-isolation gain computation:
- 1-st kernel computes the force in all 4 directions for each node
  => iterate, for each node, on its touching hedges, and on each node in each hedge, for each node updating all 4 forces
- 2-nd kernel computes ONE positive tension per node, the highest gain one, that becomes the node's candidate move (use again MAX_CANDIDATES, renamed MAX_MOVES, to enable up to 4 moves, if tension is positive)
- only one move must involve any node, this means that after the second kernel we need an upward and a downward tree walk, exactly like 
for partitioning, that lets each find or not another with which to exchange
  => little variation on the upward-downward walk: specialize it for pairs, make it so that at the end the slots of nodes in a pair point one to the other, and slots of unaffected nodes contain UINT32_MAX
    => upward: make my target point to me
    => downward: point back to my target, if he still points to me
  => use UINT32_MAX - 1,2,3,4 to flag nodes that wanted to move to an empty cell dx,sx,up,down, keep them as roots if no-one else bet their gain while going to the empty cell

In-sequence gain updates:
- create one event per pair/swap
- rank pairs/swaps by gain, assign a rank to each node, giving both nodes in each pair the same rank
  - purpose of the rank: see if something else is scheduled to happen before me or not
- now we know that each node will attempt to move at most once
- one thread for each event (one per pair) goes and re-computes both involved forces and the tension assuming all previous events (by rank) happen
  => again, iterate, for each node, on its touching hedges, and on each node in each hedge, for each node updating all 4 forces using the node's new position IFF the node had a lower rank
- apply all moves up to the highest-ranked one with the highest updated cumulative gain

About how to encode events:
- to know, given a node, where it went, you can just use "slots" as it was after the upward-downward walks
- each node needs to be given a rank by the gain of its move, with both nodes in a swap having the same rank
  => the lowest-id node in the pair is the anchor for the event, the other node is found by just following the slot of the anchor
  => first, custom kernel to reduce how many individual swaps there are and give an offset ... nah
=> as soon as the pairing (upward-downward walk) kernel locks a pair, the lower-id node set to 1 a flag in an array of zeros, one per node. Then you exclusive scan the flags and another stupid kernel writes “ev_score” and “ev_lower_node” for each lower-id node in a pair. Finally, by using scores and the lower nodes, via which you can find the whole pair back, you can find the longest subsequence by computing ranks!
=> no need to rank events explicitly. Sort “ev_score” and “ev_lower_node”, then one thread per event goes for each event and writes to the lower id node and going up to the higher id one to write the rank, the rank being equal to the event’s position, that is the kernel’s thread id!

Criteria:
- move: element chooses what to do on its own, and can do it by itself (pending validity)
 => just check validity and apply the highest-gain subsequence
- swap/pair: multiple elements need to agree to partake in the same collective update, each element can be part of different collectives and need to agree on which is their chosen one (highest score one available), validity constraint are at the level of the collective
 => upward-downward tree walk algorithm

 Bonus: no need for the inverse-map coordinates->node! Nevermind, computing tension needs it...

 Wild idea: we could look for minimizing circuits of length > 2 !

---

Memory solution:
- overwrite neighbors in-place
- keep an array “neighbors_count” and:
  - stop using offsets[n+1]-offsets[n] to get neighbors count, use the new array
  - never iterate over neighbors via pointer increments directly
- upgrade the neighbors coarsening kernel to do one node per warp:
  - warps can write in place, since because of deduplication you will always be writing at or before where you read, assuming you sync the warp between reads and writes
  - warp-wise exclusive scan of a 0/1 flag variable (true iff the neighbor is new) to get the offset where the thread writes, also do a reduce of the flags such that each thread can use the total count of written neighbors to advance the pointer where to write the next iteration
  - for deduplication, use SM, divided among 4 warps (like in the initial neighbors kernel) plus GM to support it OR use SM as an hash-set and if you miss just do a linear scan of the already written values PLUS a warp ballot to see if any other thread was about to write the same value (if so, only the lowest lane - first 1 in the ballot - writes)

NONONONO, this could never work! The idea that "neighbors of a node never increase" is false! Terribly false, from the moment that you aggregate nodes, increasing their united neighbors set size w.r.t. each node by itself!

Remember: even local memory eats up global memory if you give 8k or so entries to each thread!!!!

---

Idea for neighbors coarsening:
- the issue is the large local memory buffer (~28GB)
  => must vanquish any large local memory structure
- upgrade both count and scatter to a node per warp and shared memory pre-lookup
- rather than the sort + binary search like for touching, if we always use the same hash-set structure for nodes, we can use the has-set itself for deduping:
  - easy when scattering, same pattern as the outbound part of touching
  - less easy for counting, since you would need, for a node, to go take its ungroup, and for each ungroup node, go over each of my group node's neighbors set to see if it's there

The warp divergence issue with the SM pre-lookup:
Some threads will hit SM postively, others not, thus some go to GM, others not.
The solution would be to make SM not into a pre-lookup but into a very precise index over GM, that surpasses what the hash-set can do and gives the result in almost always one access. Borrow ideas from database indices!

=> for the "oversized" buffer strategy, you can compute the expected maximum number of neighbors per node as “new_num_nodes / curr_num_nodes” * (the previous max_neighbors_per_node), this does not account for deduplication tho…so as you keep updating max_neighbors_per_node, it becomes more and more oversized…


Deduplication:
- upgrade SM hash-sets to:
  1) counting bloom filter or cuckoo Filter
  2) hash-set with eviction
  => with either option, revert back to an exact SM hash-set if the maximum entries count is <= SM size

Issue:
How can we know an upper bound to the number of neighbors per node?
What we know:
- new and prev number of nodes
- initial maximum number of neighbors estimate
Idea:
- update the maximum neighbors estimate by scaling it by new_num_nodes/curr_num_nodes
- allocate maximum neighbors * curr_num_nodes space
- each warp (that handles one new node / group) takes its group size * maximum neighbors space
  => to know where its space starts and ends, each node just re-uses the ungroup offsets, the ungroup offset * maximum neighbors gives you the offset inside the dedupe buffer

For hedges it's easier: their size can only decrease! Just give them a pre-allocated size equal to min(1, 1.2 * new_num_nodes/curr_num_nodes) the previous size!

TODO: move the "1.2" safety factor to a define!

Maximum duplicates count while handling one group: MAX_GROUP_SIZE^2 
 => each new neighbor of a node could appear MAX_GROUP_SIZE times if the node was a neighbor to all those that got grouped together, and then we could be considering a group where all MAX_GROUP_SIZE nodes had another group appear in its entirety among their neighbors
 => this comes in handy to tune the counting bloom filter!

Final upgrade:
Transform the "scatter" into a true scatter. While counting, write everything in GM too, then while scattering you just need to copy from the oversized memory over to the new - correctly size - memory, a pack!
=> do this with a runtime check, only when you have enough memory to keep allocated both the correctly-sized array and the oversized one!
=>=> cuda check memory, if twice the oversized buffer's amount of memory is available, go for the "fast" strategy!

---

In place pack technique:
- one block per node / hedge
- 2 warps per block
- build a stack in shared memory, allocate exact as much a shared memory as the difference between the smallest final set (min in the counters initially stored in offsets, skip the first 0) and the initial oversized allocated for each instance
- one warp does coalesced reads from the start of each set and puts on the stack the free cells (also keep in SM the stack pointer -> or better, two indices, the start and end of a cyclic buffer realizing a queue in SM)
- one warp reads from the end and for each value it sees pops a entry from the stack/queue and writes it there
- both warps stop as soon as the reach the size/count of the final set (or 1 - count, from the top)

=> issue: this does not “pack”, just moved all free space at the end of each set!

---

> a sequence of bad ideas...

Crazy idea:
- when building neighbors from scratch, I am already doing the n*d*h work of visiting each touching hedge and for each its nodes
- while computing neighbors from scratch I could use and hash-table and already track, for each node, it’s total score, incrementing it each time it’s deduplicated w.r.t. to the hedge giving the duplicate
- thus, if we have the excuse to always rebuild neighbors from scratch, since we use them for nothing more than the candidates kernel, just merge the two things!
=> there’s no need to build or dedupe kernels at all, just produce candidates while deduping neighbors, tagging those that failed the constraint checks
=> and even if you need to keep neighbors for some reason, you can just keep them sparse, doing just the counting step into the oversized array, and using the maximum count to instantiate the next oversized array…

=>=> new candidates kernel:
- one node per block
- large SM hash-set
- GM hash-table
=> nah, no need for the hash-set this way, because both hits and misses need to go write/update score in GM…
- SM hash-table
- GM hash-table with less space thx to SM?
|
- each thread of the block keeps the maximum score it has seen, at the end you do 4 block-wide maxes, masking duplicates, to get the candidates (after all, at least one thread must see the final value of each bin to write it…it can use that value to update its maximum!)
- better if each warp keeps a running list of the last best 4 candidates

The trick behind keeping neighbors in memory (and coarsening them along): they lower the $O(d \cdot h)$ complexity of computing candidate pairs strength to an $O(\delta)$. With just an upfront $O(d \cdot h)$ paid for construction them, and an additional $O(\delta)$ to coarsen them.

FALSE! The candidates kernel still goes for each node over its hedges and for each hedge over its pins to compute scores!! This is still O(d*h)!!
If you want the TRUE benefit:
opt 1) don’t keep neighbors, just dedupe (merge scores) while computing candidates
opt 2) keep neighbors WITH SCORES! Each neighbor brings along its score while you coarsen too (as before, dedupe = sum scores)

Best opt for no-neighbors candidates kernel:
Fill up SM, complete one pass over hedges, sort SM by score, update the maximums, sort again by node id, dump only its node ids (not scores) into GM, use a binary search into GM (one for ever duped strip of SM) to see if any node has already been seen when filling SM back up.

Neighborless two kernel approach:
- kernel 1 builds in the oversized global memory buffers the hash-map with the scores for each neighbor of each node
  => one node per block, for maximum SM, change to one per warp as fewer nodes (and neighbors) remain
  => it already adds the noise
  => each hash bin must be (score, id) as to be used for sorting as a 64bit integer
- CUB ascending segmented sort of each oversized buffer
- kernel 2 reads the sorted neighbors, tries them from the first and sets aside the first MAX_CANDIDATES valid ones
  => you can optimize this for speed and have one warp per node
  => lane 0 reads the neighbor, the whole warp helps check inbounds validity

Overall, the w/neighbors version should be faster, but this neighborless uses much less VRAM! Then there is the neighbors+scores, that avoids the candidates kernel all together, but wants double the neighbors VRAM…

Note: kernel 1 could use SM as a quick lookup for GM, like an hash-map neighbor->GM-idx for all those nodes that didn't get inserted in GM with probe-length 1. Those, maybe just using it as a cache, and then dump-insert it into GM is better...
=> using SM as an hash-map just like GM lowers the pressure on GM, reducing probe lenghts!
=> since GM will be sorted anyway, be smart, don't insert from SM into GM with the hash-map method, rather have all threads in the block, all warps, do contigous accesses over GM, and in each empty space they find write an element they pop atomically from SM.
=> before doing this SM->GM dump, use the "compacting" algorithm with all threads in the block over SM! Pack it into a dense stack (search the message "In place pack technique")!

Crazier idea:
Could we make the constraint check on inbounds (and size too) permanent by removing nodes from the neighbors set? It would be permanent-ish, because of set merges, but better than nothing!

Neighborscores variant:
- no scores when doing oversized & counting uniques
- scores when doing the exact scatter

Neighborless:
- must be done with the 3 steps, build GM hash-sets, CUB segmented sort, pick maximums and check constraints


avg. number of nodes in common between two hedges: d^2 / n
avg. number of nodes in common between k hedges: d^k / n^(k-1)

----

Separate handling for inbound and outbound / src and dst:
- two rounds of hedges counting, one for srcs, one for dsts, also keeping a "d_hedge_src_count" array!
- custom pack operation over twin segments OR! If the pack is stable, use it as it is now!
- coarsening touching hedges now uses two atomics with two counters, one inbound, one outbound! Then the scatter happens in one kernel, with two loops, one for each!

The two kernel strategy for touching is better.

The unlimited outbound mod needs:
- counter d_hedge_src_count
- two kernels with in between a sort to coarsen touching, like now, but dedupe each subset independently
- pack kernel variant that skips the first X elements of the (sparse) src set and Y elements of the dst!

Known flaw, or feature:
With duplicates admitted between inbound and outbound, the candidates kernel could visit the same hedge twice, if that hedge is both inbound and outbound to the same node. This is not an issue when there are no cycles, but what about when there are? Should the hedge count twice in neighbor-score or not? IMO there just should not be cycles...

----

Candidates kernel:
- I am already reaching nodes from every hedge they have in common with the present node, once per hedge
- in the histogram, I can keep a counter on every node starting with its inbound set size
- when a node is a destination in the hedge, I decrement its counter
- the final counter is the number of hedges the node would add to my inbound set, that is, the number of hedges NOT yet seen

Another upgrade that could have been:
When a neighbor fails constraint checks during candidate selection, delete it from neighbors by setting it to UINT32_MAX, and the pack will do the rest. Never retry an invalid neighbor at the coarser level. Sure, node merges may bring it back, but that it’s acceptable…

----

Initial partitioning for k-way:

- random init:
  - thrust fill partitions array with |N|/k entries for the numbers 0..k-1
  - thrust random (with see) scramble of the array
- build pins per partition
- propose in-isolation move (FM gain style)

- in-isolation moves and gains
- sort by (target partition, gain) in lex order
- segmented scan by target partition
- apply moves up to the one, for each partition, reaching its size limit (on thread per move / node) (atomics to update partition size…)
- even allow negative gain moves, all applied in parallel, just to scramble things up
- after T rounds of the above, switch to a best improving subsequence FM style like normal -> only this phase respects monotonically improving connectivity

----

Instead of updating pins to pins_in by removing outbound connections, why not generate connection/disconnection events for every hedge, regardless of in/out.
Then, we can filter the events, generating inbound set size variation events solely when the pins count going to zero is a result of an inbound hedge?
NO! It would fail, because you would be also have in the pin count the outbound pins, hence missing out on a disconnection unless the source pin moves too...

----

How to build an ordering of an hypergraph's nodes such that nodes that share strong connections are close together:
- let each node build an histogram over its neighbors, ranking each by total connection weight between them
- then extract the maximum out of each node's histogram, tie-breaking deterministically by node id
- for each node, count how many other have it as their maximum, then take every node with a count >=2 ("not alone") and use it as a team-leader
- each leader builds a chain of nodes, taking in, one after the other, all those that had chosen it as their maximum, and ordering them by the rank they had in the leader's histogram
- each chain then becomes one of the next "supernodes" and the process repeats, appending to each chain until a single chain is formed

In other words:
- each node picks a strongest "attractor"
- popular attractors are promoted to temporary leaders
- form local ordered groups around those leaders
- contract each group into a supernode
- repeat until one chain remains

----

Placement improvement:
Why just the 4 elements up-left-right-down, couldn't we have a custom stencil? Like the 8-neighbors stencil! After all, when tensions are proposed, then we build the matching, and thus regardless of candidates, it will work!
=> how: define a parametric struct with one entry per "adjacent node" and a corresponding enum that tells you the offset w.r.t. you for each adjacent slot!
=>=>=> this is the natural step to generalize towards hardware graphs instead of lattices!

----

Old initial placement IDEA:
- every warp handles a node and builds a SM histogram over its neighbors, containing their total connection weight with the node
  - when the histogram is full, with a certain probability, evict a small-valued entry
- construct linearized arborescences out of best-neighbor relations, every node is given a "string_id" identifying its arborescence, an "depth" inside it (distance from root), and a "score" inside such arborescence
- every node moves to its best neighbor, and to its best neighbor, and so on, until entering a cycle or reaching a dead end, in doing so:
  - every hop it takes, increases the original node's depth by 1
  - the score of each node it visits is incremented by the score such node held, if any, in the original node's histogram (maybe multiplied by 1/2 + 1/2*depth)
  - the node's string_id becomes the id of the root node (or the lowest of two root nodes forming a 2-cycle)
- return once every warp reached a root starting from its node
- sort nodes by the tuple (string_id, depth, score)
  => TODO: sorting by depth->score means that after a branch in the arborescence, nodes from all branches interleave, this is not desirable, rather, one branch should come first, the other next, or maybe even interleave ter finding out how many neighbors they have in common
- every warp handles a node again and builds an histogram over arborescences, going over all neighbors of the node and accumulating their connection weight for the arborescence they belong to
- same game as before to now order arborescences (use a stable sort, obv.)

----

Initial placement IDEA:
- bisection, k=2 epsilon=0.1 balanced partitioning, minimizes the cut-net metric! Hence, it gives you two sets of nodes that’s fine to put far from each other as they have little locality in between them!
- so if you do recursive bysection, you end up with partitions of partitions down to single nodes, order the nodes internally in each partition, and concatenate as you go back up the recursion!

Additional crazy/dumb ideas:

Crazy ass idea again, a reverse disjoint set:
- HP: the hypergraph is fully connected
- every node has a certain random id assigned
- every node builds the histogram over neighbors, treating it as ranking neighbors
- everyone picks the first neighbor (tie-break) and walks up the resulting pseudo tree, the path every node defined gets “remembered”
- upon reaching the root each node inherits its tree’s id, taken as the id of the lowest-id node in the 2-cycle root
- now each node picks its second best neighbor:
  - if it finds it has the same id it already has, do nothing
  - if it has a different id, the whole tree of the two with a smaller id is overridden with the other’s id, and the node causing the link remembers the additional path
- repeat the whole hypergraph is under the same id
- now used paths constitute a sort of maximum weight minimum hops spanning tree over the hypergraph
- to linearize this giant tree, then this is the problem...

Crazy ass ordering idea:
- give every node a real number in [-1, 1], starting randomly
- extract the highest degree hedge and put it at 0.0, propagate to all its pins with strength w(e) (normalized in 0-1) and they propagate it to their pins with the strength multiplied by the w(e’) they are seen with (propagate only once, for speed)
- extract another hedge, and compute how much it overlaps with the first, based on that put it at a certain distance, the propagate it a value to nodes too
- ultimately, make it so you lay a few different hedges on the range -1:1 and have them pull nodes along over it with a recursion of 2 over pins

Not so great idea:
- build trees out of best neighbors
- build pieces of arborescences inside each tree by looking at the second-best neighbor (among those in the same tree)
- repeat until you have only pairs
- order each pair randomly, and re-expand
- every time you look at a later tree, order it subtrees sequentially (greedy pop from queue - queue rank is the histogram sum in that component for the elements already taken)
- finally, order the outermost forest
=> ISSUE: ordering is sequential…

Another crazy idea:
- order hyperedges by highest w first
- give idx 0 to all nodes in the highest w hedge, idx 1 to all those NEW in the second, 2 to those NEW in the third and so on
- this gives you groups of node to order, groups already being globally ordered by id
- internally, order each group by the total weights of hedges each node partakes in
=> ISSUE: where TF is the locality...
||
I could add to any already seen once being seen by an hedge after the first the 1/2^n value of the current n-th hedge

Ordering IDEA:
- each hedge visits its pins and their incidence sets
- each hedge counts how many times it has seen other hedges among its pins (histogram over its overlapping hedges)
- each hedge ranks its pins by the total occurrencies of other hedges in the histogram that every pin owns -> a higher-up pin will have the highest overlapping incidence set with the hedge
- and then what!? Aggreate ranks per node and use them as proxies for their resulting overlap when together?
=> I am just doing random bullshit at this point...

IDEAS:
- instead of going p\*2 and p\*2+1, split partitions in p and p+num_p, this way you can “fold in half” events and start the right number of threads to add those of the p+num_p over their p counter part
- permanently allocate arrays for events outside the loop, sized as nodes
  => this way you can have a third array encoding the second node in every pair, and “pack” every two events in one while you recompute scores (use atomics to add scores?)
NAHHH! Too tedious!

----

How to run mobile net:
Build inbound sets on the GPU! Disable their construction the host (remove the call).
-> iterate hedges and their pins, count how many times each node is a pin (incident set size) and how many times is a source (source count)
-> there are NO duplicates, use pins count to size a segment for each node
-> keep two counters for each node, seen sources and seen destinations
=>=> HP: hedges have had their sources already deduplicated wrt destinations (keep the dst)
-> go over hedges and their pins again, insert sources and destinations per node at the offset given by their counter, after atomically incrementing the counter (the beauty of no dedupe)

New neighbors dedupe:
- mini kernel to add together set sizes from d_ungroup
- if there is space, allocate a new array, merge sets (with SM buffer) and pack
- if there’s no space, clear the existing array, access it with the new offsets, and like that re-run the build neighbors from touching sets and hedge
- check that the pack can work with arbitrary-sized offsets

In coarsenNeighbors, if cfg.sum_of_merged is true:
 - compute sum of sizes for each group, from neigh_offsets
 - initialize coarse_neigh_offsets to that sum
 - pass coarse_neigh_offsets to the kernel to access the hash-sets through it
 - if cfg.sum_of_merged is false, write coarse_neigh_offsets with all curr_max_neighbors
- do the same for hedges (no need for touching...)

----

How to fix all the crap in ordering.cu:
- build from the start two set of event arrays, one for odd-events one for even-events, like "d_even_event_node" and similar
- this way, you fill them at once, switching based on partition id
- then they go through patters in parallel, with each kernel running once for each of odd and even
- access both at once by simply doing >>1 to your partition id
- infer minimim events count per pair by doing the minimum of the difference between part_offsets on each array set
- sum from one array to the other directly, set to zero past the end, and a single scan is enough
- apply moves starting from the array where the sum, scan, and max occurred, fetch the paired move from the other array by adding your distance from the offset to the other array's offset for the same p>>1
=> what currently broke me is just the concatenation of odd and even

----

THIS DAMN GIANT pins-per-partition!!

IDEA TO TOLERATE A STUPIDLY LARGE NUMBER OF HEDGES:
- allocate pins-per-partition once per level
- scan hedge offsets deltas to count how many hedges have a single pin and to compute the index of each hedge among those with >1 pin
- allocate a row of pins per partition only for those hedges with >1 pin
=> ISSUE: by the time you reach level 0, you are gonna be back to the fully size of pins-per-partition anyway...
KEY IDEA:
- I need pins-per-partition, but only when it's not zero!
- I could use
  - a key-value representation (hash-table) -> NAH, still stupidly larger
  - a compressed form, where zeros are stored as (zero, count of zeros)
  => this would also be fine with the transformation in inbound-pins-per-partition, since that only subtracts, so you get a few zeros here and there, who cares

HOW TO BUILT PINS-PER-PARTITION (one every ref. repeat):
- use two arrays (CSR):
 - array of size |hedges|, called per-per-part-offsets
 - a contigous array of segments, of type "pin_count", a new struct containing the partition id alongside the pins count for an hedge and that partition
 - each segmented, pointed by an offset for an hedge, contains the sparse pins-per-partition of that hedge over partitions
- first kernel: one warp per hedge, go over pins and use a SM+GM hash-table (->TABLE<-) to count their uniques number and how mnay times each occurs, flush the SM in 
  => if memory is really right, do not flush, rebuild
- a scan over unique counts gives you the offsets => allocate the array of segments
- do a pack from hash-tables to the array of segments (or rebuild, depending on available memory)
=> ISSUE: forget the pack! You need exact immediate access, so you need to keep the hash-tables, but then memory may not be enough...rehashing?

=>=> OR, I come up with another version of the sparse representation...
TODO:
- pay double the memory
- build two pins-per-partition, the total-pins version and the inbound-pins version
- keep them updated, instead of rebuilding
=> POSSIBLE iff the sparse representation is "dynamic enough"

NEW pins-per-partition sparse data structure:
- define a struct (uint32_t, uint64_t), call it "entry"
- say that I have R rows (~60M) and C cols (~8k)
- allocate a matrix of R x (C / 64) entries
- iterate other structures and write, for each row, for each entry, a 1 or a 0 in the bit of the uint64_t value corresponding to the column index % 64,
    in the entry in column "column / 64", simultaneously increment by 1 the entry's uint32_t counter
- then do an exclusive scan of each row's entries, over only the uint32_t counters
- hence, the counter tell essentially how many bits were set to 1 before them in that row
- now, with a reduce on the last entry of each row, infer the total number of non-zero bits that were set (i.e. the total of all counters)
- moreover, the scan tells the offset where each row's data will commence
- use the full total to allocate a compressed array of segments, one segmente per row
- re-do the iteration over other structures to fill the new array
- to access a cell in the array, access first the matrix, read the offset of the row, and the specific entry, add to the entry's uint32_t counter,
    the count of bits in the uint64_t value that are 1 before your own bit; add the offset, and the final counter to get the cell exact offset
---> maybe I could even sacrifice memory, make the counters 64bit too, and not have any need for the offsets of rows, I could just do a scan over the whole matrix at
     once and accumulate all offsets in the then-64bit counters...

----

Solution to the slow OpenMP multistart technique in placement:
- the issue clearly is shitty coalescing of memory accesses with simultaneous kernel launches
- rethink the multistart strategy and move it all on the device -> create multiple solutions, and have single kernel launch, for each of my kernels, spawn a thread per node per solution, or a warp per node per solution, and all solutions update in parallel
=> this is batching of multi-start attempts in a single kernel launch!
=> for once, you can launch kernels with TWO DIMENSIONS in blocks, one dimension over nodes/hedges, the other over "solutions" or "attempts"!ù
- maybe it does not make sense to do this for the initial placement, that can run with OpenMP, but it does and will make sense for force directed refinement, that can be fully parallel! Still a pain to separate events per attempt, but whatever...

----

Placement quality estimation:


Idea - Steiner approximation:
- for each hedge, approximate its minimum Steiner tree span
- you cannot "ricochet" off of other lattice points
- for each node, add to the total distance the distance between it, and the closes among all other nodes in the same hedge
  => this the facto creates a minimum spanning tree over the hedge's pins, connecting each pin to the other one closes to it
    => the "minimum spanning tree" is defined over a higher-level fully connected graph where only lattice points occupied by
       the hedge's pins exist, and each edge is weighted by the manhattan distance between said points
    => hence, with the minimum spanning tree, you always pay the minimum path distance between node pairs, even if, in reality, you reuse links
  => the minimum spanning tree will surely have span <= 2x the minimum Steiner tree
- minimum spanning tree complexity: iterate over hedges, over pins of each, and for each pin over pins again (to find the closest), e+d^2
Little issue:
- a pin already part of the minimum spanning tree must not be reconsidered by subsequent nodes...

Solution:
- a random pin (e.g. the source, if one) starts as "marked"
- every pin calculates its distance from its closest marked pin, and tries to atomically acquire it with it
- pin contend the acquisiting by who is closer, the fine one holding, connects to the marked pin, adds its distance to the total,
  and becomes itself marked, while the previously marked pin is excluded
- repeat until every pin is excluded or marked
=> sequential over 'd' ...
=>=> can't we do Dijkstra inside each hedge!? In parallel, per pin, as if it were distributed?

Spanning tree algorithm:
- each warp gets to handle an hedge, that is a fully connected and weighted graph
- for each pin, track its min current distance from any pin already in the MST and a flag telling if the pin is itself in the MST already
- arbitrarily add the first pin to the MST
  - for consistency, always add the last pin
- reduce (argmin) the pin with smallest distance to the MST, and add it to it
  - every other pin updates its min distance from the MST based on the new pin (lower it iff closer to the new pin)
- repeat until no pin remains outside the MST
- no need to track which edges were used for the MST, we just need to accumulate the total weight used when adding pins to it


----

Compute cutnet IDEA:
- prepare a copy of the segmented hedge buffer
- map operation to replace each pin with its partition
- segmented sort inside each hedge
- filter operation to keep only (within each segmente) the even numbers that are followed by their value +1 (their odd partition in the pair)
  - not need exactly to remove the elements, but to spot relevant ones
- flag surviving elements and prefix sum the flags, this gives you a unique offset per element
- for each surviving element create an event containing the tuple (hedge weight, partition id / 2), divide by 2 to get the parent partition's id
  - could put in the event the hedge's id, and recover the weight later, but little would changes
  - could sort immediately after filtering, but work with a larger buffer...
- sort events by parent partition id, and do a segmented reduce within each parent id, that's each parent partition's cutnet cost

----

Found the failure - or better, uselessness - mode of the tree-based ordering once going to 2D placement:

My tree-ordering optimizer is not really producing "a good sequence."
It is producing a **binary separator hierarchy**.
Hilbert/Moore/Z-order preserve locality of contiguous intervals reasonably well, but they do not preserve the semantics of "this subtree is the left side of a cut, that subtree is the right side of the same cut."
A 2D recursive partition should be represented as a binary space partition / slicing tree, where each internal node is explicitly a horizontal or vertical cut and each leaf is a rectangle.
That is exactly the structure used by k-d trees and slicing floorplans.

Or well, in other terms (thanks GPT):

"""
Ok, say then that my sequence has this structure, a consequence of the tree optimization:
- every pair 4*i+1 and 4*i+2 has a mildly connection between them 
- every pair of groups (8*i+2, 8*i+3) and (8*i+4, 8*i+5) has a bit stronger connection between them
- every pair of groups (16*i+4 ... 16*i+7) and (16*i+8 ... 16*i+11) has a stronger connection between them
- and so on ...
Could there be a way to exploit this to map effectively the sequence from 1D to 2D?

What if the property became circular, as in:
- every pair 4*i+1 and 4*i+2 has a mildly connection between them, and so does every pair 4*i and 4*i+3
- every pair of groups (8*i+2, 8*i+3) and (8*i+4, 8*i+5) has a bit stronger connection between them, and so does every pair (8*i, 8*i+1) and (8*i+6, 8*i+7)
- every pair of groups (16*i+4 ... 16*i+7) and (16*i+8 ... 16*i+11) has a stronger connection between them, and so does every pair (16*i ... 16*i+3) and (16*i+12 ... 16*i+15)
- and so on ...
Could this help in the 2D layout efforth?
"""

The main idea is this:
- your sequence is not "just a sequence";
- it carries a **dyadic hierarchy**;
- so the right 2D map should read the bits of the index as tree decisions, not treat the index as a scalar to be sent through an off-the-shelf curve.

That is exactly the kind of hierarchy that Morton/Z-order encodes: bit prefixes correspond to recursive quadtree cells.
In Z-order, 2D coordinates are built by bit interleaving, and the order is equivalent to a depth-first traversal of a quadtree.

=> you have a nested "middle seam" structure!

Your first pattern says, roughly:
- at one scale, the interesting interaction is between the two middle elements of each block of 4;
- at the next scale, between the two middle groups of each block of 8;
- then again for 16, 32, ...
That means the strong relation is not "adjacent in the linear order" in the usual sense.
It is "adjacent across the recursive split seam".