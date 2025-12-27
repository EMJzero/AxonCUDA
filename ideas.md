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

Remember: even local memory eats up global memory if you give 8k or so entries to each thread!!!!