Interesting (but not useful) findings!

What I expected:
Now the score is cumulative from one level to another, candidates are sampled not based on the connection strength of coarse nodes with one another, but based on the total connection strenght of the original nodes inside each cluster with those of other clusters!
This COULD have lead to better guidance during coarsening, especially with the original hypergraph in mind.

What happens:
Results worsens.
More coarsening levels required.
Execution time increases.
Likely because now strong "attractor" nodes as of the original hypergraph become stronger and stronger as we coarsen, preventing any other node from forming meaningful clusters, except those swept in by attractors. That is, until attractors saturate. The original method instead, that re-computed scores at each level, was much more fair!
In brief, with this approach the attractive force of large clusters grows exponentially, abandoning all nodes that don't get swept in.
With my original method, instead, at every level scores are reset, and just depend on hyperedges and their individiual connections to clusters, leading to a fair amount of participation by every (coarse) node, proportially to its number of connections.