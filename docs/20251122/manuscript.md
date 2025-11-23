### Presentation Script: Autonomous ACO for Bottleneck Bandwidth Maximization

**(Slide 1: Title and Introduction)**

"Good morning, everyone. My name is Takumi Senaha, from Tokyo Metropolitan University.

Today, I am presenting our research: **"Routing for Bottleneck Bandwidth Maximization using Ant Colony Optimization with an Autonomous Learning Mechanism in Dynamic Networks."**

The core problem we are addressing is fundamental to modern content delivery: how do we find the best path for high-bandwidth applications, like video streaming, in a network where the content source is constantly moving?

**(Slide 2: Background and Problem)**

In traditional **IP networks**, traffic is concentrated on central servers. This leads to inefficiency, high costs, and low fault tolerance.

**Information-Centric Networking, or ICN**, is a key solution. In ICN, routers themselves hold caches, allowing content to be provided from a much closer location. However, this creates a new challenge: these caches can be mobile or replicated. This means we need a dynamic routing system to find this moving content.

This leads us to our optimization problem.

**(Slide 3: Problem and Objective)**

This is where traditional **Ant Colony Optimization (ACO)** routing comes in. As you can see in the diagram, ACO is a distributed algorithm inspired by ant foraging. It's excellent for finding the _shortest_ path—a path where the evaluation value is the **Sum** of weights, like distance or delay.

But for video streaming, the _total_ delay doesn't matter if one link on the path is very slow. The entire stream is throttled by its weakest link.

Our problem is the **Maximum Bottleneck Link (MBL)** problem. Our evaluation value is the **Minimum** of the link weights. We must find the path where this _minimum_ bandwidth is maximized.

This is a fundamentally different goal. Our objective is to establish a distributed, ACO-based method to solve this MBL problem, especially in a network where bandwidth is actively fluctuating.

**(Slide 4: Proposed Method - Overview)**

Our proposed method modifies all three phases of ACO by introducing an **Autonomous Node Learning** mechanism.

Let's look at each phase.

**(Slide 5: Proposed Method - The 3 Phases)**

First, **(1) Path Selection**. We use a standard $\epsilon$-greedy strategy. 90% of the time, the ant selects a path based on pheromone ($\tau$) and bandwidth ($w$). 10% of the time, it explores a random path.

Second, **(2) Pheromone Deposit**. This is our first innovation. When an ant completes a path, it calculates its bottleneck bandwidth, $B$. It then checks the memory of the destination node, $K_j$. If the path $B$ is _better_ than the node's memory ($B \ge K_j$), it receives an **Achievement Bonus**, $B_a$. This deposits 1.5 times the normal pheromone, strongly reinforcing this new, high-quality path.

Third, **(3) Pheromone Evaporation**. We introduce a **Penalty**. Normally, pheromones evaporate at a constant rate, $V=0.98$. However, if an edge's bandwidth, $w_{ij}$, is _worse_ than the memory of the node it's leading to ($w_{ij} < K_j$), it's a "disappointing" path. We penalize this edge by evaporating its pheromones much faster, (in this case, 0.5), telling other ants to avoid it.

**(Slide 6: Proposed Method - The $K_v$ Learning Rule)**

So, the critical question is: how does this node memory, $K_v$, work? This is the core of our "Autonomous Node Learning."

The $K_v$ update rule has two distinct phases to handle a dynamic environment.

- **Phase A: When an ant notifies the node.**
  The node does _not_ just keep the best value forever. It uses a **Ring Buffer** (of size $N=10$). When a new bottleneck $B_t$ is reported, it's added to this buffer, and the oldest value is _forcibly discarded_.
  The node's memory, $K_v$, is then reset to be the **maximum of this buffer**. This is a **fast-forgetting** mechanism, which allows the node to quickly adapt if a high-bandwidth path suddenly becomes worse.

- **Phase B: When _no_ ant notifies the node.**
  What if a path becomes obsolete and ants stop visiting? To prevent this "stale" memory from persisting, we apply a **slow-forgetting** mechanism, or "aging." For every generation a node is _not_ visited, its $K_v$ value is slowly decayed by multiplying it by 0.999.

This dual-forgetting system—a fast ring buffer for adaptation and a slow decay for aging—is what allows our method to thrive in a fluctuating network.

**(Slide 7: Simulation and Results)**

We validated this method on a 100-node Barabási-Albert network.

First, in **Environment 1**, a static network with fixed bandwidths. As the left graph shows, our method quickly learns the best path. The **optimal path selection rate converges to 80%**.

More importantly, in **Environment 2**, a dynamic network. Here, the available bandwidth of the main "hub" links was changed _every single generation_ using an AR(1) model to simulate real-world traffic.
As the right graph shows, our method successfully **adapts to these constant fluctuations**, with the **optimal path selection rate rising to and stabilizing at 60%**. This demonstrates its robustness.

**(Slide 8: Conclusion and Future Work)**

In **conclusion**, our proposed ACO method, enhanced with an autonomous node learning mechanism, can effectively solve the Maximum Bottleneck Link problem. The dual-forgetting $K_v$ rule, using both a ring buffer and a slow decay, is critical for adapting to dynamic network environments.

For **future work**, we plan to expand this into a multi-objective problem, balancing both bandwidth _and_ delay. We also plan to conduct a detailed evaluation of convergence speed compared to other ACO algorithms.

Thank you for your attention."
