# Decentralized Multi-Agent Reinforcement Learning: Convergence to Team-Optimality

This repository contains code associated with this [paper](https://www.awni.xyz/files/marl_report.pdf) studying the convergence of a decentralized multi-agent reinforcement learning algorithm to team-optimality.

**Abstract:**

> Reinforcement learning methods generally aim to find policies that are optimal for an agent to follow in a given environment. Single-agent learning typically uses Markov decision processes (MDPs) as a model for the environment. For multi-agent systems, stochastic dynamic games and teams are powerful generalizations that can model systems in which multiple agents act on the environment simultaneously. In multi-agent learning, the aim is similarly to find “optimal” policies for each agent. Furthermore, we might be interested in studying decentralized systems in which agents are unable to communicate with each other. Recently, a decentralized Q-learning algorithm was proposed with formal guarantees of convergence to an equilibrium joint policy. However, for cooperative team problems, convergence to equilibria is not enough to achieve optimality. In this paper, we study the algorithm’s convergence properties with respect to team-optimality and present results to characterize this probabilistically. In doing so, we present a criticism of the algorithm and suggest a future line of research.


The repository includes the following:

- `multi_agent_learning.py`: module implementing the multi-agent Q-learning algorithm.

- `br_graph_analysis.py`: module for learning and analyzing the best-reply graph of a game problem.

- `team_learning.py`: module implementing decentralized team-learning algorithm.

- `sim_utils.py`: module containing some basic utility functions for running simulations.

- look at the `notebooks` folder for examples using these modules (the notebooks are varied in content; at the moment this is just a dump of various experiments that are part of the research project).