import itertools
import numpy as np
from tqdm import tqdm

from multi_agent_learning import *

def team_learning_alg(n_Us, n_states, reward_funcs, betas,
                    get_initial_state, transition_state,
                    n_exploration_phases, T, memory_lengths,
                    action_exp, policy_exp_opt, policy_exp_subopt,
                    alpha_func, br_tolerances, asp_tolerances, inertias):
    """
    An implementation of the decentralized Q-learning algorithm for
    Stochastic Dynamic teams in "Decentralized Learning for Optimality
    in Stochastic Dynamic Teams and Games with Local Control and Global
    State Information" by Yongacoglu, Arslan, and Yüksel.

    Args:
        n_Us (List[int]): list of size of action spaces for each agent.
        n_states (int): number of states.
        reward_funcs (List[function]): the reward funcction for each agent: (state, agent_actions) -> reward
        betas (List[float]): list of discount factors for each agent.
        get_initial_state (function): function which returns initial state.
        transition_state (function): the state transition function: (state, agent_actions) -> next_state.
        n_exploration_phases (int): the number of exploration phases.
        T (int): length of exploration phases.
        memory_lengths (List[int]): list of memory length/window size for each agent.
        action_exp (List[float]): list of action experimention rate for each agent.
        policy_exp_opt (List[float]): list of policy experimentation rate when policy is optimal for each agent.
        policy_exp_subopt (List[float]): list of policy experimentation rate when policy is suboptimal for each agent
        alpha_func (function): update/learning rate function.
        br_tolerances (float): the error tolerance generating the best-reply set
        asp_tolerances (float): the tolerance for error in determining aspiration and optimality.
        inertias (List[float]): list of inertias of each agent.

    Returns:
        Tuple: Qs, agent_policies, (policy_history, Qs_history, is_opt_history)
    """

    # calculate n_agents
    n_agents = len(n_Us)

    # initialize policies for each agent (agent_policies[agent] gives agent's policy)
    agent_policies = [initialize_policy(n_Ui, n_states) for n_Ui in n_Us]

    # initalized agent memories
    agent_memories = [np.zeros(memory_length) for memory_length in memory_lengths]
    # initialize agent aspirations
    agent_aspirations = [np.max(agent_memory) for agent_memory in agent_memories] # equivalently initialize to 0

    # keep track of history of policies, Q-factors, and whether the policy is deeemed a best-response
    policy_history = []
    Qs_history = []
    is_opt_history = []

    # receive initial state
    x_0 = get_initial_state()
    x_t = x_0

    # iterate over exploration phases
    for k in tqdm(range(n_exploration_phases)):

        # log current policies
        policy_history.append(agent_policies.copy())

        # evaluate current policies (stationary)
        Qs, _ = evaluate_Qs(agent_policies, n_states, n_Us, x_t, transition_state,
                            reward_funcs, betas, T, action_exp, alpha_func)

        Qs_history.append(Qs) # log current Q-factors

        is_opt_k = []
        for i in range(n_agents):
            # calculate estimate of best-reply policy space
            full_policy_space_i = list(itertools.product(range(n_Us[i]), repeat=n_states))
            br_policy_space_i = [policy for policy in full_policy_space_i if is_best_reply(Qs[i], policy, br_tolerances[i])]

            # calculate agent's reward score for current policy
            curr_reward = np.sum([Qs[i][x][agent_policies[i][x]] for x in range(Qs[i].shape[0])])

            # add reward score to agent's memory and clip memory to memory_length
            agent_memories[i] = np.insert(agent_memories[i], 0, curr_reward)[:memory_lengths[i]]

            # update agent's aspiration
            agent_aspirations[i] = np.max(agent_memories[i]) - asp_tolerances[i]

            # if agent i's policy is better its aspiration
            if curr_reward >= agent_aspirations[i]:
                # update policy according optimal experimentaiton levels
                agent_policies[i] = update_policy(agent_policies[i], full_policy_space_i, br_policy_space_i,
                                                  policy_exp_opt[i], inertias[i])

                # log whether agent i's policy is a best reply
                is_opt_k.append(True)
            else:
                # update policy according to suboptimal experimentaiton levels
                agent_policies[i] = update_policy(agent_policies[i], full_policy_space_i, br_policy_space_i,
                                                  policy_exp_subopt[i], inertias[i])
                is_opt_k.append(False)

        is_opt_history.append(is_opt_k)



    return Qs, agent_policies, (policy_history, Qs_history, is_opt_history)