import itertools
import numpy as np
import networkx as nx
from tqdm import tqdm

from multi_agent_learning import evaluate_Qs, is_best_reply


def learn_br_graph(n_agents, n_states, n_Us, init_state, transition_state, reward_funcs, betas,
                   T, experimentation_probs, alpha_func, deltas):
    """learn a game's best-reply graph

    Args:
        n_agents (int): number of agents in game.
        n_states (int): number of states in game.
        n_Us (List[int]): list of number of actions available to each agent.
        init_state (int): the initial state.
        transition_state (function): the state transition function: (state, agent_actions) -> next_state.
        reward_funcs (List[function]): the reward function for each agent: (state, agent_actions) -> reward.
        betas (List[float]): list of discount factors for each agent.
        T (int): length of learning phase.
        experimentation_probs (List[float]): list of experimentation probabilities for each agent.
        alpha_func (function): update/learning rate function.
        deltas (List[float]): list of tolerance for suboptimality of each agent.

    Returns:
        networkx.DiGraph: br_graph
    """


    # generate joint policy space
    agent_policy_spaces = [list(itertools.product(range(n_Us[i]), repeat=n_states)) for i in range(n_agents)]
    joint_policy_space = list(itertools.product(*agent_policy_spaces))

    br_graph = nx.DiGraph()
    br_graph.add_nodes_from(joint_policy_space)


    for jpolicy in tqdm(joint_policy_space):

        # learn Q-values for joint policy
        Qs, Q_changes = evaluate_Qs(jpolicy, n_states, n_Us, init_state, transition_state,
                                    reward_funcs, betas,  T, experimentation_probs, alpha_func)

        # iterate over agents and their Q-factors
        for i, Q in enumerate(Qs):
            # generate agent i's policy space
            full_policy_space_i = itertools.product(range(n_Us[i]), repeat=n_states)

            # compute agent i's best reply policies and add edges to BR graph
            br_edge_i = [(jpolicy, sub_agent_br_policy(jpolicy, policy, i))
                          for policy in full_policy_space_i if is_best_reply(Q, policy, deltas[i])]

            br_graph.add_edges_from(br_edge_i, agent=i)

    return br_graph


def sub_agent_br_policy(joint_policy, agent_br_policy, agent_ind):
    '''substitute an agent's best-reply policy into a joint policy'''
    return tuple(ap if i!=agent_ind else agent_br_policy for i,ap in enumerate(joint_policy))


def transition_matrix_from_br_graph(br_graph, agent_inertias, joint_policy_space):
    """calculates the Markov transition matrix of the best-reply process from the best-reply graph.

    Args:
        br_graph (networkx.DiGraph): best-reply graph of game.
        agent_inertias (List[float]): list of inertias of each agent.
        joint_policy_space (List[tuple]): the joint-agent policy space of the game.

    Returns:
        Tuple[np.ndarray, dict]: transition_matrix, jps_dict
    """

    # dictionary to translate joint policy to its index in the joint policy space
    jps_dict = {joint_policy: ind for ind, joint_policy in enumerate(joint_policy_space)}
    n_jps = len(joint_policy_space) # size of joint policy space

    transition_matrix = np.zeros(shape=(n_jps, n_jps))

    for jp_i in joint_policy_space:
        for jp_j in joint_policy_space:
            P_tr = joint_transition_prob(br_graph, jp_i, jp_j, agent_inertias)
            i, j = jps_dict[jp_i], jps_dict[jp_j]
            transition_matrix[i][j] = P_tr

    return transition_matrix, jps_dict


def joint_transition_prob(br_graph, joint_policy1, joint_policy2, agent_inertias):
    '''returns the probability of transitioning from one joint policy to another'''
    return np.product([agent_tr_prob(br_graph, joint_policy1, agent_policy, agent, agent_inertia)
                       for agent, (agent_policy, agent_inertia) in enumerate(zip(joint_policy2, agent_inertias))])


def agent_tr_prob(br_graph, joint_policy, agent_policy, agent, agent_inertia):
    '''returns probability of an agent transitioning to a particular policy from a given joint policy'''

    # compute agent's best-reply policies
    agent_br_policies = get_agent_br_policies(br_graph, joint_policy, agent)

    # given policy is current policy
    if joint_policy[agent] == agent_policy:
        # current policy is agent's best-reply (or no other BR's)
        if agent_policy in agent_br_policies or len(agent_br_policies)==0:
            return 1
        # inertia
        else:
            return agent_inertia
    # given policy is not current policy
    else:
        # given policy is agent's best-reply
        if agent_policy in agent_br_policies:
            return (1 - agent_inertia) / len(agent_br_policies)
        # given policy is not agent's best-reply
        else:
            return 0



def get_agent_br_policies(br_graph, joint_policy, agent):
    return [edge[1][agent] for edge in br_graph.out_edges(joint_policy, data=True) if edge[2]['agent']==agent]