import itertools
import numpy as np
import networkx as nx

from tqdm import tqdm
from functools import partialmethod

import matplotlib.pyplot as plt

from sympy import *


from br_graph_analysis import learn_br_graph, transition_matrix_from_br_graph
from multi_agent_learning import q_learning_alg1

def agent_distance(pi1, pi2):
    '''computes the number of agents in which the policies differ'''

    dist = 0

    for agent_policy1, agent_policy2 in zip(pi1, pi2):
        if agent_policy1 != agent_policy2:
            dist += 1

    return dist


def get_deviating_agent(pi1, pi2):
    '''if policies have a single deviating agent, returns that agent'''

    assert(agent_distance(pi1, pi2) == 1)

    for agent, (agent_policy1, agent_policy2) in enumerate(zip(pi1, pi2)):
        if agent_policy1 != agent_policy2:
            return agent


def reorder_matrix_by_states(matrix, reorder):
    return matrix[reorder][:, reorder]


def gen_adj_matrix(jp_order, jp_dict):
    '''generates the best-reply graph from the order of joint policies'''

    n_policies = len(jp_dict)
    inv_jp_dict = {jp: ind for ind, jp in jp_dict.items()}

    # initialize adjacency matrix
    adjacency_matrix = np.zeros((n_policies, n_policies))

    for rank, jp1 in enumerate(jp_order):
        # if cost of jp2 is worse, and agent distance is 1
        for jp2 in jp_order[rank:]:
            if agent_distance(jp1, jp2) == 1:
                jp1_ind = inv_jp_dict[jp1]
                jp2_ind = inv_jp_dict[jp2]

                # add an edge
                adjacency_matrix[jp2_ind][jp1_ind] = 1

    return adjacency_matrix



def gen_graph_from_adj_matrix(adj_matrix, jp_dict):
    '''generates a networkx Digraph from adjacency matrix with nodes labelled'''

    # get joint policy space from jp_dict
    joint_policy_space = list(jp_dict.values())

    # initialize the best-reply graph and add its nodes
    br_graph = nx.DiGraph()
    br_graph.add_nodes_from(joint_policy_space)

    # get the directed edges in the graph from the adjacency matrix
    edges = np.where(adj_matrix == 1)

    # add all edges to graph (with attribute for deviating agent)
    for jp_from_ind, jp_to_ind in zip(*edges):
        jp_from = jp_dict[jp_from_ind]
        jp_to = jp_dict[jp_to_ind]
        deviating_agent = get_deviating_agent(jp_from, jp_to)

        br_graph.add_edge(jp_from, jp_to, agent=deviating_agent)

    return br_graph

def gen_br_graph_from_order(jp_order, jp_dict):

    adj_matrix = gen_adj_matrix(jp_order, jp_dict)
    return gen_graph_from_adj_matrix(adj_matrix, jp_dict)


def gen_br_graph_space(jp_dict):
    '''generates the space of valid best-reply graphs through all possible orderings of joint policy space'''

    # get joint policy space from jp_dict
    joint_policy_space = list(jp_dict.values())

    # initialize best-reply graph space
    br_graph_space = []

    # for each permuation of joint policies, generate best-reply graph
    # and add it the space
    for perm in tqdm(itertools.permutations(range(len(joint_policy_space)))):
        jp_order = [jp_dict[i] for i in perm]
        adj_matrix = gen_adj_matrix(jp_order, jp_dict)
        br_graph = gen_graph_from_adj_matrix(adj_matrix, jp_dict)

        br_graph_space.append((jp_order, br_graph))

    return br_graph_space


def get_team_opt_prob(team_opt_policy, br_graph, jp_dict, return_all=False):
    '''gets symbolic expression for the probability of being absorbed into the team-optimal policy'''

    inv_jp_dict = {jp: ind for ind, jp in jp_dict.items()}
    joint_policy_space = list(jp_dict.values())
    n_agents = len(joint_policy_space[0])


    # indices of absorbing states, transient states, and the team-optimal state
    abs_states = [inv_jp_dict[node] for node in br_graph.nodes if br_graph.out_degree(node) == 0]
    trans_states = [i for i in range(len(joint_policy_space)) if i not in abs_states]
    team_opt = inv_jp_dict[team_opt_policy]

    # reorder abs_states so that team-optimal state is at the end
    abs_states.remove(team_opt)
    abs_states.append(team_opt)

    # create inertia symbolic variables
    agent_inertias = symbols(f'lambda_0:{n_agents}')

    # get transition matrix of Markov chain from best-reply graph
    P_, jps_dict = transition_matrix_from_br_graph(br_graph, agent_inertias, joint_policy_space)

    # calculate Q and R submatrices according absorbing Markov chain tools
    Q_ = P_[trans_states][:, trans_states]
    Q = Matrix(Q_).applyfunc(nsimplify)

    R_ = P_[trans_states][:, abs_states]
    R = Matrix(R_).applyfunc(nsimplify)

    # get the transition matrix in absorbing canonical form
    reorder = trans_states + abs_states
    P_ro_ = P_[reorder][:, reorder]
    P_ro = Matrix(P_ro_).applyfunc(nsimplify)

    # calculate the fundamental matrix N
    Id = Matrix(np.eye(Q.shape[0])).applyfunc(nsimplify)
    N = simplify((Id - Q).adjugate() / (Id - Q).det())

    # calculate the absorbing matrix B
    B = simplify(N@R)

    # calculate P_inf in absorbing canonical form
    left_side = np.zeros((len(abs_states) + len(trans_states), len(trans_states)))
    right_side = np.vstack((B, np.eye(len(abs_states))))
    P_inf_can = Matrix(np.hstack((left_side, right_side))).applyfunc(nsimplify)

    # calculate the final distribution of states assuming uniform random initialization
    uniform_dist = Matrix([1/len(joint_policy_space)]*len(joint_policy_space)).T
    final_dist = simplify(Matrix(uniform_dist @ P_inf_can))

    # calculate the absorbtion probability of the team-optimal policy
    team_opt_prob = final_dist[-1]

    if return_all:
        return agent_inertias, team_opt_prob, (P_ro, Q, R, N, B, P_inf_can)

    return agent_inertias, team_opt_prob


def get_emperical_abs_probs(game_problem, agent_inertias, n_trials=1000, T=1000, max_K=10_000):

    # unpack game
    n_Us, n_states, n_agents, reward_funcs, betas, get_initial_state, \
    transition_state, experimentation_probs, calc_alpha, deltas = game_problem

    agent_policy_spaces = [list(itertools.product(range(n_Us[i]), repeat=n_states)) for i in range(n_agents)]
    joint_policy_space = list(itertools.product(*agent_policy_spaces))


    # initialize data
    end_data = {p: [] for p in joint_policy_space}
    transition_data = {p: [] for p in joint_policy_space}

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    for trial in tqdm(range(n_trials), disable=False):
        K, Qs, agent_policies, (policy_history, Qs_history, is_BR_history) = q_learning_alg1(n_Us, n_states, reward_funcs, betas,
                                                                        get_initial_state, transition_state,
                                                                        max_K, T, experimentation_probs,
                                                                        calc_alpha, deltas, agent_inertias, early_stopping=True)

        init_policy = tuple(policy_history[0])
        end_data[init_policy].append(tuple(agent_policies))

        for p1, p2 in zip(policy_history[:-1], policy_history[1:]):
            transition_data[tuple(p1)].append(tuple(p2))


    markov_end_prob = {p1: {} for p1 in joint_policy_space}
    for p1 in joint_policy_space:
        arr = np.array(end_data[p1])
        # print('initial_policy = ', p1)
        for p2 in joint_policy_space:
            P12 = np.average(np.all(arr==p2, axis=1))
            markov_end_prob[p1][p2] = P12
            # print(f'prob ending at {p2} = {P12}')
        # print()

    # calculate frequency/probability of transitioning to each policy given a starting policy


    markov_transition_prob = {p1: {} for p1 in joint_policy_space}

    for p1 in joint_policy_space:
        arr = np.array(transition_data[p1])
#         print('initial_policy = ', p1)
        if len(arr) > 0:
            for p2 in joint_policy_space:
                P12 = np.average(np.all(arr==p2, axis=1))
                markov_transition_prob[p1][p2] = P12
#                 print(f'prob transitioning to {p2} = {P12}')
        else:
            for p2 in joint_policy_space:
                P12 = 1 if p1==p2 else 0
                markov_transition_prob[p1][p2] = P12
#                 print(f'prob transitioning to {p2} = {P12}')

#         print()

    emperical_transition_matrix = np.array([[markov_transition_prob[p1][p2] for p2 in joint_policy_space] \
                                            for p1 in joint_policy_space])
    emperical_end_probs = np.array([[markov_end_prob[p1][p2] for p2 in joint_policy_space] for p1 in joint_policy_space])


    return (markov_transition_prob, emperical_transition_matrix), (markov_end_prob, emperical_end_probs)