import itertools
import numpy as np
from tqdm import tqdm


def get_random_policy(action_space, n_states):
    return tuple(np.random.choice(action_space, size=(n_states,)))

def is_best_reply(Qi, policy, delta_i):
    '''estimate whether a policy is approximately a best-response to the learned Q-factor'''

    policy_value = Qi[range(len(policy)), policy]
    opt_value = np.min(Qi, axis = 1)

    return np.all(policy_value <= opt_value + delta_i)

def update_Qi(Qi, x_t, ui_t, cost_i, beta_i, alpha_i_n):
    Qi_new = np.copy(Qi)
    Qi_new[x_t, ui_t] = (1-alpha_i_n) * Qi[x_t, ui_t] + alpha_i_n * (cost_i + beta_i*np.min(Qi[x_t, :]))
    abs_delta = np.max(np.abs(Qi_new - Qi))

    return Qi_new, abs_delta

def random_policy(action_space, state):
    '''uniformly choose from `action_space`'''
    return np.random.choice(action_space)

def randomize_action(policy, state, action_space, rho):
    '''follow random policy w/ prob `rho`; otherwise follow given policy'''

    if np.random.random() < rho:
        return random_policy(action_space, state)
    else:
        return policy[state]

def q_learning_alg1(Us, n_states, cost_funcs, betas, get_initial_state, transition_state,
                    n_exploration_phases, T, experimentation_probs, alpha_func, deltas, inertias):


    # initialize Q_space? NOTE: did not account for compactness of space
    # initialize sequence {T_k}: length of exploration phase k
    # initialize exploration prob \rho
    # initialize inertia \lambda
    # initialize tolerance level for sub-optimality
    # initialize \alpha^i_n sequence of step sizes

    # calculate n_agents
    n_agents = len(Us)

    # initialize policies for each agent (agent_policies[agent] gives agent's policy)
    agent_policies = [get_random_policy(Ui, n_states) for Ui in Us]

    # initialize Q-factors for each agent (Qs[agent] gives agent's Q-factor)
    Qs = [np.zeros(shape=(n_states, len(Ui))) for Ui in Us]

    # keep track of maximal change in Q between updates
    Q_changes = []

    # keep track of best responses
    is_best_response = [[] for i in range(n_agents)]


    x_0 = get_initial_state()
    x_t = x_0

    t = 0

    # iterate over exploration phases
    for k in tqdm(range(n_exploration_phases)):

        # initialize n_ts number of visits to (x,u^i) in kth exploration phase up to t
        # n_ts[i][x,u] gives the number of visits to (x, u^i) for agent i
        n_ts = [np.zeros(shape=(n_states, len(Us[i]))) for i in range(n_agents)]

        # iterate over time in exploration phase k
        for t in range(t, t + T):
            #print(f'k={k}; t={t}')

            # choose actions
            actions_t = [randomize_action(agent_policies[i], x_t, Us[i], experimentation_probs[i]) for i in range(n_agents)]
            #print(f'actions: {actions_t}')

            # receive costs
            costs = [cost_funcs[i](x_t, actions_t) for i in range(n_agents)]
            #print(f'costs: {costs}')

            # recive next state
            next_state = transition_state(x_t, actions_t)
            #print(f'next state: {next_state}')

            # update n_ts number of visits to (x_t, u^i_t) in kth exploration phase up to t
            for i in range(n_agents):
                ui_t = actions_t[i]
                n_ts[i][x_t, ui_t] += 1

            # update Q-factors
            Q_change_k = []
            for i in range(n_agents):

                ui_t = actions_t[i]
                cost_i = costs[i]
                beta_i = betas[i]

                alpha_i_n = alpha_func(n_ts[i][x_t, ui_t])

                Qs[i], Qi_change = update_Qi(Qs[i], x_t, ui_t, cost_i, beta_i, alpha_i_n)
                Q_change_k.append(Qi_change)

            # keep track of maximum change between Qi_t and Qi_t+1
            Q_change_k = np.max(Q_change_k)
            Q_changes.append(Q_change_k)


            # update x_t
            x_t = next_state

         # calculate estimate of best reply policy space
        for i in range(n_agents):
            full_policy_space_i = itertools.product(Us[i], repeat=n_states)
            br_policy_space_i = [policy for policy in full_policy_space_i if is_best_reply(Qs[i], policy, deltas[i])]

            # if agent i's policy is not a best response replace it with a best response
            if agent_policies[i] not in br_policy_space_i:
                # with inertia, don't replace policy even if it's not a best response
                if np.random.random() < 1 - inertias[i]:
                    agent_policies[i] = br_policy_space_i[np.random.choice(len(br_policy_space_i))]

                # log whether agent i's policy is a best response
                is_best_response[i].append(False)
            else:
                is_best_response[i].append(True)

        t+=1 # increment to start on next t next exploration phase
        #print()


        # reset Q-factors to anything in Q_space. (perhapse project) [NOTE: i'm ignoring this for now]
        # Q_space is necessarily compact when implementing in code (?)

    return Qs, agent_policies, (Q_changes, is_best_response)