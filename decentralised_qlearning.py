import itertools
import numpy as np
from tqdm import tqdm

def initialize_policy(n_U, n_states):
    """
    returns a random policy given the # of states and # of actions.

    Args:
        n_U (int): number of admissable actions.
        n_states (int): number of states in state space.

    Returns:
        tuple: a random policy
    """
    return tuple(np.random.choice(range(n_U), size=(n_states,)))

def is_best_reply(Qi, policy, delta_i):
    '''estimate whether a policy is approximately a best-response to the learned Q-factor'''

    policy_value = Qi[range(len(policy)), policy]
    opt_value = np.max(Qi, axis = 1)

    return np.all(policy_value >= opt_value - delta_i)

def update_Qi(Qi, x_t, next_state, ui_t, cost_i, beta_i, alpha_i_n):
    """
    updates agent i's Q-factor given the received cost and next state.

    Args:
        Qi (np.ndarray): agent i's previous Q-factor.
        x_t (int): the state at time t.
        next_state (int): the state at time t+1.
        ui_t (int): the action taken at t.
        cost_i (float): the cost received after (x_t, u_t).
        beta_i (float): the discount factor for agent i.
        alpha_i_n (float): the update/learning rate of agent i.

    Returns:
        np.ndarray: the updated Q-factor of agent i.
    """

    Qi_new = np.copy(Qi)
    Qi_new[x_t, ui_t] = (1-alpha_i_n) * Qi[x_t, ui_t] + alpha_i_n * (cost_i + beta_i*np.max(Qi[next_state, :]))
    abs_delta = np.max(np.abs(Qi_new - Qi))

    return Qi_new, abs_delta

def initialize_q(n_states, n_U):
    '''initialize Q-factor (w/ zeros) according to # of states and actions.'''
    return np.zeros(shape=(n_states,n_U))

def random_action(n_U, state):
    '''uniformly choose from action_space'''
    return np.random.randint(n_U)

def policy_with_exploration(policy, state, n_U, rho):
    '''perform a random action w/ prob `rho`; otherwise follow given policy'''

    if np.random.random() < rho:
        return random_action(n_U, state)
    else:
        return policy[state]

def evaluate_Qs(agent_policies, n_states, n_Us, init_state, transition_state,
                cost_funcs, betas,  T, experimentation_probs, alpha_func):
    """evaluates Q-factors for each agent under the given stationary policies.

    Args:
        agent_policies (List): list of agents' policies.
        n_states (int): number of states.
        n_Us (List): list of length of action spaces for each agent.
        init_state (int): the initial state.x
        transition_state (function): the state transition function: (state, agent_actions) -> next_state.
        cost_funcs (List[function]): the cost funcction for each agent: (state, agent_actions) -> cost
        betas (List[float]): list of discount factors for each agent.
        T (int): length of exploration phases.
        experimentation_probs (List[float]): list of experimentation probabilities for each agent.
        alpha_func (function): update/learning rate function.

    Returns:
        tuple: Qs, Q_changes
    """

    n_agents = len(agent_policies)

    # initialize n_ts; n_ts[i][x,u] gives the number of visits to (x, u^i) for agent i
    n_ts = [np.zeros(shape=(n_states, n_Us[i])) for i in range(n_agents)]

    # initialize Q-factors for each agent (Qs[i] gives agent i's Q-factor)
    Qs = [initialize_q(n_states, n_Ui) for n_Ui in n_Us]
    Q_changes = [] # history of the maximal (across agents) change in Qs at time t

    x_t = init_state

    # iterate over time in exploration phase k
    for t in range(0, T):

        # choose actions
        actions_t = [policy_with_exploration(agent_policies[i], x_t, n_Us[i], experimentation_probs[i])
                     for i in range(n_agents)]

        # receive costs for each agent
        costs = [cost_funcs[i](x_t, actions_t) for i in range(n_agents)]

        # recive next state
        next_state = transition_state(x_t, actions_t)

        # update n_ts, number of visits to (x_t, u^i_t) in kth exploration phase up to t
        for i in range(n_agents):
            ui_t = actions_t[i]
            n_ts[i][x_t, ui_t] += 1

        Q_change_t = []
        # update Q-factors
        for i in range(n_agents):

            ui_t = actions_t[i]
            cost_i = costs[i]
            beta_i = betas[i]

            alpha_i_n = alpha_func(n_ts[i][x_t, ui_t])

            Qs[i], Qi_change = update_Qi(Qs[i], x_t, next_state, ui_t, cost_i, beta_i, alpha_i_n)
            Q_change_t.append(Qi_change)

        # keep track of maximum change between Qi_t and Qi_t+1
        Q_change_t = np.max(Q_change_t) # the maximum
        Q_changes.append(Q_change_t)


        # update x_t
        x_t = next_state


    return Qs, Q_changes

def q_learning_alg1(n_Us, n_states, cost_funcs, betas,
                    get_initial_state, transition_state,
                    n_exploration_phases, T, experimentation_probs,
                    alpha_func, deltas, inertias, early_stopping=False):
    """Multi-agent Q-learning algorithm from [...].

    Args:
        n_Us (List): list of length of action spaces for each agent.
        n_states (int): number of states.
        cost_funcs (List[function]): the cost funcction for each agent: (state, agent_actions) -> cost
        betas (List[float]): list of discount factors for each agent.
        get_initial_state (function): function which returns initial state.
        transition_state (function): the state transition function: (state, agent_actions) -> next_state.
        n_exploration_phases (int): the number of exploration phases
        T (int): length of exploration phases.
        experimentation_probs (List[float]): list of experimentation probabilities for each agent.
        alpha_func (function): update/learning rate function.
        deltas (List[float]): list of tolerance for suboptimality of each agent.
        inertias (List[float]): list of inertias of each agent
        early_stopping (bool, optional): whether to stop when equilibrium is reached. Defaults to False.

    Returns:
        Tuple: Qs, agent_policies, (policy_history, Qs_history, is_BR_history)
    """

    # calculate n_agents
    n_agents = len(n_Us)

    # initialize policies for each agent (agent_policies[agent] gives agent's policy)
    agent_policies = [initialize_policy(n_Ui, n_states) for n_Ui in n_Us]

    # keep track of history of policies, Q-factors, and whether the policy is deeemed a best-response
    policy_history = []
    Qs_history = []
    is_BR_history = []

    # receive initial state
    x_0 = get_initial_state()
    x_t = x_0

    # iterate over exploration phases
    for k in tqdm(range(n_exploration_phases)):

        # log current policies
        policy_history.append(agent_policies.copy())

        # evaluate current policies (stationary)
        Qs, _ = evaluate_Qs(agent_policies, n_states, n_Us, x_t, transition_state,
                            cost_funcs, betas, T, experimentation_probs, alpha_func)

        Qs_history.append(Qs) # log current Q-factors

        is_BR_k = []
        for i in range(n_agents):
            # calculate estimate of best-reply policy space
            full_policy_space_i = itertools.product(range(n_Us[i]), repeat=n_states)
            br_policy_space_i = [policy for policy in full_policy_space_i if is_best_reply(Qs[i], policy, deltas[i])]

            # if agent i's policy is not a best reply, replace it with a best reply
            if agent_policies[i] not in br_policy_space_i:
                # with inertia, don't replace policy even if it's not a best reply
                if np.random.random() < 1 - inertias[i]:
                    agent_policies[i] = br_policy_space_i[np.random.choice(len(br_policy_space_i))]

                # log whether agent i's policy is a best reply
                is_BR_k.append(False)
            else:
                is_BR_k.append(True)

        is_BR_history.append(is_BR_k)

        if early_stopping and np.all(is_BR_k):
            # if all policies are a best-reply, equilibrium has been reached, so stop
            return k, Qs, agent_policies, (policy_history, Qs_history, is_BR_history)


    return Qs, agent_policies, (policy_history, Qs_history, is_BR_history)

