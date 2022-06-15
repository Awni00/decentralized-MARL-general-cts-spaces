"""Module implementing Decentralized Continuous-Space Multi-Agent Q-learning in Stochastic Dynamic Games"""

import itertools
import numpy as np
from tqdm.auto import tqdm, trange


# TODO: add epsilon-BR check
def is_best_reply(Qi, policy, delta_i):
    """determines whether a policy is a best-reply according to the estimated Q-factor"""
    policy_value = Qi[range(len(policy)), policy]
    opt_value = np.max(Qi, axis=1)

    return np.all(policy_value >= opt_value - delta_i)


def update_Qi(Qi, x_t, next_state, ui_t, reward_i, beta_i, alpha_i_n):
    """performs one update-step of Q-learning"""
    Qi_new = np.copy(Qi)
    Qi_new[x_t, ui_t] = (1 - alpha_i_n) * Qi[x_t, ui_t] + alpha_i_n * (
        reward_i + beta_i * np.max(Qi[next_state, :])
    )
    abs_delta = np.max(np.abs(Qi_new - Qi))

    return Qi_new, abs_delta


def evaluate_Qs_quantized(
    quantized_agent_policies,
    transition_kernel,
    init_state,
    reward_funcs,
    betas,
    T,
    alpha_func,
):
    """
    Run one exploration phase to estimate Q-factors for each agent.

    Args:
        quantized_agent_policies (List[QuantizedPolicy]): Initial quantized policies.
            Contain state and action quantizers.
        transition_kernel (TranstionKernel): The stochastic dynamic game's transition kernel.
            Contains state and action spaces.
        init_state (object): The initial state. Lies in the state space.
        reward_funcs (List[function]): Stage-wise reward function for each agent.
        betas (List[float]): Discount factor for each agent.
        T (int): Exploration phase length.
        alpha_func (function): Step-size / learning rate function. e.g. n:|-> 1/n

    Returns:
        Tuple[List[ndarray], List[float]]:
            Qs (estimated Q-factor for each agent), Q_changes (maximal change between succesive Q-learning steps)
    """

    n_agents = transition_kernel.n_agents

    # get state and action quantizers from quantized_agent_policies for convenience
    state_quantizers = [
        quantized_agent_policy.state_quantizer
        for quantized_agent_policy in quantized_agent_policies
    ]
    action_quantizers = [
        quantized_agent_policy.action_quantizer
        for quantized_agent_policy in quantized_agent_policies
    ]

    # make sure there is one for each agent and everything makes sense.
    assert (
        len(quantized_agent_policies)
        == len(state_quantizers)
        == len(action_quantizers)
        == n_agents
    )

    # initialize n_ts; n_ts[i][x,u] gives the number of visits to (x_index, u^i_index) for agent i
    n_ts = [
        np.zeros(shape=(state_quantizers[i].n_bins, action_quantizers[i].n_bins))
        for i in range(n_agents)
    ]

    # initialize Q-factors for each agent (Qs[i] gives agent i's Q-factor)
    Qs = [
        np.zeros(shape=(state_quantizers[i].n_bins, action_quantizers[i].n_bins))
        for i in range(n_agents)
    ]
    Q_changes = []  # history of the maximal (across agents) change in Qs at time t

    x_t = init_state

    # iterate over time in exploration phase k
    for t in trange(0, T, leave=False):

        # choose actions
        # actual actions in original space
        actions_t = [
            quantized_agent_policies[i].get_action_with_exploration(x_t)
            for i in range(n_agents)
        ]

        # indices of action
        actions_t_indices = [
            quantized_agent_policies[i].action_quantizer.quantize_index(actions_t[i])
            for i in range(n_agents)
        ]

        # receive rewards for each agent
        rewards = [reward_funcs[i](x_t, actions_t) for i in range(n_agents)]

        # recive next state
        next_state = transition_kernel.transition(x_t, actions_t)

        # update n_ts, number of visits to (x_t, u^i_t) in kth exploration phase up to t
        for i in range(n_agents):
            ui_t = actions_t_indices[i]  # agent i's action index
            xi_t = state_quantizers[i].quantize_index(
                x_t
            )  # agent i's quantized state-index
            n_ts[i][xi_t, ui_t] += 1

        Q_change_t = []
        # update Q-factors
        for i in range(n_agents):

            ui_t_idx = actions_t_indices[i]  # agent i's action index
            xi_t_idx = state_quantizers[i].quantize_index(
                x_t
            )  # agent i's quantized state-index
            next_state_idx = state_quantizers[i].quantize_index(
                next_state
            )  # agent i's quantized next state inde
            reward_i = rewards[i]  # reward received by agent i
            beta_i = betas[i]  # agent i's discou t factor

            alpha_i_n = alpha_func(
                n_ts[i][xi_t, ui_t]
            )  # step size for agent i at current state-action pair

            Qs[i], Qi_change = update_Qi(
                Qs[i], xi_t_idx, next_state_idx, ui_t_idx, reward_i, beta_i, alpha_i_n
            )
            Q_change_t.append(Qi_change)

        # keep track of maximum change between Qi_t and Qi_t+1
        Q_change_t = np.max(Q_change_t)  # the maximum
        Q_changes.append(Q_change_t)

        # update x_t
        x_t = next_state

    return Qs, Q_changes


def quantized_q_learning_alg(
    quantized_agent_policies,
    transition_kernel,
    get_initial_state,
    reward_funcs,
    betas,
    T,
    alpha_func,
    n_exploration_phases,
    deltas,
    inertias,
    early_stopping=False,
    verbose=False,
):
    """
    Runs Quantized Continuous-Space Decentralized Multi-Agent Reinforcement Learning Algorithm.

    Args:
    quantized_agent_policies (List[QuantizedPolicy]): Initial quantized policies.
            Contain state and action quantizers.
        transition_kernel (TranstionKernel): The stochastic dynamic game's transition kernel.
            Contains state and action spaces.
        get_initial_state (function): Returns initial state.
        reward_funcs (List[function]): Stage-wise reward function for each agent.
        betas (List[float]): Discount factor for each agent.
        T (int): Exploration phase length.
        alpha_func (function): Step-size / learning rate function. e.g. n:|-> 1/n
        n_exploration_phases (int): Number of exploration phases to run.
        deltas (List[float]): Tolerance for sub-optimality for each agent.
        inertias (_type_): Inertia for each agent (probability of not updating policy at end of phase).
        early_stopping (bool, optional): Whether to stop if equilibrium is reached. Defaults to False.

    Returns:
        Tuple[List[ndarray], List[QuantizedPolicies], dict]:
            Qs (estimated Q-factor at final exploration phase),
            quantized_agent_policies (policies at final exploration phase),
            history (history of Q-factors, policies, best-replies of each agent)
    """

    # calculate n_agents
    n_agents = transition_kernel.n_agents

    # get state and action quantizers from quantized_agent_policies for convenience
    state_quantizers = [
        quantized_agent_policy.state_quantizer
        for quantized_agent_policy in quantized_agent_policies
    ]
    action_quantizers = [
        quantized_agent_policy.action_quantizer
        for quantized_agent_policy in quantized_agent_policies
    ]

    # make sure there is one for each agent and everything makes sense.
    assert (
        len(quantized_agent_policies)
        == len(state_quantizers)
        == len(action_quantizers)
        == n_agents
    )

    # keep track of history of policies, Q-factors, and whether the policy is deeemed a best-response
    policy_history = []
    Qs_history = []
    is_BR_history = []

    # receive initial state
    x_0 = get_initial_state()
    x_t = x_0

    # iterate over exploration phases
    for k in trange(n_exploration_phases):

        # log current policies
        policy_history.append(
            np.copy(
                [
                    quantized_agent_policy.get_policy_map()
                    for quantized_agent_policy in quantized_agent_policies
                ]
            )
        )

        # evaluate current policies (stationary)
        Qs, _ = evaluate_Qs_quantized(
            quantized_agent_policies,
            transition_kernel,
            x_t,
            reward_funcs,
            betas,
            T,
            alpha_func,
        )

        Qs_history.append(Qs)  # log current Q-factors

        is_BR_k = []
        if verbose:
            print(f"reached end of exploration phase {k}")
        for i in range(n_agents):
            if verbose:
                print(f"agent {i}'s policy: {quantized_agent_policies[i].get_policy_map()}")

            # calculate estimate of best-reply policy space
            full_quantized_policy_space_i = itertools.product(
                range(action_quantizers[i].n_bins), repeat=state_quantizers[i].n_bins
            )
            br_policy_space_i = [
                policy
                for policy in full_quantized_policy_space_i
                if is_best_reply(Qs[i], policy, deltas[i])
            ]

            # if agent i's policy is not a best reply, replace it with a best reply
            if quantized_agent_policies[i].index_policy not in br_policy_space_i:
                if verbose:
                    print("policy is not best-reply.")

                # with inertia, don't replace policy even if it's not a best reply
                if np.random.random() < 1 - inertias[i]:
                    updated_policy = br_policy_space_i[
                        np.random.choice(len(br_policy_space_i))
                    ]
                    quantized_agent_policies[i].update_index_policy(updated_policy)

                    if verbose:
                        print(
                            f"updating to {quantized_agent_policies[i].get_policy_map()}"
                        )

                else:
                    if verbose:
                        print("inertia activated. not updating.")
                # log whether agent i's policy is a best reply
                is_BR_k.append(False)
            else:
                is_BR_k.append(True)
                if verbose:
                    print("policy is best-reply.")

            if verbose:
                print()
        if verbose:
            print()

        is_BR_history.append(is_BR_k)

        if early_stopping and np.all(is_BR_k):
            # if all policies are a best-reply, equilibrium has been reached, so stop
            break

    history = {
        "Qs_history": Qs_history,
        "policy_history": policy_history,
        "is_BR_history": is_BR_history,
        "n_exploration_phases": k,
    }

    return Qs, quantized_agent_policies, history
