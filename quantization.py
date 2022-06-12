"""Module implementing quantization and related object classes."""

import numpy as np


class DiscreteSpace:
    """
    A discrete space with a specified finite number of points.
    """

    def __init__(self, points):
        self.points = points
        self.n_points = len(points)

        self.is_discrete = True

    def is_in(self, point):
        return point in self.points


class ContinuousInterval:
    """
    A continuous interval between two real numbers.
    """

    def __init__(self, start_point, end_point):
        if start_point >= end_point:
            raise ValueError("`end_point` must be strictly larger than start point")

        self.start_point = float(start_point)
        self.end_point = float(end_point)

    def get_length(self):
        return self.end_point - self.start_point

    def is_finite(self):
        return np.isfinite(self.get_length())

    def is_in(self, point):
        return self.start_point <= point <= self.end_point


class UniformQuantizer:
    """
    Uniform quantizer for ContinuousInterval.

    Samples `n_bins` points uniformly from the interval and quantizes by choosing the closest point.
    """

    def __init__(self, space, n_bins):
        if not isinstance(space, ContinuousInterval):
            raise ValueError("`space` must be of type ContinuousInterval")
        elif not space.is_finite():
            raise ValueError("`space` must be finite")

        self.space = space
        self.n_bins = n_bins

        self.quantization_points = np.linspace(
            space.start_point, space.end_point, n_bins
        )

        self.point_to_index_map = {
            q_point: i for i, q_point in enumerate(self.quantization_points)
        }
        self.index_to_point_map = {
            i: q_point for i, q_point in enumerate(self.quantization_points)
        }

    def quantize(self, point):
        if not self.space.is_in(point):
            raise ValueError("point is not is `space`")

        return self.quantization_points[self.quantize_index(point)]

    def quantize_index(self, point):
        if not self.space.is_in(point):
            raise ValueError("point is not is `space`")

        return np.argmin(np.abs(point - self.quantization_points))

    def map_index_to_point(self, index):
        return self.index_to_point_map[index]


class NullQuantizer:
    """
    A 'quantizer' for DiscreteSpace. Maps discrete points to integers in the range 0 - len(points).
    """

    def __init__(self, space):
        if not isinstance(space, DiscreteSpace):
            raise ValueError("`space` must of type DiscreteSpace")
        self.space = space
        self.n_bins = space.n_points

        # mapping points to integers 0 - n_points
        self.point_to_index_map = {
            point: i for i, point in enumerate(self.space.points)
        }
        self.index_to_point_map = {
            i: point for i, point in enumerate(self.space.points)
        }

    def quantize(self, point):
        return point

    def quantize_index(self, point):
        return self.point_to_index_map[point]

    def map_index_to_point(self, index):
        return self.index_to_point_map[index]


class TransitionKernel:
    """
    Transition kernel for a stochastic dynamic game.

    Contains the state space and action spaces for each agent. Transitions to the next state using current state and agent actions.
    """

    def __init__(self, state_space, action_spaces, transition_function):
        self.state_space = state_space
        self.action_spaces = action_spaces
        self.n_agents = len(action_spaces)
        self.transition_function = transition_function

    def transition(self, state, actions):
        if not self.state_space.is_in(state):
            raise ValueError("the state `x` is outside the `state_space`")

        for i, (action_space, action) in enumerate(zip(self.action_spaces, actions)):
            if not action_space.is_in(action):
                raise ValueError(
                    f"agent {i}'s `action` is outside their `action_space`"
                )

        x_next = self.transition_function(state, actions)

        return x_next


class QuantizedPolicy:
    """
    A quantized policy for a single agent.

    Policy takes quantized state as input and produces a quantized action as output. 
    Implements initialization of policy, getting actions from policy, and exploration.
    """

    def __init__(
        self,
        state_quantizer,
        action_quantizer,
        index_policy="random_init",
        exploration_prob=0,
    ):
        self.state_quantizer = state_quantizer
        self.action_quantizer = action_quantizer

        # randomly initialize policy
        if index_policy == "random_init":
            self.index_policy = tuple(
                np.random.choice(
                    range(action_quantizer.n_bins), size=(state_quantizer.n_bins,)
                )
            )

        # initialize with given policy (check it's valid)
        elif len(index_policy) == state_quantizer.n_bins and np.all(
            0 <= np.array(index_policy) < action_quantizer.n_bins
        ):
            self.index_policy = tuple(index_policy)

        else:
            raise ValueError(
                "given `index_policy` is not valid. must be a tuple of size `state_quantizer.n_bins` taking values in [0, `action_quantizer.n_bins`)"
            )

        self.exploration_prob = exploration_prob

    def get_action_index(self, state):
        state_index = self.state_quantizer.quantize_index(state)
        action_index = self.index_policy[state_index]
        return action_index

    def get_action(self, state):
        action_index = self.get_action_index(state)
        action = self.action_quantizer.map_index_to_point(action_index)
        return action

    def get_action_index_with_exploration(self, state):
        if np.random.random() < self.exploration_prob:
            action_index = np.random.randint(self.action_quantizer.n_bins)
        else:
            action_index = self.get_action_index(state)

        return action_index

    def get_action_with_exploration(self, state):
        action_index = self.get_action_index_with_exploration(state)
        action = self.action_quantizer.map_index_to_point(action_index)
        return action

    def update_index_policy(self, updated_index_policy):
        self.index_policy = updated_index_policy

    def get_policy_map(self):
        policy_map = {
            self.state_quantizer.map_index_to_point(
                x_idx
            ): self.action_quantizer.map_index_to_point(u_idx)
            for x_idx, u_idx in enumerate(self.index_policy)
        }

        return policy_map
