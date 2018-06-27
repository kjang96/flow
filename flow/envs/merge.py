from flow.envs.base_env import Env
from flow.core import rewards

from gym.spaces.box import Box

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # maximum number of controllable vehicles in the network
    "num_rl": 5,
}


class WaveAttenuationMergePOEnv(Env):
    """Environment used to train autonomous vehicles to attenuate the formation
    and propagation of waves in an open merge network.

    Required from env_params:
        * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
        * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
        * target_velocity: desired velocity for all vehicles in the network, in m/s
        * num_rl: maximum number of controllable vehicles in the network

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the ego speed of the autonomous vehicles.

        In order to maintain a fixed observation size, when the number of AVs
        in the network is less than "num_rl", the extra entries are filled in
        with zeros. Conversely, if the number of autonomous vehicles is greater
        than "num_rl", the observations from the additional vehicles are not
        included in the state space.

    Actions
        The action space consists of a vector of bounded accelerations for each
        autonomous vehicle $i$. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

        In order to account for variability in the number of autonomous
        vehicles, if n_AV < "num_rl" the additional actions provided by the
        agent are not assigned to any vehicle. Moreover, if n_AV > "num_rl",
        the additional vehicles are not provided with actions from the learning
        agent, and instead act as human-driven vehicles as well.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity, while slightly penalizing small time headways
        among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sumo_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Environment parameter "{}" not supplied'.
                               format(p))

        # maximum number of controlled vehicles
        self.num_rl = env_params.additional_params["num_rl"]
        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()
        # names of the rl vehicles controlled at any step
        self.rl_veh = []
        # used for visualization
        self.leader = []
        self.follower = []

        super().__init__(env_params, sumo_params, scenario)

    @property
    def action_space(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"],
                   shape=(self.num_rl,),
                   dtype=np.float32)

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(5 * self.num_rl,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.vehicles.get_rl_ids():
                continue
            self.apply_acceleration([rl_id], [rl_actions[i]])

    def get_state(self, rl_id=None, **kwargs):
        self.leader = []
        self.follower = []

        # normalizing constants
        max_speed = self.scenario.max_speed
        max_length = self.scenario.length

        observation = [0 for _ in range(5 * self.num_rl)]
        for i, rl_id in enumerate(self.rl_veh):
            this_speed = self.vehicles.get_speed(rl_id)
            lead_id = self.vehicles.get_leader(rl_id)
            follower = self.vehicles.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.vehicles.get_speed(lead_id)
                lead_head = self.get_x_by_id(lead_id) \
                    - self.get_x_by_id(rl_id) - self.vehicles.get_length(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.vehicles.get_speed(follower)
                follow_head = self.vehicles.get_headway(follower)

            observation[5 * i + 0] = this_speed / max_speed
            observation[5 * i + 1] = (lead_speed - this_speed) / max_speed
            observation[5 * i + 2] = lead_head / max_length
            observation[5 * i + 3] = (this_speed - follow_speed) / max_speed
            observation[5 * i + 4] = follow_head / max_length

        return observation

    def compute_reward(self, state, rl_actions, **kwargs):
        # return a reward of 0 if a collision occurred
        if kwargs["fail"]:
            return 0

        # reward high system-level velocities
        cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

        # penalize small time headways
        cost2 = 0
        t_min = 1  # smallest acceptable time headway
        for rl_id in self.rl_veh:
            lead_id = self.vehicles.get_leader(rl_id)
            if lead_id not in ["", None] \
                    and self.vehicles.get_speed(rl_id) > 0:
                t_headway = max(self.vehicles.get_headway(rl_id)
                                / self.vehicles.get_speed(rl_id), 0)
                cost2 += min((t_headway - t_min) / t_min, 0)

        # weights for cost1, cost2, and cost3, respectively
        eta1, eta2 = 1.00, 0.10

        return max(eta1*cost1 + eta2*cost2, 0)

    def sort_by_position(self):
        # vehicles are sorted by their get_x_by_id value
        sorted_ids = sorted(self.vehicles.get_ids(), key=self.get_x_by_id)
        return sorted_ids, None

    def additional_command(self):
        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.vehicles.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.vehicles.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.vehicles.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.vehicles.set_observed(veh_id)

    def reset(self):
        self.leader = []
        self.follower = []
        return super().reset()
