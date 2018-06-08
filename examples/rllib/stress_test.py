from flow.controllers import IDMController, ContinuousRouter
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario, \
    ADDITIONAL_NET_PARAMS
from flow.envs.base_env import Env
import ray


@ray.remote
def start():
	sumo_params = SumoParams(sim_step=0.1, sumo_binary="sumo")

	sumo_params.sumo_binary = 'sumo'

	vehicles = Vehicles()
	vehicles.add(veh_id="idm",
				 acceleration_controller=(IDMController, {}),
				 routing_controller=(ContinuousRouter, {}),
				 num_vehicles=22)

	env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

	additional_net_params = ADDITIONAL_NET_PARAMS.copy()
	net_params = NetParams(additional_params=additional_net_params)

	initial_config = InitialConfig(bunching=20)

	scenario = LoopScenario(name="sugiyama",
							generator_class=CircleGenerator,
							vehicles=vehicles,
							net_params=net_params,
							initial_config=initial_config)

	env = AccelEnv(env_params, sumo_params, scenario)
	env.start_sumo()

ray.init()
results = ray.get([start.remote() for i in range(10000)])