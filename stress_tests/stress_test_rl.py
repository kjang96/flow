"""
Repeatedly runs one step of an environment to test for possible race conditions
"""

import argparse
import json
import time
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.rllib import FlowParamsEncoder

# use this to specify the environment to run
from benchmarks.lanedrop0 import flow_params, env_name, create_env

# number of rollouts per training iteration
N_ROLLOUTS = 50
# number of parallel workers
PARALLEL_ROLLOUTS = 50

EXAMPLE_USAGE = """
example usage:
    python ./stress_test_rl.py PPO

Here the arguments are:
PPO - the name of the RL algorithm you want to use for the stress test
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Parses algorithm to run",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument("alg", type=str,
                    help="RL algorithm")

if __name__ == "__main__":
    args = parser.parse_args()
    alg = args.alg.upper()

    start = time.time()
    print("stress test starting")
    ray.init(redirect_output=False)
    flow_params["env"].horizon = 1
    horizon = flow_params["env"].horizon
    if alg == 'ARS':
        import ray.rllib.ars as ars
        config = ars.DEFAULT_CONFIG.copy()
        config["num_workers"] = PARALLEL_ROLLOUTS
        config["num_deltas"] = PARALLEL_ROLLOUTS
        config["deltas_used"] = PARALLEL_ROLLOUTS
    elif alg == 'PPO':
        import ray.rllib.ppo as ppo
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_workers"] = PARALLEL_ROLLOUTS
        config["timesteps_per_batch"] = horizon * N_ROLLOUTS
        config["vf_loss_coeff"] = 1.0
        config["kl_target"] = 0.02
        config["use_gae"] = True
        config["horizon"] = 1
        config["clip_param"] = 0.2
        config["num_sgd_iter"] = 1
        config["min_steps_per_task"] = 1
        config["sgd_batchsize"] = horizon * N_ROLLOUTS
    elif alg == 'ES':
        import ray.rllib.es as es
        config = es.DEFAULT_CONFIG.copy()
        config["num_workers"] = PARALLEL_ROLLOUTS
        config["episodes_per_batch"] = PARALLEL_ROLLOUTS
        config["timesteps_per_batch"] = PARALLEL_ROLLOUTS

    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                           indent=4)
    config['env_config']['flow_params'] = flow_json

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        "highway_stabilize": {
            "run": alg,  # Pulled from command line args
            "env": env_name,
            "config": {
                **config
            },
            "max_failures": 999,
            "stop": {"training_iteration": 50000},
            "repeat": 1,
            "trial_resources": {
                "cpu": 1,
                "gpu": 0,
                "extra_cpu": PARALLEL_ROLLOUTS - 1,
            },
        },
    })

    end = time.time()

    print("Stress test took " + str(end-start))