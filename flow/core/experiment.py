import logging
import datetime
import numpy as np

from flow.core.util import emission_to_csv


class SumoExperiment:

    def __init__(self, env, scenario):
        """
        This class acts as a runner for a scenario and environment.

        Attributes
        ----------
        env: Environment type
            the environment object the simulator will run
        scenario: Scenario type
            the scenario object the simulator will run
        """
        self.name = scenario.name
        self.num_vehicles = env.vehicles.num_vehicles
        self.env = env
        self.vehicles = scenario.vehicles
        self.cfg = scenario.cfg

        logging.info(" Starting experiment" + str(self.name) + " at "
                     + str(datetime.datetime.utcnow()))

        logging.info("initializing environment.")

    def run(self, num_runs, num_steps, rl_actions=None, convert_to_csv=False):
        """
        Runs the given scenario for a set number of runs and a set number of
        steps per run.

        Parameters
        ----------
        num_runs: int
            number of runs the experiment should perform
        num_steps: int
            number of steps to be performs in each run of the experiment
        rl_actions: list or numpy ndarray, optional
            actions to be performed by rl vehicles in the network (if there are
            any)
        convert_to_csv: bool
            Specifies whether to convert the emission file created by sumo into
            a csv file
        """
        if rl_actions is None:
            rl_actions = []

        rets = []
        for i in range(num_runs):
            logging.info("Iter #" + str(i))
            ret = 0
            self.env.reset()
            for j in range(num_steps):
                state, reward, done, _ = self.env.step(rl_actions)
                ret += reward
                if done:
                    break
            rets.append(ret)
            print("Round {0}, return: {1}".format(i, ret))

        print("Average, std return: {}, {}".format(np.mean(rets),
                                                   np.std(rets)))
        self.env.terminate()

        if convert_to_csv:
            # collect the location of the emission file
            dir_path = self.env.sumo_params.emission_path
            emission_filename = \
                "{0}-emission.xml".format(self.env.scenario.name)
            emission_path = \
                "{0}/{1}".format(dir_path, emission_filename)

            # convert the emission file into a csv
            emission_to_csv(emission_path)
