# starcraft_projects/resource_management_agent.py
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import numpy as np
import random
import math

# Define action IDs
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

# Define feature IDs
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Constants
PLAYER_SELF = 1
TERRAN_COMMAND_CENTER = 18
TERRAN_SCV = 45
SUPPLY_DEPOT = 19
BARRACKS = 21

class ResourceManagementAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ResourceManagementAgent, self).__init__()
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def simulated_annealing(self, current_score, new_score):
        if new_score > current_score:
            return True
        else:
            probability = math.exp((new_score - current_score) / self.temperature)
            self.temperature *= self.cooling_rate
            return random.random() < probability

    def step(self, obs):
        super(ResourceManagementAgent, self).step(obs)

        if _TRAIN_SCV in obs.observation['available_actions']:
            return actions.FunctionCall(_TRAIN_SCV, [0])

        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        # Select SCV if not already selected
        if _SELECT_POINT in obs.observation['available_actions']:
            scv_y, scv_x = (unit_type == TERRAN_SCV).nonzero()
            if len(scv_y) > 0:
                index = 0
                return actions.FunctionCall(_SELECT_POINT, [[0], [scv_x[index], scv_y[index]]])

        # Command SCV to harvest minerals if possible
        if _HARVEST_GATHER in obs.observation['available_actions']:
            mineral_y, mineral_x = (unit_type == 1680).nonzero()
            if len(mineral_y) > 0:
                index = 0
                return actions.FunctionCall(_HARVEST_GATHER, [[0], [mineral_x[index], mineral_y[index]]])

        # Build supply depot if needed
        if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
            free_supply = obs.observation['player'][2] - obs.observation['player'][3]
            if free_supply < 5:
                depot_y, depot_x = (unit_type == 0).nonzero()
                if len(depot_y) > 0:
                    index = 0
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [[0], [depot_x[index], depot_y[index]]])

        # Build barracks if supply is enough
        if _BUILD_BARRACKS in obs.observation['available_actions']:
            if obs.observation['player'][1] >= 150:
                barracks_y, barracks_x = (unit_type == 0).nonzero()
                if len(barracks_y) > 0:
                    index = 0
                    return actions.FunctionCall(_BUILD_BARRACKS, [[0], [barracks_x[index], barracks_y[index]]])

        return actions.FunctionCall(_NO_OP, [])

# Run the agent
def main():
    agent = ResourceManagementAgent()
    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=True) as env:

            agent.setup(env.observation_spec(), env.action_spec())

            timesteps = env.reset()
            agent.reset()

            while True:
                step_actions = [agent.step(timesteps[0])]
                if timesteps[0].last():
                    break
                timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

# starcraft_projects/mini_game_rl_agent.py
from pysc2.agents import base_agent
from pysc2.lib import actions
import numpy as np
import random
import math

class MiniGameRLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(MiniGameRLAgent, self).__init__()
        self.q_table = np.zeros([100, 100, len(actions.FUNCTIONS)])
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 0.1
        self.temperature = 1.0
        self.cooling_rate = 0.995

    def simulated_annealing(self, current_score, new_score):
        if new_score > current_score:
            return True
        else:
            probability = math.exp((new_score - current_score) / self.temperature)
            self.temperature *= self.cooling_rate
            return random.random() < probability

    def step(self, obs):
        super(MiniGameRLAgent, self).step(obs)
        state = (random.randint(0, 99), random.randint(0, 99))

        if np.random.rand() < self.exploration_rate:
            action = random.choice(range(len(actions.FUNCTIONS)))
        else:
            action = np.argmax(self.q_table[state])

        reward = obs.reward
        next_state = (random.randint(0, 99), random.randint(0, 99))
        best_next_action = np.argmax(self.q_table[next_state])

        if self.simulated_annealing(self.q_table[state][action], reward):
            self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (
                reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action]
            )

        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

# implemented Q-learning tweaks