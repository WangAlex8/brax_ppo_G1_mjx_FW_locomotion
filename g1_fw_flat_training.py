# jax imports
import jax
import jax.numpy as jnp

# mujoco imports
import mujoco
import mujoco_playground
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
from mujoco_playground import wrapper

# brax ppo imports
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo

# other import
import os
import functools
import numpy as np

# gpu stuff
jax.config.update('jax_platforms', 'cuda')
print("JAX DEFINITIVE BACKEND:", jax.devices()) # when it prints, ensure its not using cpu

def patch_device_put_replicated(x, devices):
    devices_arr = np.array(devices)
    mesh = jax.sharding.Mesh(devices_arr, ('devices',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('devices',))
    return jax.tree_util.tree_map(
        lambda leaf: jax.device_put(jnp.stack([leaf] * len(devices), axis=0), sharding), 
        x
    )
jax.device_put_replicated = patch_device_put_replicated


class humanoid_sim():
    def __init__ (self, env_name: str = "G1JoystickFlatTerrain"):
        self.env = registry.load(env_name, config_overrides={
            "impl": "jax",
            # desired motion config override
            "lin_vel_x": [1.0, 1.0],
            "lin_vel_y": [0.0, 0.0],
            "ang_vel_yaw": [0.0, 0.0],
            # task reward weights
            'reward_config.scales.tracking_lin_vel' : 2.5, # task reward for desired forward velocity
            'reward_config.scales.alive' : 0.7, # constant positive value given for every step where robot doesn't fall
            # stability penalities
            'reward_config.scales.orientation': -0.8, # penalizes deivations from an upright torso
            'reward_config.scales.ang_vel_xy': -0.9, # resists pitch and roll
            'reward_config.scales.lin_vel_z': -2.0, # no bouncing
            # Smoothness
            'reward_config.scales.torques': -0.00001, #penality square of joint torques; penalizes jittery movements
            'reward_config.scales.contact_force': -0.001, # impact penality, 惩罚 high ground forces to reduce stomping (faster mechanical wear)
            'reward_config.scales.action_rate': -0.01,   # penalize jerk between steps
            # rewards for lifting legs (to get it to actually walk and not cheat the forward velocity rewards)
            'reward_config.scales.feet_air_time': 1.0,
            'reward_config.scales.feet_slip': -0.15,
            'reward_config.scales.feet_clearance': 0.1,
            'reward_config.scales.feet_phase': 0.5

        })

        self.mj_model = self.env.mj_model # loads the MuJoCo model from environment
        self.mjx_model = self.env.mjx_model # loads the MuJoCo model to GPU
        self.data = mujoco.MjData(self.mj_model) 
        
        self.rng = jax.random.PRNGKey(0) # starting point for all the random numbers the simulation might need
        self.state = self.env.reset(self.rng) # creates the first snapshot of the world
    
    @property  # this turns a function into a read-only attribute, cannot be modified after it is initially initialized
    # because of this, there is no need to call a method like g1.mjx_data() every time, u can just write g1.mjx_data
    def mjx_data(self):
        # in JAX, self.state changes every time the robot moves. since it is a property, every time i say g1.mjx_data,
        # python will pull the current & active object self.state & pulls out the live GPU physics matrix with qpos, qvel, forces etc
        return self.state.data

    def step(self, action):
        # in JAX, random keys will give the same data if used more than once, so they must be split
        self.rng, subkey = jax.random.split(self.rng)
        # this line takes current key and forks it into two new unique keys
        # self.rng changes and a new subkey is generated
        self.state = self.env.step(self.state, action)
        return self.state
    

def progress(num_steps, metrics):
    reward = metrics.get('eval/episode_reward', 0)
    print(metrics.keys())


def main():
    g1 = humanoid_sim()
    eval_g1 = humanoid_sim()
    
    env_name = 'G1JoystickFlatTerrain'
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_training_params = dict(ppo_params)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    
    if "network_factory" in ppo_training_params:
        del ppo_training_params["network_factory"]

    ppo_training_params.update({
        'num_timesteps': 100000000,
        'num_envs': 4096,           
        'episode_length': 1000,
        'learning_rate': 0.0003,    
        'entropy_cost': 0.008,
        'unroll_length': 16,      
        'batch_size': 16384,        
        'num_minibatches': 8,       
        'normalize_observations': True,
        'reward_scaling': 0.1,
        'num_evals': 10,            
    })

    training_function = ppo.train(
        environment = g1.env,
        eval_env = eval_g1.env,
        wrap_env_fn = wrapper.wrap_for_brax_training,
        network_factory = network_factory, 
        progress_fn = progress,
        **ppo_training_params
    )

    make_inference_fn, params, metrics = training_function
    
    checkpoint_dir = os.path.abspath('./g1_walking_param_A100_cloud')

    options = ocp.CheckpointManagerOptions(max_to_keep = 3, create=True)
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        ocp.PyTreeCheckpointer(),
        options = options
    )

    mngr.save(step = 100000000, args = ocp.args.PyTreeSave(item=params))
    mngr.wait_until_finished()

    print(f"Parameters successfully saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()

    

