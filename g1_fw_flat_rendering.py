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
from brax.training import networks as brax_networks

# other import
import os
import functools
import imageio
import numpy as np
import collections


# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def main():
    env_name = 'G1JoystickFlatTerrain'
    env = registry.load(env_name, config_overrides={"impl": "jax"})
    
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    
 
    ppo_networks_instance = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=jax.nn.standardize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_networks_instance)



    checkpoint_dir = os.path.abspath('./g1_walking_param_A100_cloud')
    mngr = ocp.CheckpointManager(checkpoint_dir, ocp.PyTreeCheckpointer())
    
    step = mngr.latest_step()
    if step is None:
        print("No checkpoint found.")
        return
        
    restored_params = mngr.restore(step)

    norm_dict = restored_params[0]
    if isinstance(norm_dict, dict):
        RunningStatisticsState = collections.namedtuple(
            'RunningStatisticsState', ['count', 'mean', 'summed_variance', 'std']
        )
        
        norm_state = RunningStatisticsState(
            count=norm_dict['count'],
            mean=norm_dict['mean'],
            summed_variance=norm_dict['summed_variance'],
            std=norm_dict['std']
        )

        restored_params = (norm_state, restored_params[1], restored_params[2])

    inference_fn = make_inference_fn(restored_params)
    jit_inference_fn = jax.jit(inference_fn)


    fps = 60
    duration = 10 # seconds
    total_steps = fps * duration
    
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    

    renderer = mujoco.Renderer(mj_model, height=1080, width=1920)
    
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    frames = []
    print(f"Rendering {total_steps} steps of the G1 robot...")

    for _ in range(total_steps):
        rng, subkey = jax.random.split(rng)
        

        act_rng, subkey = jax.random.split(subkey)
        action, _ = jit_inference_fn(state.obs, act_rng)
    
        state = env.step(state, action)
      
        mujoco.mj_step(mj_model, mj_data)
        mj_data.qpos = np.array(state.data.qpos)
        mj_data.qvel = np.array(state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        
        renderer.update_scene(mj_data, camera="track") # track movement
        pixels = renderer.render()
        frames.append(pixels)

  
    output_path = "g1_walking_demo.mp4"
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"video save to {output_path}")

if __name__ == "__main__":
    main()
