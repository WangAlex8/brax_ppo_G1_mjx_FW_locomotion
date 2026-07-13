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
from brax.training.acme import running_statistics

# other imports
import os
import functools
import imageio
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def main():
    env_name = 'G1JoystickFlatTerrain'
    
    raw_env = registry.load(env_name, config_overrides={
        "impl": "jax",
        "lin_vel_x": [0.5, 0.5],       
        "lin_vel_y": [0.0, 0.0],       
        "ang_vel_yaw": [0.0, 0.0],    
    })
    
    env = wrapper.wrap_for_brax_training(raw_env)
    
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    
    ppo_networks_instance = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize
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
        norm_state = running_statistics.RunningStatisticsState(
            count=norm_dict['count'],
            mean=norm_dict['mean'],
            std=norm_dict['std'],
            summed_variance=norm_dict['summed_variance']
        )
        restored_params = (norm_state, restored_params[1], restored_params[2])


    inference_fn = make_inference_fn(restored_params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    fps = 60
    duration = 10 # seconds
    total_steps = fps * duration
    
    mj_model = raw_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    renderer = mujoco.Renderer(mj_model, height=1080, width=1920)
    
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    
   
    try:
        cam.trackbodyid = mj_model.body('pelvis').id
    except Exception:
        try:
            cam.trackbodyid = mj_model.body('torso').id
        except Exception:
            cam.trackbodyid = 1  
            
    cam.distance = 2.0      
    cam.azimuth = 60.0      
    cam.elevation = -15.0   
    

    rng = jax.random.PRNGKey(42)
    rng_batched = jnp.stack([rng])
    state = env.reset(rng_batched)
    
    frames = []
    
    print(f"Simulating {total_steps} steps on the A100 GPU...")
    
    jit_step = jax.jit(env.step)
    
    trajectory_qpos = []
    trajectory_qvel = []
    
  
    for _ in range(total_steps):
        rng, subkey = jax.random.split(rng)
        act_rng, subkey = jax.random.split(subkey)
        
        action, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, action)
        

        if hasattr(state, 'pipeline_state'):
            trajectory_qpos.append(state.pipeline_state.qpos[0])
            trajectory_qvel.append(state.pipeline_state.qvel[0])
        else:
            trajectory_qpos.append(state.data.qpos[0])
            trajectory_qvel.append(state.data.qvel[0])

  
    print("Transferring trajectory to CPU...")
    trajectory_qpos = jax.device_get(trajectory_qpos)
    trajectory_qvel = jax.device_get(trajectory_qvel)

    print("Rendering frames to video...")
    for i in range(total_steps):

        mj_data.qpos = trajectory_qpos[i]
        mj_data.qvel = trajectory_qvel[i]
        
        # kinematics pass
        mujoco.mj_forward(mj_model, mj_data)
        
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render())

    output_path = "g1_walking_demo.mp4"
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"video saved to {output_path}")

if __name__ == "__main__":
    main()