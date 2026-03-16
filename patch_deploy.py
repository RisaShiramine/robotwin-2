import sys
import os

target = "/mnt/hdd/Project/RoboTwin/policy/DP3/deploy_policy.py"
with open(target, 'r') as f:
    text = f.read()

new_encode = """def encode_obs(observation, TASK_ENV=None):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']
    try:
        obs['left_gripper'] = observation['joint_action']['left_gripper']
    except:
        pass
    if TASK_ENV is not None:
        try:
            obs['left_z'] = TASK_ENV.get_arm_pose("left")[2]
        except:
            obs['left_z'] = 0.0
    elif 'endpose' in observation and 'left_endpose' in observation['endpose']:
        try:
            obs['left_z'] = observation['endpose']['left_endpose'][2]
        except:
            obs['left_z'] = 0.0
    return obs

"""

text = text.replace("def encode_obs(observation):  # Post-Process Observation\n    obs = dict()\n    obs['agent_pos'] = observation['joint_action']['vector']\n    obs['point_cloud'] = observation['pointcloud']\n    return obs\n\n", new_encode)

text = text.replace("obs = encode_obs(observation)  # Post-Process Observation", "obs = encode_obs(observation, TASK_ENV)  # Post-Process Observation")

with open(target, 'w') as f:
    f.write(text)
print("Patch applied to deploy_policy.py")
