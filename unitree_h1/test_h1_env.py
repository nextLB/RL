import gymnasium as gym
import time

# 注意：H1 有 20 个关节，而 Humanoid-v5 默认动作空间是 17 维
# 因此 H1 的后 3 个关节将不会被控制（保持 0 力矩）
env = gym.make(
    "Humanoid-v5",
    xml_file="./scene.xml",   # 你的场景文件，它包含 h1.xml
    forward_reward_weight=1,
    ctrl_cost_weight=0.1,
    healthy_reward=1,
    healthy_z_range=(0.8, 2.0),   # 根据 H1 站立高度调整
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.0,
    frame_skip=5,
    max_episode_steps=5000,
    render_mode="human"
)

obs, info = env.reset()
for _ in range(10000):
    # 随机动作，让机器人乱动（可能摔倒）
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.02)
    if terminated or truncated:
        obs, info = env.reset()
env.close()



