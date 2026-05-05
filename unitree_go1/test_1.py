import mujoco
import mujoco.viewer
import numpy as np
import time

# ===================== 1. 加载模型 =====================
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# ===================== 2. 初始化状态 =====================
# 加载关键帧 "home"（如果有），否则使用默认 qpos
try:
    key_id = model.key_name2id("home")
    data.qpos[:] = model.key_qpos[key_id]
    data.ctrl[:] = model.key_ctrl[key_id]   # 初始控制信号
except:
    # 没有 keyframe 时手动设置躯干高度
    data.qpos[2] = 0.445   # 躯干初始高度（来自 go1.xml 中的 pos="0 0 0.445"）
    data.qpos[3:7] = [1, 0, 0, 0]   # 四元数

mujoco.mj_forward(model, data)

# ===================== 3. 步态参数 =====================
# Go1 关节顺序（根据 actuator 定义）：
# FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
# RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
#
# 我们设计一个简单的正弦波步态：
# - 髋关节（hip）负责前后摆动，幅度约 0.5 rad，频率 2 Hz
# - 膝关节（thigh）和踝关节（calf）配合，但为了简单，我们只驱动髋关节和膝关节，
#   让腿做类似“划水”的运动。实际中需要更精细的轨迹，但这里演示基本运动。
#
# 注意：关节范围在 go1.xml 中定义：
#   abduction (hip_yaw): ±0.863 rad
#   hip (hip_pitch): -0.686 ~ 4.501 rad
#   knee: -2.818 ~ -0.888 rad
# 我们仅使用 hip 和 knee 关节，abduction 保持 0。
#
# 索引（0-based）：
#   0: FR_hip (abduction)  → 保持 0
#   1: FR_thigh (hip)      → 正弦波
#   2: FR_calf (knee)      → 正弦波（相位滞后）
#   3: FL_hip (abduction)  → 0
#   4: FL_thigh            → 正弦波（与 FR 反相）
#   5: FL_calf             → 正弦波（反相）
#   6: RR_hip (abduction)  → 0
#   7: RR_thigh            → 正弦波（与 FR 同相）
#   8: RR_calf             → 正弦波（同相）
#   9: RL_hip (abduction)  → 0
#   10: RL_thigh           → 正弦波（与 FL 同相）
#   11: RL_calf            → 正弦波（同相）

# 定义髋关节（thigh）的正弦波参数
hip_amplitude = 0.6      # 弧度，范围 [-0.686, 4.501] 内安全
hip_frequency = 2.0      # Hz

# 膝关节（calf）的波（通常与髋关节有相位差，模拟行走）
knee_amplitude = 0.8
knee_frequency = 2.0
knee_phase_offset = 0.5  # 弧度（相对于髋关节的滞后）

# 左右腿相位差（对角步态）
left_right_phase = np.pi   # 左右腿反相

# ===================== 4. 运行仿真 =====================
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    dt = model.opt.timestep
    frame_skip = 5        # 每次决策跳过的仿真步数（与 gym 中的 frame_skip 类似）
    step_duration = dt * frame_skip   # 实际时间步长（秒）

    # 可选：让仿真持续运行直到用户关闭窗口
    while viewer.is_running():
        t = time.time() - start_time

        # ----- 计算当前目标关节角度 -----
        # 髋关节目标（thigh）：正弦波
        # 前腿与后腿同相，左右腿反相
        for leg_idx, (hip_joint, knee_joint, phase_factor) in enumerate([
            (1, 2,  0),      # FR: 相位 0
            (4, 5,  left_right_phase),   # FL: 加 π
            (7, 8,  0),      # RR: 同 FR
            (10, 11, left_right_phase)   # RL: 同 FL
        ]):
            # 髋关节目标
            hip_target = hip_amplitude * np.sin(2 * np.pi * hip_frequency * t + phase_factor)
            # 膝关节目标（滞后 hip_target）
            knee_target = knee_amplitude * np.sin(2 * np.pi * knee_frequency * t + phase_factor + knee_phase_offset)
            # 限制在关节范围内（安全起见）
            hip_target = np.clip(hip_target, -0.6, 0.6)   # 实际范围 [-0.686,4.501]，但这里我们只取小幅度
            knee_target = np.clip(knee_target, -1.2, -0.9)  # 膝关节范围 [-2.818, -0.888]，取典型弯曲位置

            # 设置执行器控制信号（位置控制）
            data.ctrl[hip_joint] = hip_target
            data.ctrl[knee_joint] = knee_target

        # 外展关节保持 0
        for abd_joint in [0, 3, 6, 9]:
            data.ctrl[abd_joint] = 0.0

        # ----- 执行仿真步 -----
        mujoco.mj_step(model, data, nstep=frame_skip)

        # ----- 更新可视化 -----
        viewer.sync()

        # 控制仿真速度（近似实时）
        time_until_next = start_time + (t + step_duration) - time.time()
        if time_until_next > 0:
            time.sleep(time_until_next)

        # 可选：打印前进速度
        if int(t * 10) % 10 == 0:   # 每 1 秒打印一次
            forward_vel = data.qvel[0]   # 躯干线速度 X 方向
            print(f"Time: {t:.2f}s, Forward velocity: {forward_vel:.2f} m/s")

    print("Simulation ended.")
