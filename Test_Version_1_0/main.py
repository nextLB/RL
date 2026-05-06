
import mujoco
import mujoco.viewer

# 加载仿真模型对象
m = mujoco.MjModel.from_xml_path('./car.xml')

# 创建了一个动态状态容器，用来存储仿真过程中每一时刻的“快照”
d = mujoco.MjData(m)




# 启动一个“被动”查看器窗口，并开始渲染模型 m 与数据 d
# 被动模式：仿真步进由你自己的代码控制，查看器只负责显示
# 使用 with 语句，确保退出时自动关闭窗口、清理图形资源
with mujoco.viewer.launch_passive(m, d) as viewer:

    # 只要查看器窗口没有被用户手动关闭，就保持循环
    while viewer.is_running():



        # 推进一个仿真步长：
        #   1. 根据当前状态计算所有物理量（力、加速度等）
        #   2. 更新 d 中的 qpos、qvel 等状态到下一时刻
        #   3. 自动将新一帧的画面数据发送給查看器进行渲染
        mujoco.mj_step(m, d)





