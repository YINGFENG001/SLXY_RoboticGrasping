from dm_control import mujoco
import cv2
import numpy as np
import random
import ikpy.chain
import ikpy.utils.plot as plot_utils
import transformations as tf
import PIL
import sys


# 加载 dm_control 的物理模型（MuJoCo XML 文件）
model = mujoco.Physics.from_xml_path('assets/chef_can.xml')

# 从 URDF 文件加载机械臂的运动学链（用于逆运动学求解）
my_chain = ikpy.chain.Chain.from_urdf_file("assets/piper_right.urdf")

DENSE_PATH_MAX_STEP = 0.03
SMOOTHING_ITERATIONS = 120
    
class RRTConnect:
    class Node:
        def __init__(self, q):
            self.q = np.array(q[:6])
            self.path_q = []
            self.parent = None

    def __init__(self, start, goal, joint_limits, expand_dis=0.1, path_resolution=0.01, max_iter=5000):
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.joint_limits = joint_limits
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_iter = max_iter

    def planning(self, model):
        tree_start = [self.start]
        tree_goal = [self.end]
        active_tree_from_start = True

        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            new_node = self.extend(tree_start, rnd_node, model)

            if new_node is not None:
                connect_node = self.connect(tree_goal, new_node, model)
                if connect_node is not None:
                    return self.generate_connected_path(
                        new_node,
                        connect_node,
                        active_tree_from_start,
                    )

            tree_start, tree_goal = tree_goal, tree_start
            active_tree_from_start = not active_tree_from_start

        return None

    def extend(self, tree, target_node, model):
        nearest_ind = self.get_nearest_node_index(tree, target_node)
        nearest_node = tree[nearest_ind]
        if self.calc_distance(nearest_node.q, target_node.q) < 1e-9:
            return None

        new_node = self.steer(nearest_node, target_node, self.expand_dis)
        if not self.check_collision(new_node, model):
            return None

        tree.append(new_node)
        return new_node

    def connect(self, tree, target_node, model):
        while True:
            new_node = self.extend(tree, target_node, model)
            if new_node is None:
                return None

            if self.calc_distance(new_node.q, target_node.q) <= self.path_resolution:
                return new_node

    def get_nearest_node_index(self, node_list, rnd_node):
        """
        Find the index of the nearest node to the random node.
        
        Args:
            node_list: List of nodes in the RRT tree.
            rnd_node: Randomly generated node.
        
        Returns:
            Index of the nearest node in the node list.
        """
        dlist = [np.linalg.norm(np.array(node.q) - np.array(rnd_node.q[:6])) for node in node_list]
        min_index = dlist.index(min(dlist))
        return min_index
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(np.array(from_node.q))
        distance = np.linalg.norm(np.array(to_node.q[:6]) - np.array(from_node.q))
        if distance < 1e-9:
            new_node.parent = from_node
            return new_node
        if extend_length > distance:
            extend_length = distance
        num_steps = max(int(np.ceil(extend_length / self.path_resolution)), 1)
        delta_q = (np.array(to_node.q[:6]) - np.array(from_node.q)) / distance

        for i in range(1, num_steps + 1):
            new_q = np.array(from_node.q) + delta_q * extend_length * (i / num_steps)
            new_node.q = np.clip(new_q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
            new_node.path_q.append(new_node.q.copy())

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        rand_q = [random.uniform(joint_min, joint_max) for joint_min, joint_max in self.joint_limits]
        return self.Node(rand_q)

    def check_collision(self, node, model):
        for q in node.path_q:
            if not check_collision_with_dm_control(model, q):
                return False
        return check_collision_with_dm_control(model, node.q)

    def trace_path(self, node):
        path = []
        while node is not None:
            path.append(node.q)
            node = node.parent
        return path

    def generate_connected_path(self, active_node, connected_node, active_tree_from_start):
        active_path = self.trace_path(active_node)
        connected_path = self.trace_path(connected_node)

        if active_tree_from_start:
            return active_path[::-1] + connected_path[1:]
        return connected_path[::-1] + active_path[1:]

    def calc_distance(self, q_from, q_to):
        return np.linalg.norm(np.array(q_to[:6]) - np.array(q_from[:6]))

def get_depth(sim):
    # 获取深度图（单位为米），返回的是一个浮点数组
    depth = sim.render(camera_id=0, height=480, width=640,depth=True)
    # 将最近的深度值平移到原点（最小值变为0）
    depth -= depth.min()
    # 以近距离（小于等于1米）的平均值作为归一化因子进行缩放
    depth /= 2*depth[depth <= 1].mean()
    # 将深度值限制在 [0, 1] 范围并映射到 [0, 255]，用于图像显示
    pixels = 255*np.clip(depth, 0, 1)
    image=PIL.Image.fromarray(pixels.astype(np.uint16))
    return image

def check_collision_with_dm_control(model, joint_config):
    """
    Function to check if a given joint configuration results in a collision using dm_control's collision detection.
    Args:
        model: dm_control Mujoco model
        joint_config: List of joint angles to check for collision
    Returns:
        True if collision-free, False if there is a collision
    """
    model.data.qpos[0:6] = joint_config 
    # 设置机械臂前6个关节的位置（角度）
    model.forward()  
    # 推进仿真状态，使模型更新到当前配置

    # 当前仿真中接触（碰撞）点的数量
    contacts = model.data.ncon  
    # 如果没有发生碰撞，或者只是夹爪与物体接触，则认为是安全的（可执行动作）
    return contacts == 0 or check_gripper_collision(model) # True 表示无碰撞或夹爪正常接触

def check_gripper_collision(model):
    all_contact_pairs = []
    for i_contact in range(model.data.ncon):
        id_geom_1 = model.data.contact[i_contact].geom1
        id_geom_2 = model.data.contact[i_contact].geom2
        name_geom_1 = model.model.id2name(id_geom_1, 'geom')
        name_geom_2 = model.model.id2name(id_geom_2, 'geom')
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)
    touch_chef_can_right = ("piper_gripper_finger_touch_right", "chef_can_collision") in all_contact_pairs
    touch_chef_can_left = ("piper_gripper_finger_touch_left", "chef_can_collision") in all_contact_pairs
    return touch_chef_can_left or touch_chef_can_right

def is_edge_collision_free(q_from, q_to, model, resolution=0.02):
    q_from = np.array(q_from[:6])
    q_to = np.array(q_to[:6])
    distance = np.linalg.norm(q_to - q_from)
    steps = max(int(np.ceil(distance / resolution)), 1)

    for t in np.linspace(0.0, 1.0, steps + 1):
        q = (1.0 - t) * q_from + t * q_to
        if not check_collision_with_dm_control(model, q):
            return False
    return True

def smooth_path(path, model, iterations=SMOOTHING_ITERATIONS):
    if path is None or len(path) <= 2:
        return path

    smoothed = [np.array(q[:6]) for q in path]
    for _ in range(iterations):
        if len(smoothed) <= 2:
            break
        i, j = sorted(random.sample(range(len(smoothed)), 2))
        if j <= i + 1:
            continue
        if is_edge_collision_free(smoothed[i], smoothed[j], model):
            smoothed = smoothed[:i + 1] + smoothed[j:]
    return smoothed

def densify_path(path, max_step=DENSE_PATH_MAX_STEP):
    if path is None or len(path) <= 1:
        return path

    dense_path = [np.array(path[0][:6])]
    for q_from, q_to in zip(path[:-1], path[1:]):
        q_from = np.array(q_from[:6])
        q_to = np.array(q_to[:6])
        distance = np.linalg.norm(q_to - q_from)
        steps = max(int(np.ceil(distance / max_step)), 1)
        for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
            dense_path.append((1.0 - t) * q_from + t * q_to)
    return dense_path

def get_end_effector_pose(physics, body_name="link6"):
    """获取末端执行器位姿（位置和旋转矩阵）"""
    physics.forward()  # 必须先更新物理状态
    
    # 获取body ID
    body_id = physics.model.name2id(body_name, 'body')
    if body_id == -1:
        raise ValueError(f"未找到body: {body_name}")
    
    # 位置 (x,y,z)
    pos = physics.named.data.xpos[body_name]  
    # 使用named接口
    
    # 旋转矩阵 (3x3)
    rot = physics.named.data.xmat[body_name].reshape(3, 3)
    
    return pos.copy(), rot.copy()  
    # 返回拷贝避免后续修改
    
def apply_rrt_path_to_dm_control(model, path, video_name="rrt_robot_motion_1.mp4"):
    """
    Function to apply the RRT-generated path (list of joint configurations) to the dm_control simulation,
    while recording the frames into a video.
    
    Args:
        model: dm_control Mujoco model
        path: List of joint configurations generated by the RRT planner
        video_name: Name of the output video file
    """
    # 初始化视频录制设置
    width, height = 640, 480  #每个摄像头的分辨率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码格式
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (1280, 480))  # 创建视频文件，画面为两个摄像头并排拼接

    # 设置起始关节角（初始姿态）
    model.data.qpos[0:6] = start
    model.forward()

    # 执行路径并录制视频
    for q in path:
        # model.data.qpos[:] = q  # （可选）直接设置关节角
        model.data.ctrl[0:6] = q[0:6]  # 设置控制命令以改变前6个关节的角度
        
        # 分别从两个摄像头渲染图像并拼接在一起
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # 将 RGB 图像转换为 BGR 格式以适配 OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # 写入当前帧到视频文件
        out.write(frame_bgr)

        # 推进仿真一步，执行控制命令
        model.step()

    # 这里为了稳定悬空
    for i in range(100):
        # 从两个摄像头渲染图像，并拼接成一张画面
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # 将 RGB 格式转换为 BGR，适配 OpenCV 写入视频
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # 将当前帧写入视频文件
        out.write(frame_bgr)
        # 推进仿真一步（保持当前位置）
        model.step()

    start_joints_down = model.data.qpos[0:6].copy()
    target_position_down = target_position.copy()

    #TODO 1.在这里你需要填入一个数字设置夹爪末端在 Z 轴方向的下降高度，使其准确接近物体进行抓取。通常应略低于目标物体中心，取值范围在 0 到 1 米之间。
    target_position_down[2] = 0.10

    target_orientation_euler_down = target_orientation_euler
    target_orientation_down = tf.euler_matrix(*target_orientation_euler_down)[:3, :3]
    joint_angles_down = my_chain.inverse_kinematics(target_position_down, target_orientation_down, "all")
    joint_angles_down = joint_angles_down[1:7]
    print("joint_angles_down:",joint_angles_down * 57.29)
    # 生成插值因子，num 表示插值的数量，比如 10 表示插值 10 次
    num_interpolations = 30
    t_values = np.linspace(0, 1, num=num_interpolations)

    #TODO 2.在这里你需要生成一段从当前位置到抓取位置的关节角插值轨迹，用于驱动机械臂平稳下移。  
    interpolated_lists_down = np.array([
        (1.0 - t) * start_joints_down + t * joint_angles_down
        for t in t_values
    ])
    # 仿真中不涉及真实时间控制，因此只需设置插值数量，例如使用 30 步插值，生成等间距角度序列。生成关节角度轨迹，基于起始和结束位置，仿真条件下不需要考虑时间

    if interpolated_lists_down.size > 0:
        print("down path found")
        #  如果插值轨迹非空，则执行下移动作并将过程录制为视频
        for q in interpolated_lists_down:
            #TODO 3.在这里你需要填写一个变量，用于指定当前插值步骤中的关节角配置，作为控制指令输入。  
            # 它将驱动机械臂执行从当前位置向目标位置的下移动作。
            model.data.ctrl[0:6] = q
            
            # 从两个摄像头渲染画面，并将图像拼接成一帧
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            # 将图像从 RGB 格式转换为 BGR 格式，适配 OpenCV
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            # 将当前帧写入视频文件
            out.write(frame_bgr)

            # 推进仿真一步，更新机械臂状态
            model.step()
    
    # #现在已经找到了路径并渲染了 目前需要关闭夹爪 并渲染
  
    for i in range(50):
        # 从两个摄像头渲染图像，并拼接成一帧画面
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # 将图像从 RGB 转换为 BGR 格式，适用于 OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        # 在后 10 帧内闭合夹爪，实现抓取动作
        if i >= 40 :
            close_gripper()
        # 将当前帧写入视频
        out.write(frame_bgr)
        # 推进仿真一步，更新场景状态
        model.step()      


    #TODO 4.在这里你需要从当前机械臂状态中读取上抬的起始关节角；该值应来源于抓取动作完成后的关节位置。  
    # 随后设置机械臂 Z 轴上抬高一定距离（0-1），以实现抓取完成后的抬手动作。
    start_joints_up = model.data.qpos[0:6].copy()
    target_position_up = target_position.copy()
    target_position_up[2] = 0.25

    target_orientation_euler_up = target_orientation_euler
    target_orientation_up = tf.euler_matrix(*target_orientation_euler_up)[:3, :3]
    joint_angles_up = my_chain.inverse_kinematics(target_position_up, target_orientation_up, "all")
    joint_angles_up = joint_angles_up[1:7]
    print("joint_angles_up:",joint_angles_up * 57.29)

    #TODO 5.在这里你需要生成一个多维数组，表示从当前姿态到上抬目标姿态的关节角插值序列。
    interpolated_lists_up = np.array([
        (1.0 - t) * start_joints_up + t * joint_angles_up
        for t in t_values
    ])

    if interpolated_lists_up.size > 0:
        print("up path found")  
        # 将路径应用到仿真中并录制视频
        for q in interpolated_lists_up:
            # 检查关节角是否超出限制

            # 设置关节角度（可选：直接设置 qpos）
            model.data.ctrl[0:6] = q  # 设置控制输入的关节角度
            
            # 从两个摄像头渲染图像，并拼接为一帧
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            
            # 将 RGB 图像转换为 BGR 格式，适配 OpenCV
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            
            # 将当前帧写入视频
            out.write(frame_bgr)

            # 推进仿真一步，更新状态
            model.step()      

    for i in range(50):
        # 从两个摄像头渲染图像并拼接成一帧
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # 将图像从 RGB 转换为 BGR 格式以适配 OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # 将当前帧写入视频
        out.write(frame_bgr)
        model.step()

    # 释放视频写入器资源
    out.release()
   
    print(f"Video saved as {video_name}")

def close_gripper():
    # 夹爪闭合控制
    model.data.ctrl[6]=0.0
    model.data.ctrl[7]=0.0

def open_gripper():
    # 夹爪打开控制
    model.data.ctrl[6]=0.035
    model.data.ctrl[7]=-0.035



start = [145.0/57.2958, 90/57.2958, -80/57.2958, 0/57.2958, 65/57.2958, 45/57.2958]   # Start joint angles

#TODO 6.在这里你需要填写一个三维坐标，用于指定机械臂抬起时在世界坐标系中的位置（x,y,z），目标的坐标为 x=0.15, y=0.25,请根据任务需要设定z 可稍后用于下降调整。
target_position = np.array([0.15, 0.25, 0.25])

#TODO 7.在这里你需要填写欧拉角，用于定义目标的朝向，目标的旋转欧拉角为 绕x轴旋转180度, 绕y轴旋转0度, 绕z轴旋转-90度，输入弧度制
# 随后需将欧拉角转换为旋转矩阵，作为逆运动学的目标姿态输入。
target_orientation_euler = np.array([np.pi, 0.0, -np.pi / 2.0])
target_orientation = tf.euler_matrix(*target_orientation_euler)[:3, :3]

# IK逆运动学求解，获取目标位姿对应的关节角
joint_angles = my_chain.inverse_kinematics(target_position, target_orientation, "all")

# 提取目标关节角（去除固定基座关节）并作为路径规划终点
goal=joint_angles[1:7]
print("goal",goal*57.2958)
# 设置每个关节的运动范围（单位：弧度）
joint_limits = [[-2.618,2.618],[0,3.14158],[-2.697,0],[-1.832,1.832],[-1.22,1.22],[-3.14158,3.14158]] 

# 设置初始关节角度
model.data.qpos[:6] = start
model.forward()

# 获取初始末端位姿
init_pos, init_rot = get_end_effector_pose(model)
print("\n初始末端位置:", np.round(init_pos, 4))
print("初始旋转矩阵:\n", np.round(init_rot, 4))

# 可选：转换为欧拉角（ZYX顺序，角度制）
from scipy.spatial.transform import Rotation as R
euler_angles = R.from_matrix(init_rot).as_euler('xyz', degrees=True)
print("初始欧拉角(度):", np.round(euler_angles, 2))

#TODO 8.在这里你需要初始化 RRT 路径规划器(使用定义好的类)，传入起点、终点与关节限制，并生成一条抓取路径。
# 请确保已正确设置 RRT 类与碰撞检测逻辑，最终返回一条可行路径 rrt_path。
rrt = RRTConnect(start, goal, joint_limits, expand_dis=0.15, path_resolution=0.02, max_iter=4000)
rrt_path = rrt.planning(model)

# 将生成的路径应用于 MuJoCo 仿真，并录制视频
if rrt_path:
    print("Path found!")
    rrt_path = smooth_path(rrt_path, model)
    rrt_path = densify_path(rrt_path)
    print(f"Smoothed and densified path length: {len(rrt_path)}")

    #TODO 9.在这里你需要填写一行代码，调用前面定义的函数，打开夹爪准备抓取。 
    open_gripper()

    # 执行路径并生成视频文件 
    apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_robot_motion_rrtconnect.mp4")
else:
    print("No path found!")
