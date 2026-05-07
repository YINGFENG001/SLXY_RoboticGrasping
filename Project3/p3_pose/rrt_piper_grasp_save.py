# import cv2
from dm_control import mujoco
import cv2
import numpy as np
import random
import ikpy.chain
import ikpy.utils.plot as plot_utils
import transformations as tf
import PIL
from scipy.spatial.transform import Rotation

# Load dm_control model
model = mujoco.Physics.from_xml_path('assets/cracker_box.xml')

# Load the robot arm chain from the URDF file
my_chain = ikpy.chain.Chain.from_urdf_file("assets/piper_right.urdf")

class RRT:
    class Node:
        def __init__(self, q):
            self.q = q
            self.path_q = []
            self.parent = None

    def __init__(self, start, goal, joint_limits, expand_dis=0.1, path_resolution=0.01, goal_sample_rate=5, max_iter=1000):
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.joint_limits = joint_limits
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

    def planning(self, model):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, model):
                self.node_list.append(new_node)
            
            if self.calc_dist_to_goal(self.node_list[-1].q) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, model):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

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
        if extend_length > distance:
            extend_length = distance
        num_steps = int(extend_length / self.path_resolution)
        delta_q = (np.array(to_node.q[:6]) - np.array(from_node.q)) / distance

        for i in range(num_steps):
            new_q = new_node.q + delta_q * self.path_resolution
            new_node.q = np.clip(new_q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
            new_node.path_q.append(new_node.q)

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_q = [random.uniform(joint_min, joint_max) for joint_min, joint_max in self.joint_limits]
        else:
            rand_q = self.end.q
        return self.Node(rand_q)

    def check_collision(self, node, model):
        return check_collision_with_dm_control(model, node.q)

    def generate_final_course(self, goal_ind):
        path = [self.end.q]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.q)
            node = node.parent
        path.append(self.start.q)
        return path[::-1]
    
    def calc_dist_to_goal(self, q):
        return np.linalg.norm(np.array(self.end.q[:6]) - np.array(q))

def get_depth(sim):
    # depth is a float array, in meters.
    depth = sim.render(camera_id=0, height=480, width=640,depth=True)
    # Shift nearest values to the origin.
    depth -= depth.min()
    # Scale by 2 mean distances of near rays.
    depth /= 2*depth[depth <= 1].mean()
    # Scale to [0, 255]
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
    model.data.qpos[0:6] = joint_config  # Set joint positions
    model.forward()  # Update the simulation state

    # Check for collisions
    contacts = model.data.ncon  # Number of contacts (collisions)
    # contacts=0
    return contacts == 0 or check_gripper_collision(model) # True if no contacts (collision-free)

def check_gripper_collision(model):
    all_contact_pairs = []
    for i_contact in range(model.data.ncon):
        id_geom_1 = model.data.contact[i_contact].geom1
        id_geom_2 = model.data.contact[i_contact].geom2
        name_geom_1 = model.model.id2name(id_geom_1, 'geom')
        name_geom_2 = model.model.id2name(id_geom_2, 'geom')
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)
    touch_banana_right = ("piper_gripper_finger_touch_right", "cracker_box") in all_contact_pairs
    touch_banana_left = ("piper_gripper_finger_touch_left", "cracker_box") in all_contact_pairs
    return touch_banana_left or touch_banana_right

def get_end_effector_pose(physics, body_name="link6"):
    """获取末端执行器位姿（位置和旋转矩阵）"""
    physics.forward()  # 必须先更新物理状态
    
    # 获取body ID
    body_id = physics.model.name2id(body_name, 'body')
    if body_id == -1:
        raise ValueError(f"未找到body: {body_name}")
    
    # 位置 (x,y,z)
    pos = physics.named.data.xpos[body_name]  # 使用named接口
    
    # 旋转矩阵 (3x3)
    rot = physics.named.data.xmat[body_name].reshape(3, 3)
    
    return pos.copy(), rot.copy()  # 返回拷贝避免后续修改
    
def apply_rrt_path_to_dm_control(model, path, video_name="rrt_robot_motion_1.mp4"):
    """
    Function to apply the RRT-generated path (list of joint configurations) to the dm_control simulation,
    while recording the frames into a video.
    
    Args:
        model: dm_control Mujoco model
        path: List of joint configurations generated by the RRT planner
        video_name: Name of the output video file
    """
    # Setup for video recording
    width, height = 640, 480  # Resolution of each camera
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (1280, 480))  # Two 640x480 images side by side

    # set initial joint angles
    model.data.qpos[0:6] = start
    model.forward()

    # Apply the path to the simulation and record the video
    for q in path:
        # model.data.qpos[:] = q  # Set joint angles
        model.data.ctrl[0:6] = q[0:6]  # Set joint angles
        
        # Render from both cameras and concatenate side by side
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame_bgr)

        # Step the simulation forward to the next state
        model.step()

    # 这里为了稳定悬空
    for i in range(100):
        # Render from both cameras and concatenate side by side
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame_bgr)
        model.step()

    start_joints_down=model.data.qpos[0:6]
    target_position_down = target_position

    # 设置最终抓取位置
    target_position_down[2] = 0.15
    target_orientation_euler_down = target_orientation_euler
    target_orientation_down = tf.euler_matrix(*target_orientation_euler_down)[:3, :3]
    joint_angles_down = my_chain.inverse_kinematics(target_position_down, target_orientation_down, "all")
    joint_angles_down = joint_angles_down[1:7]
    print("joint_angles_down:",joint_angles_down * 57.2958)
    # 生成插值因子，num 表示插值的数量，比如 10 表示插值 10 次
    num_interpolations = 50
    t_values = np.linspace(0, 1, num=num_interpolations)
    # 生成角度路径
    interpolated_lists_down = np.array([(1-t)*start_joints_down + t*joint_angles_down for t in t_values])
    if interpolated_lists_down.size > 0:
        print("down path found")
        # Apply the path to the simulation and record the video

        for q in interpolated_lists_down:
            model.data.ctrl[0:6] = q[0:6]
            
            # Render from both cameras and concatenate side by side
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            # Convert frame from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            # Write the frame to the video
            out.write(frame_bgr)

            # Step the simulation forward to the next state
            model.step()
    
    # #现在已经找到了路径并渲染了 目前需要关闭夹爪 并渲染
  
    for i in range(30):
        # Render from both cameras and concatenate side by side
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        if i >= 20 :
            close_gripper()
        # Write the frame to the video
        out.write(frame_bgr)
        model.step()      

    start_joints_up = model.data.qpos[0:6]
    target_position_up = target_position
    target_position_up[2] = 0.5

    target_orientation_euler_up = target_orientation_euler
    target_orientation_up = tf.euler_matrix(*target_orientation_euler_up)[:3, :3]
    joint_angles_up = my_chain.inverse_kinematics(target_position_up, target_orientation_up, "all")

    joint_angles_up = joint_angles_up[1:7]
    print("joint_angles_up:",joint_angles_up * 57.2958)

    interpolated_lists_up = np.array([(1-t)*start_joints_up + t*joint_angles_up for t in t_values])
    if interpolated_lists_up.size > 0:
        print("up path found")
        # Apply the path to the simulation and record the video
        for q in interpolated_lists_up:
            # Check joint limits

            # model.data.qpos[:] = q  # Set joint angles
            model.data.ctrl[0:6] = q  # Set joint angles
            
            # Render from both cameras and concatenate side by side
            frame_1 = model.render(camera_id=0, width=width, height=height)
            frame_2 = model.render(camera_id=1, width=width, height=height)
            frame_combined = np.concatenate((frame_1, frame_2), axis=1)
            
            # Convert frame from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
            
            # Write the frame to the video
            out.write(frame_bgr)

            # Step the simulation forward to the next state
            model.step()

    for i in range(50):
        # Render from both cameras and concatenate side by side
        frame_1 = model.render(camera_id=0, width=width, height=height)
        frame_2 = model.render(camera_id=1, width=width, height=height)
        frame_combined = np.concatenate((frame_1, frame_2), axis=1)
        
        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_combined, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame_bgr)
        model.step()

    # Release the video writer
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

# 初始化物体的真实位姿
def pose_gt_init(physics, body_name="cracker_box", joint_name = 'cracker_box_joint'):
    """获取末端执行器位姿（位置和旋转矩阵）"""
    physics.forward()  # 必须先更新物理状态
    
    # 获取body ID
    body_id = physics.model.name2id(body_name, 'body')
    if body_id == -1:
        raise ValueError(f"未找到body: {body_name}")

    T = cracker_gt_pose[:3,3]
    R = cracker_gt_pose[:3,:3]


    print("真实目标位置1:", T)

    T[0] = -T[0] - 0.1
    T[1] = T[1] + 0.3
    T[2] = 0.1

    print("真实目标位置:", T)

    model.named.data.qpos[joint_name][0:3] = T[0:3]

    # 转换为 Rotation 对象
    rot = Rotation.from_matrix(R)  # 如果是旋转矩阵
    # 提取欧拉角（XYZ 顺序）
    euler_angles = rot.as_euler('xyz')  # [rx, ry, rz] 弧度制

    # 仅保留 z 轴旋转，
    rx = 0.0
    ry = 0.0
    rz = euler_angles[2]
    print("真实Z轴旋转角度", rz * 57.2958)     
    # 重新构造仅 X 轴旋转的四元数
    rot_z_only = Rotation.from_euler('xyz', [rx, ry, rz])
    
    quat = rot_z_only.as_quat()
    quat_mujoco = [quat[3], quat[0], quat[1], quat[2]]  # MuJoCo 格式: [w, x, y, z]
    model.named.data.qpos[joint_name][3:7] = quat_mujoco  

    return T, R

# 初始化物体的预测位姿
def pose_predict_init(physics, body_name="cracker_box", joint_name = 'cracker_box_joint'):
    """获取末端执行器位姿（位置和旋转矩阵）"""
    physics.forward()  # 必须先更新物理状态
    
    # 获取body ID
    body_id = physics.model.name2id(body_name, 'body')
    if body_id == -1:
        raise ValueError(f"未找到body: {body_name}")

    T = cracker_pose_predict[:3,3]
    R = cracker_pose_predict[:3,:3]

    print("预测目标位置1:", T)

    temp = T[0]
    T[0] = T[1] + 0.3
    T[1] = temp + 0.1
    T[2] = 0.4

    print("预测目标位置:", T)

    # 转换为 Rotation 对象
    rot = Rotation.from_matrix(R)  # 如果是旋转矩阵
    # 提取欧拉角（XYZ 顺序）
    euler_angles = rot.as_euler('xyz')  # [rx, ry, rz] 弧度制 

    # 仅保留 Z 轴旋转
    rx = 0.0
    ry = 0.0
    rz = -euler_angles[2]
    print("预测Z轴旋转角度", -rz * 57.2958) 
    # 重新构造仅 z 轴旋转的四元数(这里是x轴旋转)
    rot_z_only = Rotation.from_euler('xyz', [rx,ry,rz])

    R = rot_z_only.as_matrix()
    print("R",R)
    return T, R


start = [0.0/57.2958, 110/57.2958, -80/57.2958, 0/57.2958, 65/57.2958, 45/57.2958]   # Start joint angles
reflect_flag = np.array([[1., 0, 0.],[0., 1., 0.],[0., 0, -1.]])
target_orientation_init = np.array([[0., -1, 0.],[-1., -0., 0.],[0., 0, 1.]])


######################################################################
##### 以下是利用训练出来的PoseCNN模型，对随机抽取一张测试集图像进行识别推理#####
import os
import torch
from torch.utils.data import DataLoader
import torchvision.models as models

import utils
from pose_cnn import PoseCNN, eval
from PROPSPoseDataset import PROPSPoseDataset
import matplotlib
matplotlib.use('Agg')  # 必须在其他 matplotlib 导入前设置
import matplotlib.pyplot as plt
import random
import numpy as np

BATCH_SIZE = 4
DRIVE_PATH = '.'

DATA_PATH = "./data"
train_dataset = PROPSPoseDataset(
    DATA_PATH, "train",
    download=False  # =True (for the first time)
) 
val_dataset = PROPSPoseDataset(DATA_PATH, "val")
dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

utils.reset_seed(0)
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# 定义设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

posecnn_model = PoseCNN(pretrained_backbone = vgg16, 
                models_pcd = torch.tensor(val_dataset.models_pcd).to(DEVICE, dtype=torch.float32),
                cam_intrinsic = val_dataset.cam_intrinsic).to(DEVICE)
posecnn_model.load_state_dict(torch.load(os.path.join(DRIVE_PATH, "posecnn_model.pth"),weights_only=True))

# 创建 output 文件夹（如果不存在）
output_dir = os.path.join(os.getcwd(), "output")  # 获取当前路径并拼接 output
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹已存在则不会报错

def pose_eval(model, dataloader, device, alpha = 0.35):
    import cv2
    model.eval()

    sample_idx = random.randint(0,len(dataloader.dataset)-1)
    ## image version vis
    rgb = torch.tensor(dataloader.dataset[sample_idx]['rgb'][None, :]).to(device)
    inputdict = {'rgb': rgb}
    pose_predict, label = model(inputdict)

    rgb =  (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    vis_result = dataloader.dataset.visualizer.vis_oneview(
        ipt_im = rgb, 
        obj_pose_dict = pose_predict[0],
        alpha = alpha
        )
    
    return vis_result, pose_predict,sample_idx

def evaluate_and_save_samples(model = posecnn_model, 
                              dataloader = dataloader, 
                              dataset = val_dataset, 
                              device = DEVICE, 
                              output_dir = output_dir, 
                              num_samples=1):
    results = []
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估指定数量的样本
    sample_indices = []  # 用于记录选取的样本索引
    for i in range(num_samples):
        # 获取预测结果
        vis_result, pose_predict, sample_idx = pose_eval(model, dataloader, device)
        sample_indices.append(sample_idx)
        
        # 保存可视化结果
        plt.axis('off')
        plt.imshow(vis_result)
        plt.savefig(os.path.join(output_dir, f"output_{i}.png"))
        plt.close()
        
        # 获取真实位姿
        sample = dataset[sample_idx]
        gt_pose_dict = sample['RTs']
        
        # 存储结果
        result = {
            'sample_idx': sample_idx,
            'predicted_poses': pose_predict,
            'ground_truth_poses': gt_pose_dict
        }
        results.append(result)
    
    return sample_idx,pose_predict,gt_pose_dict


# PoseCNN获取位姿
sample_idx,pose_predict,gt_pose = evaluate_and_save_samples()

##### 以上是利用训练出来的PoseCNN模型，对随机抽取一张测试集图像进行识别推理#####
######################################################################

print('采样图片id:', sample_idx)
# print("PoseCNN预测位姿:", pose_predict)
# print("Ground Truth位姿:", gt_pose)

cracker_pose_predict = pose_predict[0][2]
cracker_gt_pose = gt_pose[1]

# 初始化物体真实位姿与预测物资
gt_postion, gt_orientation = pose_gt_init(model)
predict_position, predict_orientation = pose_predict_init(model)

target_position = predict_position
target_orientation = target_orientation_init @ predict_orientation  @ reflect_flag
# target_orientation = target_orientation_init @ reflect_flag

print("target_orientation" , target_orientation)
rot = Rotation.from_matrix(target_orientation)
target_orientation_euler = rot.as_euler('xyz')
print("euler_angles = " , target_orientation_euler)

# IK
joint_angles = my_chain.inverse_kinematics(target_position, target_orientation, "all")

# goal and joint limits
goal = joint_angles[1:7]
print("goal",goal*57.2958)
#joint limits
joint_limits = [[-2.618,2.618],[0,3.14158],[-2.697,0],[-1.832,1.832],[-1.22,1.22],[-3.14158,3.14158]] 

# ----------------- 新增代码 -----------------
# 设置初始关节角度
model.data.qpos[:6] = start
model.forward()

# 初始化RRT
rrt = RRT(start, goal, joint_limits)
rrt_path = rrt.planning(model)  # Generate the RRT path

# Apply the path to the MuJoCo simulation and record video
if rrt_path:
    print("Path found!")

    # 打开夹爪
    open_gripper()
    # Apply RRT Path 
    apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_posecnn_grasp.mp4")
else:
    print("No path found!")