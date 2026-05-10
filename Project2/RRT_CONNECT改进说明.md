# RRT-Connect 改进说明

对比基线：`rrt_piper_original.py`

当前实现：`rrt_piper_rrtconnect.py`

## 核心改进

1. 普通 RRT 改为 RRT-Connect

- 原版：只从起点生长一棵树。
- 新版：起点树和目标树同时生长，扩展成功后让另一棵树连续连接。
- 对应代码：`class RRTConnect`、`planning()`、`extend()`、`connect()`、`generate_connected_path()`。

2. 目标关节角改为 IK 求解

- 原版：直接写死 `goal` 关节角。
- 新版：给定目标位置和姿态，通过 `ikpy` 反解目标关节角。
- 对应代码：`target_position`、`target_orientation_euler`、`my_chain.inverse_kinematics()`。

3. 碰撞检测覆盖整段路径

- 原版：只检查扩展终点。
- 新版：遍历 `node.path_q`，检查扩展过程中的所有插值点。
- 对应代码：`check_collision()`。

4. 路径后处理减少摇摆

- 新增 `smooth_path()` 删除可直连的多余折返点。
- 新增 `densify_path()` 加密轨迹点，降低控制跳变。
- 对应代码：`SMOOTHING_ITERATIONS=120`、`DENSE_PATH_MAX_STEP=0.03`。

5. 补全抓取动作

- 原版：只执行规划路径。
- 新版：打开夹爪、移动到物体上方、下降、闭合夹爪、上抬。
- 对应代码：`open_gripper()`、`close_gripper()`、下降/上抬插值段。

## RRT-Connect 修改点

1. 双树结构

```python
tree_start = [self.start]
tree_goal = [self.end]
```

每轮扩展后交换两棵树，让搜索从起点和目标两端交替推进。

2. 单步扩展

```python
def extend(self, tree, target_node, model):
```

从当前树找到最近节点，调用 `steer()` 扩展一步；无碰撞才加入树。

3. 连续连接

```python
def connect(self, tree, target_node, model):
```

另一棵树不断向新节点扩展；距离小于 `path_resolution` 时连接成功。

4. 路径拼接

```python
def trace_path(self, node):
def generate_connected_path(self, active_node, connected_node, active_tree_from_start):
```

从两棵树的连接节点回溯父节点，拼接成 `start -> goal` 完整路径。

5. `steer()` 稳定性修正

```python
if distance < 1e-9:
    new_node.parent = from_node
    return new_node
num_steps = max(int(np.ceil(extend_length / self.path_resolution)), 1)
new_node.path_q.append(new_node.q.copy())
```

- 避免除零。
- 避免 `num_steps=0`。
- 避免路径点数组引用被后续修改污染。

6. 主程序调用

```python
rrt = RRTConnect(start, goal, joint_limits, expand_dis=0.15, path_resolution=0.02, max_iter=4000)
rrt_path = rrt.planning(model)
```

RRT-Connect 不再使用 `goal_sample_rate`。

## 其他代码修改点

1. 场景模型

```python
model = mujoco.Physics.from_xml_path('assets/chef_can.xml')
```

原版是 `assets/piper_rrt.xml`，新版切到带目标物体的抓取场景。

2. IK 运动学链

```python
my_chain = ikpy.chain.Chain.from_urdf_file("assets/piper_right.urdf")
```

用于把目标位姿转换为机械臂 6 关节目标。

3. 起始关节角

```python
start = [145.0/57.2958, 90/57.2958, -80/57.2958, 0/57.2958, 65/57.2958, 45/57.2958]
```

新版起点面向抓取任务，不再使用 original 的演示起点。

4. 目标位姿

```python
target_position = np.array([0.15, 0.25, 0.25])
target_orientation_euler = np.array([np.pi, 0.0, -np.pi / 2.0])
```

先规划到物体上方，再下降抓取。

5. 控制维度

```python
model.data.qpos[0:6] = ...
model.data.ctrl[0:6] = ...
```

前 6 维控制机械臂，`ctrl[6]`、`ctrl[7]` 控制夹爪。

6. 视频输出

```python
apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_robot_motion_rrtconnect.mp4")
```

避免覆盖普通 RRT 输出。

## TODO 对应实现

1. TODO 1：下降高度

```python
target_position_down[2] = 0.10
```

2. TODO 2：下降插值轨迹

```python
interpolated_lists_down = np.array([
    (1.0 - t) * start_joints_down + t * joint_angles_down
    for t in t_values
])
```

3. TODO 3：执行下降控制

```python
model.data.ctrl[0:6] = q
```

4. TODO 4：上抬起点和目标

```python
start_joints_up = model.data.qpos[0:6].copy()
target_position_up = target_position.copy()
target_position_up[2] = 0.25
```

5. TODO 5：上抬插值轨迹

```python
interpolated_lists_up = np.array([
    (1.0 - t) * start_joints_up + t * joint_angles_up
    for t in t_values
])
```

6. TODO 6：目标位置

```python
target_position = np.array([0.15, 0.25, 0.25])
```

7. TODO 7：目标姿态

```python
target_orientation_euler = np.array([np.pi, 0.0, -np.pi / 2.0])
target_orientation = tf.euler_matrix(*target_orientation_euler)[:3, :3]
```

8. TODO 8：初始化 RRT-Connect

```python
rrt = RRTConnect(start, goal, joint_limits, expand_dis=0.15, path_resolution=0.02, max_iter=4000)
rrt_path = rrt.planning(model)
```

9. TODO 9：打开夹爪

```python
open_gripper()
```

## 执行流程

```python
rrt_path = rrt.planning(model)
rrt_path = smooth_path(rrt_path, model)
rrt_path = densify_path(rrt_path)
open_gripper()
apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_robot_motion_rrtconnect.mp4")
```
