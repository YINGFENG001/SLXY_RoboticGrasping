# Project3 作业报告：PoseCNN 6D 姿态估计与抓取

## 1. Task 1：`pose_cnn.py` TODO 补全

Task 1 的目标是补全 PoseCNN，从 RGB 图像预测物体分割、中心深度和旋转姿态，最终生成 6D 位姿。

### 1.1 分割分支：TODO 1-5

**TODO 1：初始化分割分支前两层卷积**

将 VGG16 输出的两层特征都用 `1x1 Conv + ReLU` 降维：

```python
self.conv1 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
self.conv2 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
```

输入特征尺寸大致为：

```text
feature1: (B, 512, 60, 80)
feature2: (B, 512, 30, 40)
```

**TODO 2：初始化最终分割输出层**

分割输出为 `10` 个物体类别加 `1` 个背景类别：

```python
self.conv3 = nn.Conv2d(hidden_layer_dim, num_classes + 1, kernel_size=1)
```

**TODO 3：定义 Softmax**

在通道维度做 softmax，得到每个像素属于各类别的概率：

```python
self.softmax = nn.Softmax(dim=1)
```

**TODO 4：实现分割分支前向卷积和特征融合**

`feature1`、`feature2` 分别卷积后，将 `feature2` 上采样到 `feature1` 尺寸并相加：

```python
x1 = self.relu1(self.conv1(feature1))
x2 = self.relu2(self.conv2(feature2))
temp = interpolate(x2, size=x1.shape[-2:]) + x1
```

**TODO 5：上采样到原图尺寸**

将融合特征上采样到 `(480, 640)`，输出分割概率图和预测框：

```python
up_sample = interpolate(temp, size=(480, 640))
probability = self.softmax(self.conv3(up_sample))
segmentation = torch.argmax(probability, dim=1)
bbx = self.label2bbx(segmentation)
```

另外修正了 `label2bbx`：只遍历 `1~10` 类物体，不把背景 `0` 当作目标；没有检测框时返回空 tensor，避免后续 RoI 处理报错。

### 1.2 平移分支：TODO 6-8

**TODO 6：初始化平移分支卷积层**

平移分支也融合两层 VGG 特征。最终输出每类 `nx, ny, depth` 三个通道，因此输出通道是：

```text
3 * num_classes = 30
```

对应代码：

```python
self.conv1 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
self.conv2 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
self.conv3 = nn.Conv2d(hidden_layer_dim, 3 * num_classes, kernel_size=1)
```

**TODO 7：执行卷积和 ReLU**

```python
x1 = self.relu1(self.conv1(feature1))
x2 = self.relu2(self.conv2(feature2))
```

**TODO 8：融合特征并输出 centermap**

先把 `feature2` 对齐到 `feature1` 尺寸，再输出并上采样到原图大小：

```python
temp = interpolate(x2, size=x1.shape[-2:]) + x1
x3 = self.conv3(temp)
translation = interpolate(x3, size=(480, 640))
```

训练时用 `L1Loss` 约束预测 centermap 和真实 centermap。

### 1.3 旋转分支：TODO 9-13

**TODO 9：初始化 RoI Pooling**

ROI 框是原图坐标，而两层特征分别是原图的 `1/8` 和 `1/16`，所以设置：

```python
self.roi1 = RoIPool(output_size=(7, 7), spatial_scale=1.0 / 8.0)
self.roi2 = RoIPool(output_size=(7, 7), spatial_scale=1.0 / 16.0)
```

**TODO 10：初始化第一层全连接**

RoI Pooling 后特征尺寸为 `512 * 7 * 7`，映射到隐藏层：

```python
self.lin1 = nn.Linear(512 * 7 * 7, 4096)
```

**TODO 11：初始化四元数输出层**

每个类别预测一个四元数，所以输出维度为 `4 * num_classes = 40`：

```python
self.lin2 = nn.Linear(4096, 4 * num_classes)
```

**TODO 12：执行 RoI Pooling**

```python
out1 = self.roi1(feature1, bbx)
out2 = self.roi2(feature2, bbx)
out = out1 + out2
```

**TODO 13：预测旋转四元数**

```python
out = torch.flatten(out, start_dim=1)
out = self.relu_lin(self.lin1(out))
quaternion = self.lin2(out)
```

之后根据 ROI 的类别取对应四元数，归一化后转成旋转矩阵。

### 1.4 PoseCNN 总模型：TODO 14-22

**TODO 14：初始化四个模块**

```python
self.feature_extractor = FeatureExtraction(pretrained_backbone)
self.segmentation_branch = SegmentationBranch(num_classes=10)
self.RotationBranch = RotationBranch(num_classes=10)
self.TranslationBranch = TranslationBranch(num_classes=10)
```

**TODO 15：训练阶段提取图像特征**

```python
feat1, feat2 = self.feature_extractor(input_dict)
```

**TODO 16：训练阶段分割并计算分割 loss**

```python
probab, segmk, d_bbx = self.segmentation_branch(feat1, feat2)
loss_dict["loss_segmentation"] = loss_cross_entropy(probab, input_dict["label"])
```

同时补全平移和旋转损失：

```text
loss_centermap: L1Loss(trans, centermaps)
loss_R: loss_Rotation(pred_R, gt_R, label, models_pcd)
```

旋转损失只对 IoU 满足阈值的 ROI 计算：

```python
filter_bbx_R = IOUselection(d_bbx, gt_bbx, self.iou_threshold)
quater = self.RotationBranch(feat1, feat2, filter_bbx_R[:, :5])
```

**TODO 17：推理阶段提取特征**

```python
feat1, feat2 = self.feature_extractor(input_dict)
```

**TODO 18：推理阶段分割并预测 centermap**

```python
probab, segmentation, bb_xs = self.segmentation_branch(feat1, feat2)
trans_i = self.TranslationBranch(feat1, feat2)
```

**TODO 19：转换 bounding box 类型**

将 `bb_xs` 转为 float，供 RoI Pooling 使用：

```python
bb_xs = bb_xs.to(device=feat1.device, dtype=torch.float32)
```

**TODO 20：预测旋转四元数**

```python
quater = self.RotationBranch(feat1, feat2, bb_xs[:, :5])
```

**TODO 21：估计旋转矩阵**

```python
pred_R, _ = self.estimateRotation(quater, bb_xs)
```

**TODO 22：HoughVoting 估计中心和深度**

```python
pred_centers, pred_depths = HoughVoting(segmentation.cpu(), trans_i.cpu())
output_dict = self.generate_pose(pred_R.cpu(), pred_centers, pred_depths, bb_xs.cpu())
```

推理输出为物体的 `4x4` 位姿矩阵。代码中还修正了部分 CPU/GPU tensor 设备不一致问题。

Task 1 最终生成：

```text
posecnn_model.pth
```

完整 PoseCNN 训练 loss 从约 `0.729` 降到 `0.074`，5°5cm 指标准确率约为 `48.0%`。该指标要求旋转误差小于 5 度、平移误差小于 5 cm，比较严格，因此该结果说明模型可用但仍可能不稳定。

## 2. Task 2：`rrt_piper_grasp.py` TODO 补全

Task 2 使用 Task 1 训练好的 `posecnn_model.pth`，预测 cracker box 位姿，并通过 IK 和 RRT 完成抓取。

### 2.1 位姿转换：TODO 1-6

**TODO 1：GT 旋转矩阵转 Rotation 对象**

```python
rot = Rotation.from_matrix(R)
```

**TODO 2：提取 GT 欧拉角**

```python
euler_angles = rot.as_euler('xyz')
rz = euler_angles[2]
```

这里只保留 Z 轴旋转，用于桌面平面内的抓取验证。

**TODO 3：GT Z 轴旋转转 MuJoCo 四元数**

```python
rot_z_only = Rotation.from_euler('xyz', [0.0, 0.0, rz])
quat = rot_z_only.as_quat()
quat_mujoco = [quat[3], quat[0], quat[1], quat[2]]
```

`Rotation.as_quat()` 输出 `[x, y, z, w]`，MuJoCo 需要 `[w, x, y, z]`。

**TODO 4：预测旋转矩阵转 Rotation 对象**

```python
rot = Rotation.from_matrix(R)
```

**TODO 5：提取预测欧拉角**

```python
euler_angles = rot.as_euler('xyz')
```

**TODO 6：只保留预测 Z 轴旋转并转回旋转矩阵**

```python
rz = -euler_angles[2]
rot_z_only = Rotation.from_euler('xyz', [0.0, 0.0, rz])
R = rot_z_only.as_matrix()
```

这里取负号是为了匹配当前仿真坐标方向。Task 2 没有完整使用 6D 旋转，而是只使用 Z 轴旋转，让抓取更稳定。

### 2.2 PoseCNN 推理调用：TODO 7

**TODO 7：调用 `evaluate_and_save_samples`**

```python
sample_idx, pose_predict, gt_pose = evaluate_and_save_samples(
    model=posecnn_model,
    dataloader=dataloader,
    dataset=val_dataset,
    device=DEVICE,
    output_dir=output_dir,
    num_samples=1,
)
```

由于随机样本可能没有检测到 cracker box，代码增加了最多 20 次重试。如果预测结果中没有类别 `2`，就重新采样。

### 2.3 IK 与 RRT 抓取

得到 PoseCNN 预测位姿后，代码将目标位置和目标姿态交给 IK：

```text
目标位姿 -> inverse_kinematics -> 目标关节角
```

然后 RRT 在关节空间中从初始关节角规划到目标关节角，并用 MuJoCo 检查碰撞。找到路径后，机械臂执行：

```text
移动到目标上方 -> 下探 -> 闭合夹爪 -> 抬起
```

运行结果：

```text
Path found!
down path found
up path found
Video saved as rrt_posecnn_grasp.mp4
```

最终生成：

```text
rrt_posecnn_grasp.mp4
output/output_0.png
```

## 3. 对两个 Task 流程的理解

Task 1 可以理解为训练“眼睛”：输入 RGB 图像，PoseCNN 判断物体类别、分割区域、估计中心深度和旋转姿态，最终输出物体 6D 位姿。

Task 2 可以理解为用这双“眼睛”控制机械臂：PoseCNN 给出目标位姿，IK 将目标位姿转成关节角，RRT 规划无碰撞路径，最后机械臂按路径完成抓取。

关系如下：

```text
Task 1: 图像 -> 物体 6D 位姿
Task 2: 物体 6D 位姿 -> IK/RRT -> 机械臂抓取
```

## 4. 对视频结果的理解

最终视频 `rrt_posecnn_grasp.mp4` 中，机械臂根据 PoseCNN 预测的 cracker box 位姿移动到目标附近，然后下探、闭合夹爪并抬起物体。

视频能成功生成，说明 PoseCNN 成功预测到了目标物体，IK 求解出了目标关节角，RRT 找到了可行路径，MuJoCo 中的抓取动作也完整执行。

需要注意的是，Task 2 中只保留了 Z 轴旋转，没有完整使用全部 6D 姿态。这降低了抓取难度，也提高了演示稳定性。但如果 PoseCNN 的位置预测偏差较大，抓取仍可能失败。

## 5. 总结

本项目完成了从图像位姿估计到机械臂抓取的完整流程。Task 1 训练 PoseCNN 得到目标物体 6D 位姿，Task 2 使用该位姿作为机械臂目标，通过 IK 和 RRT 完成抓取规划与执行。最终模型能够生成可用的预测结果，并成功输出抓取演示视频。
