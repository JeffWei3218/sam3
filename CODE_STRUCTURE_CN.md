# SAM3 代码结构总结（简化版）

这是 SAM3 代码结构的简化中文总结。完整版本请参考 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。

## 快速导航

- [训练流程](#训练流程简述)
- [推理流程](#推理流程简述)
- [关键文件](#关键文件说明)

---

## 训练流程简述

### 启动训练

```bash
# 本地训练
python sam3/train/train.py -c configs/your_config.yaml --use-cluster 0

# 集群训练
python sam3/train/train.py -c configs/your_config.yaml --use-cluster 1
```

### 训练步骤

1. **初始化阶段**
   - 加载配置文件（Hydra）
   - 设置分布式环境（DDP）
   - 初始化模型和优化器

2. **数据加载**
   - 从COCO格式JSON加载标注
   - 应用数据增强（调整大小、翻转、归一化）
   - 生成多模态提示（文本/点/框/掩码）

3. **训练循环**
   ```
   for epoch in epochs:
       for batch in dataloader:
           # 前向传播
           outputs = model(batch)
           
           # 计算损失
           loss = criterion(outputs, targets)
           
           # 反向传播
           loss.backward()
           
           # 更新权重
           optimizer.step()
   ```

4. **保存检查点**
   - 定期保存模型权重
   - 保存优化器状态
   - 保存训练指标

---

## 推理流程简述

### 图像推理

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1. 构建模型
model = build_sam3_image_model()
processor = Sam3Processor(model)

# 2. 设置图像
from PIL import Image
image = Image.open("image.jpg")
inference_state = processor.set_image(image)

# 3. 添加文本提示
output = processor.set_text_prompt(
    state=inference_state,
    prompt="a dog"
)

# 4. 获取结果
masks = output["masks"]      # 分割掩码
boxes = output["boxes"]      # 边界框
scores = output["scores"]    # 置信度分数
```

### 视频推理

```python
from sam3.model_builder import build_sam3_video_predictor

# 1. 构建预测器
predictor = build_sam3_video_predictor()

# 2. 开始会话
response = predictor.handle_request({
    "type": "start_session",
    "resource_path": "video.mp4"
})
session_id = response["session_id"]

# 3. 添加提示
response = predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "a person"
})

# 4. 传播到整个视频
for result in predictor.handle_stream_request({
    "type": "propagate_in_video",
    "session_id": session_id
}):
    frame_idx = result["frame_idx"]
    masks = result["masks"]
    # 处理每帧结果...
```

---

## 关键文件说明

### 模型相关

| 文件 | 作用 |
|------|------|
| `sam3/model/sam3_image.py` | 图像分割模型主类 |
| `sam3/model/sam3_video_predictor.py` | 视频预测器接口 |
| `sam3/model/vitdet.py` | Vision Transformer 骨干网络 |
| `sam3/model/text_encoder_ve.py` | 文本编码器 |
| `sam3/model/encoder.py` | Transformer 编码器 |
| `sam3/model/decoder.py` | Transformer 解码器 |
| `sam3/model_builder.py` | 模型构建工具函数 |

### 训练相关

| 文件 | 作用 |
|------|------|
| `sam3/train/train.py` | 训练启动脚本 |
| `sam3/train/trainer.py` | 训练循环实现 |
| `sam3/train/data/sam3_image_dataset.py` | 图像数据集 |
| `sam3/train/data/sam3_video_dataset.py` | 视频数据集 |
| `sam3/train/loss/sam3_loss.py` | 损失函数 |
| `sam3/train/matcher.py` | 匈牙利匹配器 |

### 配置文件

| 目录 | 说明 |
|------|------|
| `sam3/train/configs/odinw13/` | ODinW13数据集配置 |
| `sam3/train/configs/roboflow_v100/` | Roboflow100数据集配置 |

---

## 模型架构概览

```
输入（图像/视频帧 + 提示）
    ↓
┌─────────────────────────────┐
│   共享骨干 (Shared Backbone) │
│                              │
│  • Vision Transformer (ViT) │
│  • 文本编码器 (Text Encoder) │
└──────────────┬──────────────┘
               ↓
        ┌──────┴──────┐
        ↓             ↓
┌────────────┐  ┌────────────┐
│  检测器    │  │  跟踪器    │
│ (Detector) │  │ (Tracker)  │
│            │  │            │
│ • 编码器   │  │ • 记忆编码 │
│ • 解码器   │  │ • 时序处理 │
│ • 分割头   │  │ • 掩码传播 │
└─────┬──────┘  └─────┬──────┘
      ↓               ↓
    图像输出        视频输出
  (masks, boxes)  (tracking masks)
```

### 核心组件尺寸

- **输入图像**: 1008 × 1008
- **ViT Patch**: 14 × 14
- **ViT 特征维度**: 1024
- **Transformer 维度**: 256
- **对象查询数量**: 200
- **编码器层数**: 6
- **解码器层数**: 6
- **记忆帧数**: 7（跟踪器）

---

## 数据流

### 训练数据流

```
数据集 (COCO/ODinW/Roboflow)
    ↓
加载标注 (JSON)
    ↓
数据增强
    ↓
生成提示
    ↓
批次整理 (Collator)
    ↓
模型输入
```

### 推理数据流

```
原始图像/视频
    ↓
调整大小 (1008×1008)
    ↓
归一化
    ↓
提取特征
    ↓
融合提示
    ↓
生成预测
    ↓
后处理 (NMS等)
    ↓
输出结果
```

---

## 损失函数

SAM3 使用多任务损失：

1. **掩码损失** (Mask Loss)
   - Focal Loss: 处理类别不平衡
   - Dice Loss: 优化IoU

2. **边界框损失** (Box Loss)
   - L1 Loss: 坐标回归
   - GIoU Loss: 几何约束

3. **分类损失** (Classification Loss)
   - Focal Loss: 二分类（目标/背景）

总损失 = α₁·Mask + α₂·Box + α₃·Class

---

## 性能优化建议

### 训练优化
1. 使用混合精度训练 (AMP)
2. 启用梯度累积（大批次）
3. 使用激活检查点（节省内存）
4. 多GPU/多节点训练（DDP）

### 推理优化
1. 批处理推理
2. 模型编译 (torch.compile)
3. TF32 加速（Ampere GPU）
4. 异步帧加载（视频）

---

## 常见配置

### 训练配置示例

```yaml
trainer:
  mode: train
  max_epochs: 12
  batch_size: 2
  
  optim:
    lr: 0.0001
    weight_decay: 0.05
    amp: true
    
launcher:
  num_nodes: 1
  gpus_per_node: 8
```

### 推理配置示例

```python
model = build_sam3_image_model(
    device="cuda",
    eval_mode=True,
    enable_segmentation=True,
    compile=False
)
```

---

## 常见问题

### Q: 如何修改输入图像大小？
A: 修改 `img_size` 参数（默认1008），但需要重新训练。

### Q: 如何减少内存使用？
A: 
- 减小批次大小
- 启用激活检查点
- 使用混合精度训练

### Q: 如何提高推理速度？
A:
- 启用模型编译
- 使用批处理推理
- 调整NMS阈值

### Q: 支持哪些数据格式？
A:
- 图像: COCO JSON格式
- 视频: MP4或JPEG序列

---

## 更多资源

- **完整文档**: [CODE_STRUCTURE.md](CODE_STRUCTURE.md)
- **训练指南**: [README_TRAIN.md](README_TRAIN.md)
- **项目主页**: [README.md](README.md)
- **示例代码**: [examples/](examples/)
- **GitHub**: https://github.com/facebookresearch/sam3

---

**最后更新**: 2025-11-20
