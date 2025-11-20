# SAM 3 批量推理指南 (Batch Inference Guide)

## 概述 (Overview)

SAM 3 支持高效的批量推理，允许在单次前向传播中处理多个图像和查询。批量推理通过减少模型加载和初始化开销，显著提高了处理大量图像时的吞吐量。

SAM 3 supports efficient batch inference, enabling the processing of multiple images and queries in a single forward pass. Batch inference significantly improves throughput when processing large numbers of images by reducing model loading and initialization overhead.

## 主要特性 (Key Features)

1. **多图像批处理** - 在单个批次中处理多个图像
2. **多提示支持** - 每个图像可以包含多个文本或视觉提示
3. **灵活的后处理** - 可配置的置信度阈值和检测数量限制
4. **高效的内存使用** - 优化的数据整理和设备传输

1. **Multi-image batching** - Process multiple images in a single batch
2. **Multiple prompt support** - Each image can contain multiple text or visual prompts
3. **Flexible post-processing** - Configurable confidence thresholds and detection limits
4. **Efficient memory usage** - Optimized data collation and device transfer

## 工作流程 (Workflow)

### 1. 环境设置 (Environment Setup)

```python
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import Datapoint

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use bfloat16 for mixed precision (if supported)
torch.set_default_dtype(torch.bfloat16)
```

### 2. 模型加载 (Model Loading)

```python
# Load the SAM 3 model
model = build_sam3_image_model(bpe_path="path/to/bpe_simple_vocab_16e6.txt.gz")
```

### 3. 配置转换和后处理器 (Configure Transforms and Post-processor)

```python
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
)
from sam3.eval.postprocessors import PostProcessImage

# Setup transforms (resize to 1008x1008 and normalize)
transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Setup post-processor
postprocessor = PostProcessImage(
    max_dets_per_img=-1,      # -1 for no limit, or set a positive number for top-k
    iou_type="segm",          # "segm" for masks, "bbox" for boxes only
    use_original_size=True,   # Return results in original image size
    conf_threshold=0.3,       # Confidence threshold for filtering results
)
```

### 4. 创建数据点 (Create Datapoints)

每个数据点代表一个图像及其相关的查询提示：

Each datapoint represents an image and its associated query prompts:

```python
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage

# Create a datapoint for each image
datapoint = Datapoint(
    find_metadatas=[],  # Will store query metadata
    queries=[],         # Will store query data
    image=None,         # Will store the image
)

# Add image to datapoint
def set_image(datapoint, pil_image):
    """Set the image for a datapoint."""
    datapoint.image = SAMImage(image=pil_image, dataset_name="custom")
    return datapoint

# Add text prompt
def add_text_prompt(datapoint, text, query_id):
    """Add a text prompt to the datapoint."""
    metadata = InferenceMetadata(
        dataset_name="custom",
        img_id="img",
        query_id=query_id,
        is_negative=False,  # Set to True for negative prompts
    )
    query = FindQueryLoaded(
        text=text,
        boxes=None,
        points=None,
    )
    datapoint.find_metadatas.append(metadata)
    datapoint.queries.append(query)
    return query_id

# Add visual prompt (boxes or points)
def add_visual_prompt(datapoint, boxes=None, points=None, query_id=None):
    """Add a visual prompt (boxes or points) to the datapoint."""
    metadata = InferenceMetadata(
        dataset_name="custom",
        img_id="img",
        query_id=query_id,
        is_negative=False,
    )
    query = FindQueryLoaded(
        text=None,
        boxes=boxes,  # Format: [[x1, y1, x2, y2], ...]
        points=points,  # Format: [[x, y], ...]
    )
    datapoint.find_metadatas.append(metadata)
    datapoint.queries.append(query)
    return query_id
```

### 5. 批量推理 (Batch Inference)

```python
# Example: Create multiple datapoints
img1 = Image.open("image1.jpg")
datapoint1 = Datapoint(find_metadatas=[], queries=[], image=None)
set_image(datapoint1, img1)
id1 = add_text_prompt(datapoint1, "a dog", query_id=1)

img2 = Image.open("image2.jpg")
datapoint2 = Datapoint(find_metadatas=[], queries=[], image=None)
set_image(datapoint2, img2)
id2 = add_text_prompt(datapoint2, "a cat", query_id=2)

# Transform datapoints
datapoint1 = transform(datapoint1)
datapoint2 = transform(datapoint2)

# Collate into a batch and move to GPU
batch = collate([datapoint1, datapoint2], dict_key="batch")["batch"]
batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

# Run inference
output = model(batch)

# Post-process results
processed_results = postprocessor.process_results(output, batch.find_metadatas)

# Access results by query ID
result1 = processed_results[id1]  # Results for "a dog" query
result2 = processed_results[id2]  # Results for "a cat" query

# Each result contains:
# - result["masks"]: Segmentation masks
# - result["boxes"]: Bounding boxes
# - result["scores"]: Confidence scores
```

## 提示类型 (Prompt Types)

### 文本提示 (Text Prompts)

```python
# Positive text prompt
add_text_prompt(datapoint, "a person wearing red", query_id=1)

# Negative text prompt (exclude unwanted detections)
metadata = InferenceMetadata(
    dataset_name="custom",
    img_id="img",
    query_id=2,
    is_negative=True,  # Mark as negative prompt
)
query = FindQueryLoaded(text="oven handle", boxes=None, points=None)
datapoint.find_metadatas.append(metadata)
datapoint.queries.append(query)
```

### 视觉提示 (Visual Prompts)

```python
# Box prompt (format: [x1, y1, x2, y2])
boxes = [[100, 100, 200, 200]]
add_visual_prompt(datapoint, boxes=boxes, query_id=3)

# Point prompt (format: [x, y])
points = [[150, 150]]
add_visual_prompt(datapoint, points=points, query_id=4)
```

### 组合提示 (Combined Prompts)

```python
# Text + Visual prompt
datapoint = Datapoint(find_metadatas=[], queries=[], image=None)
set_image(datapoint, img)
add_text_prompt(datapoint, "red object", query_id=1)
add_visual_prompt(datapoint, boxes=[[100, 100, 200, 200]], query_id=2)
```

## 性能优化建议 (Performance Optimization Tips)

1. **使用混合精度** - 启用 bfloat16 或 float16 以加快推理速度
2. **启用 TF32** - 在 Ampere GPU 上启用 TF32 可提高性能
3. **批次大小** - 根据 GPU 内存调整批次大小以最大化吞吐量
4. **编译延迟** - 第一次前向传播会较慢（编译），后续推理会更快
5. **置信度阈值** - 适当设置置信度阈值可减少后处理开销

1. **Use mixed precision** - Enable bfloat16 or float16 for faster inference
2. **Enable TF32** - Enable TF32 on Ampere GPUs for better performance
3. **Batch size** - Adjust batch size based on GPU memory to maximize throughput
4. **Compilation delay** - First forward pass will be slow (compilation), subsequent inferences will be faster
5. **Confidence threshold** - Setting appropriate confidence threshold reduces post-processing overhead

## 结果处理 (Result Processing)

```python
# Process results for a specific query
for query_id, result in processed_results.items():
    masks = result["masks"]      # Shape: (N, H, W) - N detected instances
    boxes = result["boxes"]      # Shape: (N, 4) - Bounding boxes [x1, y1, x2, y2]
    scores = result["scores"]    # Shape: (N,) - Confidence scores
    
    # Filter by confidence
    high_conf_idx = scores > 0.5
    masks = masks[high_conf_idx]
    boxes = boxes[high_conf_idx]
    scores = scores[high_conf_idx]
    
    # Process each detection
    for i in range(len(scores)):
        mask = masks[i]
        box = boxes[i]
        score = scores[i]
        # ... your processing code ...
```

## 可视化示例 (Visualization Example)

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_results(image, result):
    """Plot image with masks and bounding boxes."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    
    masks = result["masks"]
    boxes = result["boxes"]
    scores = result["scores"]
    
    # Plot masks
    for i, mask in enumerate(masks):
        color = np.random.rand(3)
        ax.imshow(mask, alpha=0.5, cmap='jet')
    
    # Plot boxes
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, edgecolor='red', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{score:.2f}', 
                color='white', fontsize=12, 
                bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Use the function
plot_results(img, processed_results[query_id])
```

## 常见问题 (Common Issues)

### GPU 内存不足 (Out of Memory)

- 减少批次大小
- 降低图像分辨率
- 使用梯度检查点（如果训练）

- Reduce batch size
- Lower image resolution
- Use gradient checkpointing (if training)

### 推理速度慢 (Slow Inference)

- 第一次推理由于编译会很慢，这是正常的
- 确保启用了 TF32 和混合精度
- 检查是否在 GPU 上运行

- First inference is slow due to compilation, this is normal
- Ensure TF32 and mixed precision are enabled
- Verify running on GPU

### 检测结果过多或过少 (Too Many/Few Detections)

- 调整 `conf_threshold` 参数
- 使用 `max_dets_per_img` 限制最大检测数量
- 尝试使用负面提示过滤不需要的检测

- Adjust `conf_threshold` parameter
- Use `max_dets_per_img` to limit maximum detections
- Try using negative prompts to filter unwanted detections

## 完整示例 (Complete Example)

详细的完整示例请参考：
For detailed complete examples, refer to:

- **Jupyter Notebook**: [`examples/sam3_image_batched_inference.ipynb`](examples/sam3_image_batched_inference.ipynb)

## 相关资源 (Related Resources)

- [SAM 3 主 README](README.md)
- [SAM 3 图像预测示例](examples/sam3_image_predictor_example.ipynb)
- [SAM 3 视频预测示例](examples/sam3_video_predictor_example.ipynb)
- [SA-Co 数据集](scripts/eval/gold/README.md)

## 许可证 (License)

本项目使用 SAM 许可证 - 详见 [LICENSE](LICENSE) 文件。

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file for details.
