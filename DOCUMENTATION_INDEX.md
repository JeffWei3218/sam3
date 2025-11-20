# SAM3 文档索引 (Documentation Index)

本目录包含SAM3项目的代码结构分析和流程图文档。

## 📚 文档列表

### 1. [CODE_STRUCTURE.md](CODE_STRUCTURE.md) - 完整版代码结构文档
**推荐给**: 开发者、研究人员

**内容包括**:
- ✅ 详细的目录结构和组件说明
- ✅ 完整的训练流程图（Mermaid格式）
- ✅ 图像和视频推理流程图
- ✅ 模型架构详细图解
- ✅ 数据处理流程图
- ✅ 配置参数说明
- ✅ 性能优化技巧

**文件大小**: 23KB | **行数**: 820行

### 2. [CODE_STRUCTURE_CN.md](CODE_STRUCTURE_CN.md) - 简化中文版
**推荐给**: 快速入门、实际应用开发者

**内容包括**:
- ✅ 训练和推理快速代码示例
- ✅ 关键文件功能说明
- ✅ 简化的架构概览
- ✅ 常见问题解答
- ✅ 实用配置示例

**文件大小**: 7.2KB | **行数**: 341行

---

## 🎯 快速导航

### 按需求选择文档

| 你想了解... | 推荐文档 | 章节 |
|------------|---------|------|
| 如何开始训练 | [CODE_STRUCTURE_CN.md](CODE_STRUCTURE_CN.md) | 训练流程简述 |
| 训练详细流程 | [CODE_STRUCTURE.md](CODE_STRUCTURE.md) | 训练流程 → 训练流程图 |
| 如何进行推理 | [CODE_STRUCTURE_CN.md](CODE_STRUCTURE_CN.md) | 推理流程简述 |
| 推理详细流程 | [CODE_STRUCTURE.md](CODE_STRUCTURE.md) | 推理流程 |
| 模型架构细节 | [CODE_STRUCTURE.md](CODE_STRUCTURE.md) | 模型架构 |
| 关键文件说明 | [CODE_STRUCTURE_CN.md](CODE_STRUCTURE_CN.md) | 关键文件说明 |
| 数据处理流程 | [CODE_STRUCTURE.md](CODE_STRUCTURE.md) | 数据处理流程 |
| 常见问题 | [CODE_STRUCTURE_CN.md](CODE_STRUCTURE_CN.md) | 常见问题 |

---

## 📊 流程图预览

文档中包含以下 Mermaid 流程图：

### 训练流程
- **主训练流程图**: 从启动到完成的完整训练流程
- **数据加载流程**: 数据集加载、增强和批次整理
- **前向传播流程**: 从输入到预测的详细步骤
- **损失计算流程**: 匹配、损失计算和反向传播

### 推理流程
- **图像推理流程**: 完整的图像分割推理流程
- **视频推理流程**: 视频检测和跟踪的详细流程
- **提示处理流程**: 多模态提示的处理方式

### 架构图
- **整体架构**: 检测器和跟踪器的完整架构
- **核心模块**: Transformer编码器/解码器详解
- **数据流图**: 训练和推理的数据流

> **注意**: 这些流程图使用 Mermaid 语法编写，在 GitHub 和大多数现代 Markdown 查看器中可以直接渲染显示。

---

## 🚀 快速开始

### 第一次使用？从这里开始：

1. **快速了解** → 阅读 [CODE_STRUCTURE_CN.md](CODE_STRUCTURE_CN.md)
2. **深入学习** → 参考 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)
3. **实践应用** → 查看 [examples/](examples/) 目录

### 想要训练模型？

```bash
# 查看训练文档
cat README_TRAIN.md

# 运行训练（参考 CODE_STRUCTURE_CN.md 的训练部分）
python sam3/train/train.py -c configs/your_config.yaml
```

### 想要进行推理？

```python
# 查看 CODE_STRUCTURE_CN.md 的推理示例
# 或运行 examples/ 中的 notebook
```

---

## 📖 其他文档

- [README.md](README.md) - 项目主文档
- [README_TRAIN.md](README_TRAIN.md) - 训练详细指南
- [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南
- [LICENSE](LICENSE) - 许可证信息

---

## 🔗 相关资源

- **论文**: [SAM 3: Segment Anything with Concepts](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- **项目主页**: [https://ai.meta.com/sam3](https://ai.meta.com/sam3)
- **GitHub**: [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- **Demo**: [https://segment-anything.com/](https://segment-anything.com/)
- **数据集**: SA-Co ([Gold](https://huggingface.co/datasets/facebook/SACo-Gold), [Silver](https://huggingface.co/datasets/facebook/SACo-Silver), [VEval](https://huggingface.co/datasets/facebook/SACo-VEval))

---

## 💡 贡献

如果你发现文档中有错误或需要改进的地方，欢迎：
- 提交 Issue
- 发起 Pull Request
- 联系维护者

---

**最后更新**: 2025-11-20  
**文档版本**: 1.0
