# 基于多模态可解释性的社交媒体虚假新闻检测

面向微博的多模态虚假新闻检测研究，融合**文本、图像、视频与外部证据**，通过混合专家（MoE）门控融合与可解释性分析，在测试集上达到 **F1 0.934 / AUC 0.972**。

## 核心方法

- **多模态特征提取**
  - 文本：BERT（`bert-base-chinese`）
  - 视觉/视频：ResNet50
  - 证据：Qwen1.5-4B-Chat 生成证据特征
- **跨模态对齐**：`CrossModalAlignment` 对齐文本、视觉与证据三个模态
- **MoE 检测器**：文本专家 / 视觉专家 / 证据专家 + 门控网络，按样本自适应加权融合，输出伪造概率
- **证据增强**：证据抽取、BM25 相似度筛选、搜索引擎检索、推理链
- **可解释性**：跨模态注意力权重 + SHAP 文本/视觉特征归因

## 目录结构

```
configs/           # 模型与训练配置（YAML）
src/
  data/            # 数据集与证据特征缓存
  llm/             # Qwen 封装、证据抽取、检索与推理链
  models/          # 特征提取器、跨模态对齐、MoE、可解释性
  training/        # 训练、检查点、证据预计算
  utils/           # 日志与指标
results/           # 两轮训练结果（指标、曲线）
figures/           # 结果图
process_weibo.py   # 微博数据处理脚本
```

## 环境依赖

```bash
pip install -r requirements.txt
```

需要预训练模型 `bert-base-chinese` 与 `Qwen1.5-4B-Chat`，放入 `models/` 目录（见 `src/models/*.txt` 说明）。

## 使用

```bash
# 1. 数据准备：下载 Releases 中的 processed_weibo.rar 解压到 processed_weibo/
python process_weibo.py

# 2.（可选）预计算证据特征，加速训练
python src/training/precompute_evidence.py --dataset weibo

# 3. 训练
python src/training/train.py --dataset weibo [--use-cache]

# 4. 评估
python src/evaluate.py --checkpoint checkpoints/best_model.pt [--use-cache]
```

## 实验结果（微博测试集）

| 指标 | 数值 |
|------|------|
| Accuracy | 0.9300 |
| Precision | 0.9373 |
| Recall | 0.9302 |
| F1 | 0.9337 |
| AUC | 0.9722 |

## 数据说明

原始数据及检测系统完整包通过 [GitHub Releases](https://github.com/MyriadSuns/project/releases) 分发。数据含真实社交媒体内容，仅供学术研究，请勿二次分发。

## 许可

代码部分遵循 MIT License；数据版权归原始来源所有。
