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
# 1. 数据准备：下载 Releases 中的 processed_weibo.zip 解压到 processed_weibo/
python process_weibo.py

# 2.（可选）预计算证据特征，加速训练
python src/training/precompute_evidence.py --dataset weibo

# 3. 训练
python src/training/train.py --dataset weibo [--use-cache]

# 4. 评估
python src/evaluate.py --checkpoint checkpoints/best_model.pt [--use-cache]
```

## 检测系统（Web 应用）

除训练与研究方法外，本仓库还配套一套完整的**检测系统 Web 应用**（`detectionsystem.zip`，通过 Releases 分发），可直接部署使用：

- **后端（Flask）**
  - 接口：认证、管理员、检测
  - 核心：检测流水线 `app/core/detection_pipeline.py`
  - 数据模型：用户、检测记录、审计日志、异步任务
  - 安全与运维：验证码、登录限流、密码校验、请求日志、错误处理、文件存储、监控
- **前端（Vue3）**：组件、视图、Pinia 状态管理、接口封装
- **模型**：`models/` 目录存放训练好的检测模型

部署与启动说明见压缩包内 `backend/README.md`。

## 实验结果（微博测试集）

| 指标 | 数值 |
|------|------|
| Accuracy | 0.9300 |
| Precision | 0.9373 |
| Recall | 0.9302 |
| F1 | 0.9337 |
| AUC | 0.9722 |

## 数据与系统包（Releases）

通过 [GitHub Releases](https://github.com/MyriadSuns/project/releases) 分发两个压缩包：

| 包 | 大小 | 内容 |
|----|------|------|
| `processed_weibo.zip` | 1.2GB | 预处理后的微博数据集（图片/视频/标注 CSV） |
| `detectionsystem.zip` | 674MB | 完整检测系统 Web 应用（Flask 后端 + Vue3 前端 + 模型） |

数据含真实社交媒体内容，仅供学术研究，请勿二次分发。

## 许可

代码部分遵循 MIT License；数据版权归原始来源所有。
