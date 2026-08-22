# DuVerG

本仓库是 DuVerG 的代码实现。项目面向自然语言图推理任务，将复杂度感知路由、图结构解析、自适应子图分解、LLM 规划与代码生成以及双路答案验证组织为统一流程。

系统在符号代码求解与神经语义推理之间动态选择路径，并通过独立候选答案、一致性检查和 critic 提升推理可靠性。对于大规模图，DuVerG 使用预算受控的局部子图，在保留任务相关结构的同时降低上下文与计算开销。

## 方法概览

1. **复杂度感知路由：** 根据任务语义与图规模选择符号或神经推理路径。
2. **自适应图分解：** 按任务类型分配 hop、节点与边预算，提取相关局部结构。
3. **双路协同求解：** planner 生成算法计划，两个 coder 或 reasoner 独立产生候选答案。
4. **结果验证：** 结合程序执行、一致性检查与 critic 完成答案选择和错误恢复。

## 目录结构

```text
DuVerG-main/
├── main.py              # 评测入口
├── agents/              # router、planner、coder、reasoner 与 critic
├── core/                # LLM 接口、任务定义、代码执行与评测
├── workflow/engine.py   # 图推理主流程
├── utils/               # 图解析、裁剪与结果记录
├── config/              # 模型配置、prompts 与任务知识
└── data/                # 图推理基准数据
```

## 环境安装

建议使用 Python 3.10 或更高版本：

```bash
conda create -n duverg python=3.10 -y
conda activate duverg

pip install networkx openai pydantic pyyaml tenacity
```

## LLM 配置

运行前编辑 `config/settings.yaml`：

```yaml
llm:
  model_name: "your-model-name"
  api_key: "your-api-key"
  base_url: "https://your-endpoint.example/v1"
  timeout: 300
```

所使用的服务需要兼容 OpenAI Chat Completions API。路由阈值、分解预算、执行超时和重试次数均可在 `config/settings.yaml` 中调整。所有命令应从仓库根目录执行。

## 快速开始

使用少量 GraphWiz 样本快速验证环境：

```bash
python main.py \
  --test_file data/GraphWiz/GraphWiz_test.json \
  --max_tasks 10 \
  --output_dir results
```

移除 `--max_tasks` 即可运行完整评测：

```bash
python main.py --test_file data/GraphWiz/GraphWiz_test.json --output_dir results
```

## 数据格式

输入文件为 JSON 数组，推荐格式如下：

```json
[
  {
    "id": "example_0",
    "query": "Q: What is the shortest path from node 0 to node 3?",
    "task_type": "shortest_path",
    "ground_truth": "[0, 1, 3]",
    "graph_data": {
      "directed": false,
      "nodes": [0, 1, 2, 3],
      "edges": [[0, 1], [1, 3], [0, 2], [2, 3]]
    }
  }
]
```

入口同时兼容 `question` / `answer` / `type` 等常见基准字段。`graph_data` 可使用节点与边列表，也可省略并由系统从问题文本中解析图结构。

## 评测与输出

完成运行后，终端会输出逐任务状态和整体 Accuracy，结果写入：

```text
{output_dir}/result_{input_file_stem}.json
```

结果文件记录每个样本的 `id`、`query`、`task_type` 和 `success`。评测器针对数值、集合、布尔判断、匹配、路径和拓扑序等任务使用相应判定规则，并在终端报告整体准确率。
