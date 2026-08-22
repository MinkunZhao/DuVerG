# DuVerG

This repository provides the implementation of **DuVerG**, a framework for natural-language graph reasoning. DuVerG integrates complexity-aware routing, graph structure parsing, adaptive subgraph decomposition, LLM-based planning and code generation, and dual-branch answer verification into a unified pipeline.

The framework dynamically selects between symbolic code-based solving and neural semantic reasoning. It improves reliability through independently generated candidate answers, consistency checks, and critic-based adjudication. For large graphs, DuVerG constructs budget-constrained local subgraphs to preserve task-relevant structure while reducing context and computational overhead.

## Method Overview

1. **Complexity-aware routing:** Selects a symbolic or neural reasoning path according to task semantics and graph scale.
2. **Adaptive graph decomposition:** Allocates hop, node, and edge budgets by task type to extract relevant local structures.
3. **Dual-branch collaborative solving:** A planner formulates the algorithmic strategy, while two coders or reasoners independently produce candidate answers.
4. **Answer verification:** Combines program execution, consistency checking, and critic-based adjudication for answer selection and error recovery.

## Repository Structure

```text
DuVerG-main/
├── main.py              # Evaluation entry point
├── agents/              # Router, planner, coder, reasoner, and critic
├── core/                # LLM interface, task schemas, code execution, and evaluation
├── workflow/engine.py   # Main graph-reasoning workflow
├── utils/               # Graph parsing, pruning, and result logging
├── config/              # Model settings, prompts, and task knowledge
└── data/                # Graph-reasoning benchmark data
```

## Environment Setup

Python 3.10 or later is recommended:

```bash
conda create -n duverg python=3.10 -y
conda activate duverg

pip install networkx openai pydantic pyyaml tenacity
```

## LLM Configuration

Before running the framework, edit `config/settings.yaml`:

```yaml
llm:
  model_name: "your-model-name"
  api_key: "your-api-key"
  base_url: "https://your-endpoint.example/v1"
  timeout: 300
```

The selected service must be compatible with the OpenAI Chat Completions API. Routing thresholds, decomposition budgets, execution timeouts, and retry limits can be adjusted in `config/settings.yaml`. All commands should be executed from the repository root.

## Quick Start

Run a small subset of GraphWiz to verify the environment:

```bash
python main.py \
  --test_file data/GraphWiz/GraphWiz_test.json \
  --max_tasks 10 \
  --output_dir results
```

Remove `--max_tasks` to run the complete evaluation set:

```bash
python main.py --test_file data/GraphWiz/GraphWiz_test.json --output_dir results
```

## Data Format

The input file must contain a JSON array. The recommended format is:

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

The entry point also accepts common benchmark fields such as `question`, `answer`, and `type`. The optional `graph_data` field may contain explicit node and edge lists; when omitted, the framework attempts to parse the graph structure from the task text.

## Evaluation and Outputs

During evaluation, the terminal reports the status of each task and the overall accuracy. Results are written to:

```text
{output_dir}/result_{input_file_stem}.json
```

Each result record contains the sample `id`, `query`, `task_type`, and `success` status. The evaluator applies task-specific criteria for numerical, set-based, Boolean, matching, path, and topological-ordering problems.
