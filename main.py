import argparse
import yaml
import json
import os

from core.llm import LLMEngine
from core.sandbox import CodeSandbox
from core.schema import GraphTask
from workflow.engine import GraphReasoningEngine
from utils.json_logger import JSONLogger
from core.evaluator import Evaluator


def load_config():
    with open("config/settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def adapt_data_to_task(raw_item, index, default_dataset_name="Unknown"):
    query = raw_item.get("query") or raw_item.get("question") or raw_item.get("text")
    task_type = raw_item.get("task_type") or raw_item.get("type") or "reasoning"
    ground_truth = raw_item.get("ground_truth") or raw_item.get("answer") or str(raw_item.get("target", ""))
    task_id = str(raw_item.get("id")) if "id" in raw_item else f"{default_dataset_name}_{index}"
    return {
        "id": task_id,
        "dataset_name": default_dataset_name,
        "query": query,
        "task_type": task_type,
        "graph_data": raw_item.get("graph_data", {}),
        "ground_truth": ground_truth,
    }


def run_test_mode(engine, test_file, max_tasks=None, output_dir="results"):
    print(f"🧪 [Test] Running evaluation on {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        raw_tasks = json.load(f)

    dataset_name = os.path.basename(test_file).split(".")[0]
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"result_{dataset_name}.json")

    logger = JSONLogger(output_file)
    evaluator = Evaluator(engine.llm)

    results = []
    n = len(raw_tasks) if max_tasks is None else min(len(raw_tasks), int(max_tasks))

    for i in range(n):
        raw_item = raw_tasks[i]
        task_dict = adapt_data_to_task(raw_item, i, dataset_name)
        task = GraphTask(**task_dict)
        evaluator = Evaluator(engine.llm)

        start_prompt = engine.llm.total_prompt_tokens
        start_compl = engine.llm.total_completion_tokens

        status = engine.run(task)
        is_correct = False
        if status['success']:
            is_correct = evaluator.evaluate(dataset_name, task.task_type, status['output'], task.ground_truth,
                                            task.query)

        if not is_correct:
            print(f"Attempt 1 failed for {task.id}. Retrying...")
            status = engine.run(task, is_retry=True)
            if status['success']:
                is_correct = evaluator.evaluate(dataset_name, task.task_type, status['output'], task.ground_truth,
                                                task.query)

        end_prompt = engine.llm.total_prompt_tokens
        end_compl = engine.llm.total_completion_tokens

        used_prompt = end_prompt - start_prompt
        used_compl = end_compl - start_compl
        used_total = used_prompt + used_compl

        final_success = status['success'] and is_correct
        record = {
            "id": task.id,
            "query": task.query,
            "task_type": task.task_type,
            "success": final_success,
        }
        print("id:", record["id"])
        print("task_type:", record["task_type"])
        print("success:", record["success"])
        print("")
        logger.log(record)
        results.append(record)

    acc = sum(1 for r in results if r["success"]) / len(results) if results else 0
    print(f"\n🎉 Test Complete. Accuracy: {acc:.2%}")
    print(f"📂 Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    cfg = load_config()
    llm = LLMEngine(cfg["llm"])
    sandbox = CodeSandbox(timeout=cfg["sandbox"]["timeout"])

    engine = GraphReasoningEngine(llm=llm, sandbox=sandbox, cfg=cfg)

    run_test_mode(engine, args.test_file, max_tasks=args.max_tasks, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
