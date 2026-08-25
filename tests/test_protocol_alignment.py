import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout

import networkx as nx

from agents.critic import CriticAgent
from agents.router import RouterAgent
from core.schema import GraphTask
from main import adapt_data_to_task, run_test_mode
from utils.graph_store import EdgeListGraphStore
from utils.graph_tools import locality_preserving_prune
from workflow.engine import GraphReasoningEngine


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses) if isinstance(responses, (list, tuple)) else [responses]
        self.calls = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0) if self.responses else "{}"

    def extract_code(self, text):
        return text


class FakeEngine:
    def __init__(self, output="1", success=True):
        self.llm = FakeLLM([])
        self.output = output
        self.success = success
        self.calls = 0
        self.tasks = []

    def run(self, task):
        self.calls += 1
        self.tasks.append(task)
        return {"success": self.success, "output": self.output}


class FakeCoder:
    def __init__(self, prefix):
        self.prefix = prefix
        self.calls = 0

    def generate_code(self, *args, **kwargs):
        self.calls += 1
        return f"{self.prefix}_{self.calls}"


class FakeSandbox:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def execute(self, code):
        self.calls += 1
        return self.results.pop(0)


class FakeCritic:
    def __init__(self):
        self.focused_calls = 0

    def outputs_roughly_equal(self, task_type, a, b, status_a=True, status_b=True):
        return status_a and status_b and a == b

    def consistency_check(self, query, task_type, status_a, out_a, status_b, out_b):
        return {
            "consistent": False,
            "why": "different",
            "resolution": "need_tiebreaker",
            "tiebreaker_hint": "check the existing candidates",
        }

    def focused_check(self, *args):
        self.focused_calls += 1
        return "unresolved"


class FakeReasoner:
    def __init__(self, answer):
        self.value = answer
        self.calls = 0

    def answer(self, *args, **kwargs):
        self.calls += 1
        return self.value


def make_trace():
    return {
        "verification_log": {
            "route": "symbolic",
            "consistent": True,
            "critic_opinion": "",
            "coder_a": {"code": "", "result": "", "success": False},
            "coder_b": {"code": "", "result": "", "success": False},
        }
    }


def make_engine(sandbox_results, answer_a="neural-a", answer_b="neural-b"):
    engine = GraphReasoningEngine.__new__(GraphReasoningEngine)
    engine.coder_a = FakeCoder("a")
    engine.coder_b = FakeCoder("b")
    engine.sandbox = FakeSandbox(sandbox_results)
    engine.critic = FakeCritic()
    engine.reasoner_a = FakeReasoner(answer_a)
    engine.reasoner_b = FakeReasoner(answer_b)
    engine.max_retries = 2
    engine.global_timeout = 300
    return engine


class ProtocolAlignmentTests(unittest.TestCase):
    def test_ground_truth_is_separate_and_no_external_retry_occurs(self):
        raw = {
            "id": "sample",
            "query": "Count the nodes.",
            "task_type": "count",
            "graph_data": {},
            "ground_truth": "2",
        }
        task_data, ground_truth = adapt_data_to_task(raw, 0, "sample")
        task = GraphTask(**task_data)
        self.assertEqual(ground_truth, "2")
        self.assertFalse(hasattr(task, "ground_truth"))

        with tempfile.TemporaryDirectory() as directory:
            test_file = os.path.join(directory, "sample.json")
            with open(test_file, "w", encoding="utf-8") as handle:
                json.dump([raw], handle)
            engine = FakeEngine(output="1")
            with redirect_stdout(io.StringIO()):
                run_test_mode(engine, test_file, output_dir=directory)
            self.assertEqual(engine.calls, 1)
            self.assertFalse(hasattr(engine.tasks[0], "ground_truth"))

    def test_router_applies_lexicons_guardrails_and_llm_fallback(self):
        cfg = {
            "dense_graph_density": 0.05,
            "large_graph_nodes": 2000,
            "symbolic_keywords": ["shortest", "path"],
            "neural_keywords": ["describe", "explain", "label"],
        }

        def route(query, stats, response):
            router = RouterAgent(FakeLLM(response), cfg)
            result = router.route(query, "reasoning", {"meta": stats}, 2, 400, 2000)
            self.assertEqual(len(router.llm.calls), 1)
            return result

        small = {"nodes": 20, "edges": 30, "density": 0.01}
        self.assertEqual(route("find the shortest path", small, '{"route":"neural"}')["route"], "symbolic")
        self.assertEqual(route("describe the label", small, '{"route":"symbolic"}')["route"], "neural")
        self.assertEqual(route("explain the shortest path", small, '{"route":"neural"}')["route"], "neural")
        self.assertEqual(route("choose an answer", {"nodes": 2000, "edges": 20, "density": 0.0}, '{"route":"neural"}')["route"], "symbolic")
        self.assertEqual(route("choose an answer", {"nodes": 20, "edges": 100, "density": 0.06}, '{"route":"neural"}')["route"], "symbolic")
        self.assertEqual(route("choose an answer", small, "invalid")["route"], "symbolic")
        result = route("choose an answer about node 1", small, '{"route":"neural","need_decomposition":"false"}')
        self.assertFalse(result["need_decomposition"])

    def test_adaptive_budget_uses_formula_and_bounds(self):
        router = RouterAgent(FakeLLM("{}"), {
            "dense_graph_density": 0.05,
            "large_graph_nodes": 2000,
            "symbolic_keywords": [],
            "neural_keywords": [],
        })
        hop, nodes, edges = router._calculate_adaptive_budget(
            "page_rank", "pagerank within 2 hops", {"nodes": 1000, "edges": 3000}
        )
        self.assertEqual((hop, nodes, edges), (2, 50000, 450000))
        _, nodes, edges = router._calculate_adaptive_budget(
            "page_rank", "pagerank within 2 hops", {"nodes": 1000, "edges": 1000000000}
        )
        self.assertEqual(nodes, 2000000)
        self.assertEqual(edges, 100000000)
        hop, _, _ = router._calculate_adaptive_budget(
            "page_rank", "pagerank within 20 hops", {"nodes": 1000, "edges": 3000}
        )
        self.assertEqual(hop, 3)

    def test_projection_requires_valid_seeds(self):
        engine = GraphReasoningEngine.__new__(GraphReasoningEngine)
        engine.decomp_cfg = {"enable": True, "auto_trigger_nodes": 2000, "seed_trigger_nodes": 100}
        no_seed = {"need_decomposition": True, "decomposition": {"seed_nodes": []}}
        local_seed = {"need_decomposition": True, "decomposition": {"seed_nodes": [1]}}
        nonlocal_seed = {"need_decomposition": False, "decomposition": {"seed_nodes": [1]}}
        self.assertFalse(engine._should_decompose({"nodes": 5000}, no_seed))
        self.assertTrue(engine._should_decompose({"nodes": 5000}, nonlocal_seed))
        self.assertTrue(engine._should_decompose({"nodes": 100}, local_seed))
        self.assertFalse(engine._should_decompose({"nodes": 100}, nonlocal_seed))

        graph = nx.Graph()
        graph.add_edge(1, 2)
        self.assertIsNone(locality_preserving_prune(graph, [9], 2, 10, 10))

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "graph.edgelist")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("1 2\n2 3\n")
            store = EdgeListGraphStore(path)
            self.assertIsNone(store.prune_khop([9], 2, 10, 10))

    def test_critic_normalization_is_answer_type_aware(self):
        critic = CriticAgent(FakeLLM("unresolved"))
        self.assertTrue(critic.outputs_roughly_equal("connected_nodes", "1, 2", "[2, 1]"))
        self.assertTrue(critic.outputs_roughly_equal("connected_nodes", "", "[]"))
        self.assertTrue(critic.outputs_roughly_equal("connectivity", "Yes", "true"))
        self.assertTrue(critic.outputs_roughly_equal("clustering_coefficient", "0.5000", "0.501"))
        self.assertFalse(critic.outputs_roughly_equal("shortest_path", "[1, 2, 3]", "[3, 2, 1]"))
        result = critic.consistency_check("q", "connected_nodes", True, "", False, "Runtime Error")
        self.assertEqual(result["resolution"], "pick_a")
        self.assertEqual(len(critic.llm.calls), 0)

    def test_symbolic_disagreement_never_generates_a_third_program(self):
        engine = make_engine([(True, "1"), (True, "2")])
        task = GraphTask(
            id="x",
            dataset_name="talk_like_a_graph_test",
            query="What is the maximum flow from node 1 to node 2?",
            task_type="maximum_flow",
            graph_data={"nodes": [1, 2], "edges": [[1, 2, 3]]},
        )
        output, success = engine._run_symbolic_path(task, {}, task.graph_data, make_trace(), time.time())
        self.assertEqual(output, "neural-a")
        self.assertFalse(success)
        self.assertEqual(engine.coder_a.calls + engine.coder_b.calls, 2)
        self.assertEqual(engine.sandbox.calls, 2)
        self.assertFalse(hasattr(engine, "_oracle_for_talk_like_a_graph"))

    def test_execution_failures_use_only_bounded_repairs(self):
        failures = [(False, "Runtime Error") for _ in range(4)]
        engine = make_engine(failures, answer_a="fallback", answer_b="fallback")
        task = GraphTask(
            id="x",
            dataset_name="sample",
            query="Count the nodes.",
            task_type="count",
            graph_data={"nodes": [1, 2], "edges": [[1, 2]]},
        )
        output, success = engine._run_symbolic_path(task, {}, task.graph_data, make_trace(), time.time())
        self.assertEqual(output, "fallback")
        self.assertTrue(success)
        self.assertEqual(engine.coder_a.calls + engine.coder_b.calls, 4)
        self.assertEqual(engine.sandbox.calls, 4)

    def test_legitimate_empty_output_is_not_an_execution_failure(self):
        engine = make_engine([(True, ""), (False, "Runtime Error")])
        task = GraphTask(
            id="x",
            dataset_name="sample",
            query="Which nodes are disconnected from node 1?",
            task_type="disconnected_nodes",
            graph_data={"nodes": [1], "edges": []},
        )
        output, success = engine._run_symbolic_path(task, {}, task.graph_data, make_trace(), time.time())
        self.assertEqual(output, "")
        self.assertTrue(success)
        self.assertEqual(engine.coder_a.calls + engine.coder_b.calls, 2)


if __name__ == "__main__":
    unittest.main()
