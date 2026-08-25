import json
import time
import tempfile
import os
from typing import Any, Dict, Tuple

from core.schema import GraphTask
from agents.router import RouterAgent
from agents.planner import PlannerAgent
from agents.coder import CoderAgent
from agents.critic import CriticAgent
from agents.reasoner import NeuralReasonerAgent

from utils.graph_tools import (
    build_nx_graph,
    graph_stats,
    locality_preserving_prune,
    graph_to_payload,
)

try:
    from utils.graph_store import get_graph_store
except Exception:
    get_graph_store = None


class GraphReasoningEngine:
    def __init__(self, llm, sandbox, cfg: Dict[str, Any]):
        self.llm = llm
        self.sandbox = sandbox
        self.cfg = cfg

        self.router = RouterAgent(llm=self.llm, cfg=cfg["router"])
        self.planner = PlannerAgent(llm=self.llm)

        self.coder_a = CoderAgent(llm=self.llm, name="CoderA")
        self.coder_b = CoderAgent(llm=self.llm, name="CoderB")
        self.critic = CriticAgent(llm=self.llm)

        self.reasoner_a = NeuralReasonerAgent(llm=self.llm, name="ReasonerA")
        self.reasoner_b = NeuralReasonerAgent(llm=self.llm, name="ReasonerB")

        self.max_retries = int(cfg.get("experiment", {}).get("max_retries", 2))
        self.global_timeout = int(cfg.get("experiment", {}).get("task_timeout", 300))
        self.decomp_cfg = cfg.get("decomposition", {})


    def run(self, task: GraphTask) -> Dict[str, Any]:
        t0 = time.time()

        trace = {
            "router_log": {},
            "decomposition_log": {
                "triggered": False,
                "stats_before": {},
                "stats_after": {},
                "mode": "locality",
            },
            "planner_log": {},
            "verification_log": {
                "route": "",
                "consistent": True,
                "critic_opinion": "",
                "coder_a": {"code": "", "result": "", "success": False},
                "coder_b": {"code": "", "result": "", "success": False},
            },
            "error_log": None,
        }

        try:
            graph_data0 = task.graph_data or {}
            G_full = None
            initial_stats = self._get_initial_stats(graph_data0, task.query)
            trace["decomposition_log"]["stats_before"] = initial_stats

            route_obj = self.router.route(
                query=task.query,
                task_type=task.task_type,
                graph_data=graph_data0,
                default_hop=int(self.decomp_cfg.get("hop", 2)),
                node_budget=int(self.decomp_cfg.get("node_budget", 400)),
                edge_budget=int(self.decomp_cfg.get("edge_budget", 2000)),
            )
            trace["router_log"] = route_obj
            trace["verification_log"]["route"] = route_obj.get("route", "symbolic")

            effective_graph_data = graph_data0
            graph_hint = self._make_graph_hint(initial_stats, effective_graph_data)

            if self._should_decompose(initial_stats, route_obj):
                mode = str(self.decomp_cfg.get("mode", "locality")).lower().strip()
                if mode not in ("locality", "random"):
                    mode = "locality"
                trace["decomposition_log"]["mode"] = mode

                de = route_obj.get("decomposition", {}) or {}
                hop = int(de.get("hop", self.decomp_cfg.get("hop", 2)))
                node_budget = int(de.get("node_budget", self.decomp_cfg.get("node_budget", 400)))
                edge_budget = int(de.get("edge_budget", self.decomp_cfg.get("edge_budget", 2000)))
                seeds = de.get("seed_nodes", [])

                if (
                    isinstance(graph_data0, dict)
                    and graph_data0.get("graph_file")
                    and get_graph_store is not None
                ):
                    store = get_graph_store(
                        graph_data0["graph_file"],
                        directed=bool(graph_data0.get("directed", False)),
                        weighted=bool(graph_data0.get("weighted", False)),
                        delimiter=graph_data0.get("delimiter"),
                        comment_prefix=graph_data0.get("comment_prefix", "#"),
                    )

                    seed_ints = []
                    for s in (seeds or []):
                        try:
                            seed = int(s)
                            if seed in store._nodes:
                                seed_ints.append(seed)
                        except Exception:
                            pass

                    if not seed_ints:
                        H = None
                    elif mode == "random":
                        H = self._random_project_from_store(store, node_budget, edge_budget)
                    else:
                        H = store.prune_khop(
                            seed_nodes=seed_ints,
                            hop=hop,
                            node_budget=node_budget,
                            edge_budget=edge_budget,
                        )

                else:
                    G_full = build_nx_graph(graph_data0, query=task.query)
                    valid_seeds = [s for s in seeds if s in G_full]

                    if not valid_seeds:
                        H = None
                    elif mode == "random":
                        H = self._random_project_from_nx(G_full, node_budget, edge_budget)
                    else:
                        H = locality_preserving_prune(G_full, valid_seeds, hop, node_budget, edge_budget)

                if H is not None:
                    trace["decomposition_log"]["triggered"] = True
                    effective_graph_data = graph_to_payload(H)
                    new_stats = graph_stats(H)
                    trace["decomposition_log"]["stats_after"] = new_stats
                    graph_hint = self._make_graph_hint(new_stats, effective_graph_data)

            if route_obj.get("route") == "neural":
                output, success = self._run_neural_path(task, graph_hint, trace)
                return self._wrap_response(success, output, trace, t0)

            plan = self.planner.plan(task.query, task.task_type, initial_stats, graph_hint)
            trace["planner_log"] = plan

            output, success = self._run_symbolic_path(
                task=task,
                plan=plan,
                graph_data=effective_graph_data,
                trace=trace,
                t0=t0,
            )
            return self._wrap_response(success, output, trace, t0)

        except Exception as e:
            trace["error_log"] = str(e)
            return self._wrap_response(False, "", trace, t0, error=str(e))


    def _get_initial_stats(self, graph_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        if isinstance(graph_data, dict) and isinstance(graph_data.get("meta"), dict):
            meta = dict(graph_data["meta"])
            return self._coerce_stats(meta)

        if (
            isinstance(graph_data, dict)
            and graph_data.get("graph_file")
            and get_graph_store is not None
        ):
            store = get_graph_store(
                graph_data["graph_file"],
                directed=bool(graph_data.get("directed", False)),
                weighted=bool(graph_data.get("weighted", False)),
                delimiter=graph_data.get("delimiter"),
                comment_prefix=graph_data.get("comment_prefix", "#"),
            )
            return self._coerce_stats(store.stats())

        G = build_nx_graph(graph_data or {}, query=query)
        return self._coerce_stats(graph_stats(G))

    def _coerce_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(stats or {})
        out.setdefault("nodes", 0)
        out.setdefault("edges", 0)
        out.setdefault("density", 0.0)
        out.setdefault("directed", False)
        out.setdefault("weighted", False)
        out.setdefault("capacitated", False)
        return out

    def _should_decompose(self, stats: Dict[str, Any], route_obj: Dict[str, Any]) -> bool:
        if not self.decomp_cfg.get("enable", True):
            return False

        auto_trigger = int(self.decomp_cfg.get("auto_trigger_nodes", 800))
        seed_trigger = int(self.decomp_cfg.get("seed_trigger_nodes", 200))

        seeds = (route_obj.get("decomposition", {}) or {}).get("seed_nodes", []) or []
        need = bool(route_obj.get("need_decomposition", False))

        if not seeds:
            return False

        n = int(stats.get("nodes", 0))
        if n >= auto_trigger:
            return True
        if n >= seed_trigger and need:
            return True
        return False

    def _random_project_from_nx(self, G, node_budget: int, edge_budget: int):
        import random
        import networkx as nx

        if G is None or G.number_of_nodes() == 0:
            return G

        nodes = list(G.nodes())
        if len(nodes) <= node_budget:
            H = G.copy()
        else:
            picked = random.sample(nodes, k=node_budget)
            H = G.subgraph(picked).copy()

        if H.number_of_edges() > edge_budget:
            edges = list(H.edges(data=True))[:edge_budget]
            G2 = nx.DiGraph() if H.is_directed() else nx.Graph()
            G2.add_nodes_from(H.nodes(data=True))
            for u, v, d in edges:
                G2.add_edge(u, v, **(d or {}))
            return G2

        return H

    def _random_project_from_store(self, store, node_budget: int, edge_budget: int):
        import random
        import networkx as nx

        try:
            nodes = list(store._nodes)
            directed = bool(store.directed)
            weighted = bool(store.weighted)
        except Exception:
            return nx.DiGraph() if False else nx.Graph()

        if not nodes:
            return nx.DiGraph() if directed else nx.Graph()

        if len(nodes) <= node_budget:
            picked = set(nodes)
        else:
            picked = set(random.sample(nodes, k=node_budget))

        H = nx.DiGraph() if directed else nx.Graph()
        H.add_nodes_from(list(picked))

        added = 0
        for u in list(picked):
            if added >= edge_budget:
                break
            try:
                neigh = store._adj.get(u, [])
            except Exception:
                neigh = []
            if weighted:
                for v, w in neigh:
                    if v in picked:
                        H.add_edge(u, v, weight=float(w), capacity=float(w))
                        added += 1
                        if added >= edge_budget:
                            break
            else:
                for v in neigh:
                    if v in picked:
                        H.add_edge(u, v, capacity=1.0)
                        added += 1
                        if added >= edge_budget:
                            break
        return H


    def _run_neural_path(self, task: GraphTask, graph_hint: str, trace: Dict) -> Tuple[str, bool]:
        ans_a = self.reasoner_a.answer(task.query, task.task_type, graph_hint, temperature=0.2)
        ans_b = self.reasoner_b.answer(task.query, task.task_type, graph_hint, temperature=0.6)

        trace["verification_log"]["coder_a"]["result"] = ans_a
        trace["verification_log"]["coder_b"]["result"] = ans_b

        if self.critic.outputs_roughly_equal(task.task_type, ans_a, ans_b):
            return ans_a, True

        ck = self.critic.consistency_check(task.query, task.task_type, True, ans_a, True, ans_b)
        trace["verification_log"]["consistent"] = False
        trace["verification_log"]["critic_opinion"] = ck.get("why", "")

        resolution = ck.get("resolution", "need_tiebreaker")
        if resolution == "pick_a":
            return ans_a, True
        if resolution == "pick_b":
            return ans_b, True

        resolution = self.critic.focused_check(
            task.query,
            task.task_type,
            ans_a,
            ans_b,
            ans_a,
            ans_b,
            ck.get("tiebreaker_hint", ""),
        )
        if resolution == "pick_a":
            return ans_a, True
        if resolution == "pick_b":
            return ans_b, True
        return ans_a, False


    def _run_symbolic_path(
        self,
        task: GraphTask,
        plan: Dict,
        graph_data: Dict,
        trace: Dict,
        t0: float,
    ) -> Tuple[str, bool]:
        payload_hint = self._graph_payload_hint(graph_data)

        diversity_a = "Prioritize standard NetworkX algorithms."
        diversity_b = "Focus on explicit edge-case handling (isolated nodes, missing nodes, capacity parsing)."

        code_a = self.coder_a.generate_code(task.query, task.task_type, plan, payload_hint, diversity_hint=diversity_a)
        ok_a, out_a = self.sandbox.execute(code_a or "")
        trace["verification_log"]["coder_a"] = {"code": code_a, "result": out_a, "success": ok_a}

        code_b = self.coder_b.generate_code(task.query, task.task_type, plan, payload_hint, diversity_hint=diversity_b)
        ok_b, out_b = self.sandbox.execute(code_b or "")
        trace["verification_log"]["coder_b"] = {"code": code_b, "result": out_b, "success": ok_b}

        if ok_a and ok_b:
            if self.critic.outputs_roughly_equal(task.task_type, out_a, out_b):
                trace["verification_log"]["consistent"] = True
                return out_a, True

            trace["verification_log"]["consistent"] = False
            ck = self.critic.consistency_check(task.query, task.task_type, ok_a, out_a, ok_b, out_b)
            trace["verification_log"]["critic_opinion"] = ck.get("why", "Outputs differ but reason unclear.")

            res = ck.get("resolution", "need_tiebreaker")
            if res == "pick_a":
                return out_a, True
            if res == "pick_b":
                return out_b, True

            res = self.critic.focused_check(
                task.query,
                task.task_type,
                out_a,
                out_b,
                code_a,
                code_b,
                ck.get("tiebreaker_hint", ""),
            )
            if res == "pick_a":
                return out_a, True
            if res == "pick_b":
                return out_b, True
            try:
                stats = self._get_initial_stats(graph_data, task.query)
            except Exception:
                stats = {"nodes": 0, "edges": 0, "directed": False, "weighted": False, "capacitated": False}
            return self._run_neural_path(task, self._make_graph_hint(stats, graph_data), trace)

        if ok_a:
            return out_a, True
        if ok_b:
            return out_b, True

        if not (ok_a or ok_b):
            last_error = out_a if out_a else "Unknown error"
            for _ in range(self.max_retries):
                if time.time() - t0 > self.global_timeout:
                    break
                repair_code = self.coder_a.generate_code(
                    task.query,
                    task.task_type,
                    plan,
                    payload_hint,
                    error_feedback=last_error,
                    diversity_hint="Fix the previous execution error.",
                )
                ok_r, out_r = self.sandbox.execute(repair_code or "")
                if ok_r:
                    return out_r, True
                last_error = out_r

        try:
            stats = self._get_initial_stats(graph_data, task.query)
        except Exception:
            stats = {"nodes": 0, "edges": 0, "directed": False, "weighted": False, "capacitated": False}
        return self._run_neural_path(task, self._make_graph_hint(stats, graph_data), trace)


    def _graph_payload_hint(self, graph_data: Dict[str, Any]) -> str:
        s = json.dumps(graph_data, ensure_ascii=False)
        if len(s) > 8000:
            fd, path = tempfile.mkstemp(suffix=".json", text=True)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, ensure_ascii=False)
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
            return f"Note: Graph is large. Data saved to temporary JSON file at: {path}. Please load this file in your code."
        return s

    def _make_graph_hint(self, stats: Dict[str, Any], graph_data: Dict[str, Any]) -> str:
        stats = self._coerce_stats(stats)
        cap = bool(stats.get("capacitated", False))
        return (
            f"V={stats['nodes']}, E={stats['edges']}, "
            f"Directed={stats['directed']}, Weighted={stats['weighted']}, Capacitated={cap}"
        )

    def _wrap_response(self, success: bool, output: str, trace: Dict, t0: float, error: str = None) -> Dict[str, Any]:
        return {
            "success": success,
            "output": (output or "").strip(),
            "runtime": time.time() - t0,
            "trace": trace,
            "error": error,
        }

