import json
import yaml
import re
import math
from typing import Any, Dict

from .base import BaseAgent
from utils.graph_tools import build_nx_graph, graph_stats, extract_seed_nodes
from utils.graph_store import get_graph_store


class RouterAgent(BaseAgent):
    def __init__(self, llm, cfg: Dict[str, Any]):
        super().__init__("Router", llm)
        self.cfg = cfg
        self.decomp_cfg = self._load_decomp_config()
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)["router"]

    def _load_decomp_config(self):
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("decomposition", {})

    def _estimate_hop_from_query(self, query: str, default: int) -> int:
        match = re.search(r"within\s+(\d+)\s+hops", query.lower())
        if match: return int(match.group(1))
        match = re.search(r"(\d+)-hop", query.lower())
        if match: return int(match.group(1))
        if "reach" in query.lower() or "path" in query.lower():
            return max(default, 3)
        return default

    def _calculate_adaptive_budget(self, task_type: str, query: str, graph_stats: Dict):
        strategies = self.decomp_cfg.get("strategies", {})
        task_key = (task_type or "").lower()
        query_lower = query.lower()

        strategy = strategies.get("local_micro", {})
        found = False
        for s_name, s_cfg in strategies.items():
            for kw in s_cfg.get("keywords", []):
                if kw in task_key or kw in query_lower:
                    strategy = s_cfg
                    found = True
                    break
            if found: break

        k = self._estimate_hop_from_query(query, strategy.get("default_hop", 2))
        default_hops = [int(cfg.get("default_hop", 2)) for cfg in strategies.values()]
        max_hop = int(self.decomp_cfg.get("max_hop", max(default_hops or [3])))
        k = max(1, min(int(k), max_hop))

        n_total = int(graph_stats.get("nodes", 0))
        e_total = int(graph_stats.get("edges", 0))
        avg_deg = (2 * e_total) / max(1, n_total) if n_total > 0 else 10

        base_branching = max(avg_deg, self.decomp_cfg.get("base_branching_factor", 10))
        estimated_nodes = math.pow(base_branching, k)
        scale = strategy.get("node_scale", 2.0)

        target_node_budget = int(estimated_nodes * scale)

        limit_min = self.decomp_cfg.get("min_node_budget", 1000)
        limit_max = self.decomp_cfg.get("max_node_budget", 100000)
        target_node_budget = max(limit_min, min(target_node_budget, limit_max))

        target_edge_budget = int(target_node_budget * avg_deg * 1.5)
        edge_max = self.decomp_cfg.get("max_edge_budget", 500000)
        target_edge_budget = min(target_edge_budget, edge_max)

        return k, target_node_budget, target_edge_budget

    def route(self, query: str, task_type: str, graph_data: Dict[str, Any], default_hop: int, node_budget: int,
              edge_budget: int) -> Dict[str, Any]:
        if isinstance(graph_data, dict) and graph_data.get("meta"):
            stats = graph_data["meta"]
        else:
            if isinstance(graph_data, dict) and graph_data.get("graph_file") and get_graph_store:
                store = get_graph_store(
                    graph_data["graph_file"],
                    directed=bool(graph_data.get("directed", False)),
                    weighted=bool(graph_data.get("weighted", False)),
                    delimiter=graph_data.get("delimiter"),
                    comment_prefix=graph_data.get("comment_prefix", "#"),
                )
                stats = store.stats()
            else:
                G = build_nx_graph(graph_data, query=query)
                stats = graph_stats(G)

        adaptive_hop, adaptive_node, adaptive_edge = self._calculate_adaptive_budget(task_type, query, stats)

        prompt = self.prompts["user_template"].format(
            query=query,
            task_type=task_type,
            graph_stats=json.dumps(stats, ensure_ascii=False),
            default_hop=adaptive_hop,
            node_budget=adaptive_node,
            edge_budget=adaptive_edge,
        )
        llm_obj = {}
        try:
            raw = self.llm.chat(
                messages=[
                    {"role": "system", "content": self.prompts["system"]},
                    {"role": "user", "content": prompt},
                ],
                json_mode=True,
                temperature=0.1,
            )
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                llm_obj = parsed
        except Exception:
            llm_obj = {}

        qlow = (query or "").lower()
        symbolic_hit = any(k in qlow for k in self.cfg.get("symbolic_keywords", []))
        neural_hit = any(k in qlow for k in self.cfg.get("neural_keywords", []))
        n = int(stats.get("nodes", 0))
        density = float(stats.get("density", 0.0))

        if symbolic_hit and not neural_hit:
            route = "symbolic"
            rule = "symbolic_lexicon"
        elif neural_hit and not symbolic_hit:
            route = "neural"
            rule = "neural_lexicon"
        elif n >= int(self.cfg.get("large_graph_nodes", 2000)) or density >= float(self.cfg.get("dense_graph_density", 0.05)):
            route = "symbolic"
            rule = "graph_guardrail"
        else:
            route = llm_obj.get("route") if llm_obj.get("route") in ("symbolic", "neural") else "symbolic"
            rule = "llm" if llm_obj.get("route") in ("symbolic", "neural") else "fallback"

        seeds = extract_seed_nodes(query)
        need_decomp = llm_obj.get("need_decomposition") is True and len(seeds) > 0

        return {
            "route": route,
            "why": f"{rule}; adaptive_budget(hop={adaptive_hop}, nodes={adaptive_node})",
            "need_decomposition": need_decomp,
            "decomposition": {
                "seed_nodes": seeds,
                "hop": adaptive_hop,
                "node_budget": adaptive_node,
                "edge_budget": adaptive_edge,
            },
        }
