import json
import yaml
from typing import Any, Dict, List

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__("Planner", llm)
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)["planner"]

    def plan(self, query: str, task_type: str, graph_stats: Dict[str, Any], graph_hint: str) -> Dict[str, Any]:
        prompt = self.prompts["user_template"].format(
            query=query,
            task_type=task_type,
            graph_stats=json.dumps(graph_stats, ensure_ascii=False),
            graph_hint=graph_hint,
        )
        raw = self.llm.chat(
            messages=[
                {"role": "system", "content": self.prompts["system"]},
                {"role": "user", "content": prompt},
            ],
            json_mode=True,
            temperature=0.1,
        )
        try:
            obj = json.loads(raw)
            obj.setdefault("algorithm_family", "general")
            obj.setdefault("key_steps", [])
            obj.setdefault("defensive_checks", [])
            return obj
        except:
            return {
                "algorithm_family": "general",
                "key_steps": ["construct graph", "apply standard algorithm", "print final answer"],
                "defensive_checks": ["handle missing nodes", "wrap networkx calls in try/except"],
            }
