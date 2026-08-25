import json
import ast
import math
import re
import yaml
from typing import Any, Dict

from .base import BaseAgent


class CriticAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__("Critic", llm)
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)["critic"]

    def consistency_check(self, query: str, task_type: str, status_a: bool, out_a: str,
                          status_b: bool, out_b: str) -> Dict[str, Any]:
        if status_a and not status_b:
            return {"consistent": False, "why": "candidate_b_failed", "resolution": "pick_a", "tiebreaker_hint": ""}
        if status_b and not status_a:
            return {"consistent": False, "why": "candidate_a_failed", "resolution": "pick_b", "tiebreaker_hint": ""}
        if status_a and status_b and self.outputs_roughly_equal(task_type, out_a, out_b):
            return {"consistent": True, "why": "normalized_match", "resolution": "pick_a", "tiebreaker_hint": ""}

        prompt = self.prompts["consistency_template"].format(
            query=query,
            answer_type=task_type,
            status_a="success" if status_a else "failed",
            out_a=str(out_a)[:2000],
            status_b="success" if status_b else "failed",
            out_b=str(out_b)[:2000],
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
            obj.setdefault("consistent", False)
            obj.setdefault("resolution", "need_tiebreaker")
            obj.setdefault("tiebreaker_hint", "")
            if obj.get("resolution") not in ("pick_a", "pick_b", "need_tiebreaker"):
                obj["resolution"] = "need_tiebreaker"
            return obj
        except:
            return {"consistent": False, "why": "parse_error", "resolution": "need_tiebreaker", "tiebreaker_hint": ""}

    def focused_check(self, query: str, task_type: str, out_a: str, out_b: str,
                      trace_a: str, trace_b: str, tiebreaker_hint: str) -> str:
        prompt = self.prompts["focused_check_template"].format(
            query=query,
            answer_type=task_type,
            out_a=str(out_a)[:2000],
            out_b=str(out_b)[:2000],
            tiebreaker_hint=str(tiebreaker_hint)[:1000],
            trace_a=str(trace_a)[:4000],
            trace_b=str(trace_b)[:4000],
        )
        try:
            raw = self.llm.chat(
                messages=[
                    {"role": "system", "content": self.prompts["system"]},
                    {"role": "user", "content": prompt},
                ],
                json_mode=False,
                temperature=0.1,
            )
            match = re.search(r"\b(pick_a|pick_b|unresolved)\b", str(raw).lower())
            return match.group(1) if match else "unresolved"
        except Exception:
            return "unresolved"

    @staticmethod
    def _sequence(s: str):
        text = str(s or "").strip()
        match = re.search(r"\[[^\]]*\]", text)
        if match:
            try:
                value = ast.literal_eval(match.group(0))
                if isinstance(value, (list, tuple, set)):
                    return [str(x).strip().lower() for x in value]
            except Exception:
                pass
        numbers = re.findall(r"-?\d+", text)
        return numbers if numbers else None

    @staticmethod
    def _boolean(s: str):
        words = set(re.findall(r"[a-z]+", str(s or "").lower()))
        if words.intersection({"yes", "true"}):
            return True
        if words.intersection({"no", "false", "impossible"}):
            return False
        return None

    @staticmethod
    def _scalar(s: str):
        numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(s or "").replace(",", ""))
        return float(numbers[-1]) if numbers else None

    @staticmethod
    def _kind(task_type: str) -> str:
        task = str(task_type or "").lower()
        if any(x in task for x in ("shortest_path", "distance", "topology", "dfs", "path")):
            return "path"
        if any(x in task for x in ("connected_nodes", "disconnected_nodes", "neighbors", "predecessor", "page_rank")):
            return "set"
        if any(x in task for x in ("cycle", "reachability", "connectivity", "substructure", "edge_existence", "bipartite")):
            return "boolean"
        if any(x in task for x in ("flow", "count", "degree", "diameter", "jaccard", "clustering", "coefficient")):
            return "scalar"
        return "text"

    def normalized_equal(self, task_type: str, a: str, b: str) -> bool:
        kind = self._kind(task_type)
        if kind == "boolean":
            va, vb = self._boolean(a), self._boolean(b)
            return va is not None and va == vb
        if kind == "scalar":
            va, vb = self._scalar(a), self._scalar(b)
            return va is not None and vb is not None and math.isclose(va, vb, rel_tol=0.01, abs_tol=0.001)
        if kind in ("path", "set"):
            empty_values = {"", "none", "empty", "no nodes", "no nodes.", "[]", "{}"}
            va = [] if kind == "set" and str(a or "").strip().lower() in empty_values else self._sequence(a)
            vb = [] if kind == "set" and str(b or "").strip().lower() in empty_values else self._sequence(b)
            if va is None or vb is None:
                return False
            return va == vb if kind == "path" else set(va) == set(vb)
        va = re.sub(r"[^a-z0-9]", "", str(a or "").lower())
        vb = re.sub(r"[^a-z0-9]", "", str(b or "").lower())
        return va == vb

    def outputs_roughly_equal(self, task_type: str, a: str, b: str, status_a: bool = True,
                              status_b: bool = True) -> bool:
        return status_a and status_b and self.normalized_equal(task_type, a, b)
