import json
import yaml
from typing import Any, Dict

from .base import BaseAgent


class CriticAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__("Critic", llm)
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)["critic"]

    def consistency_check(self, query: str, out_a: str, out_b: str) -> Dict[str, Any]:
        prompt = self.prompts["consistency_template"].format(query=query, out_a=out_a[:2000], out_b=out_b[:2000])
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
            return obj
        except:
            return {"consistent": False, "why": "parse_error", "resolution": "need_tiebreaker", "tiebreaker_hint": ""}

    @staticmethod
    def cheap_normalize(s: str) -> str:
        return str(s).strip().lower().replace("\n", " ").replace("\t", " ")

    def outputs_roughly_equal(self, a: str, b: str) -> bool:
        na = self.cheap_normalize(a)
        nb = self.cheap_normalize(b)
        if na == nb:
            return True
        if na.replace(" ", "") == nb.replace(" ", ""):
            return True
        return False
