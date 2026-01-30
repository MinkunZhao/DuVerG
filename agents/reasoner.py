import yaml
from typing import Optional

from .base import BaseAgent


class NeuralReasonerAgent(BaseAgent):
    def __init__(self, llm, name: str = "Reasoner"):
        super().__init__(name, llm)
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)["reasoner"]

    def answer(self, query: str, task_type: str, graph_hint: str, temperature: float = 0.2) -> str:
        prompt = self.prompts["user_template"].format(query=query, task_type=task_type, graph_hint=graph_hint)
        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": self.prompts["system"]},
                {"role": "user", "content": prompt},
            ],
            json_mode=False,
            temperature=temperature,
        )
        return (resp or "").strip()

