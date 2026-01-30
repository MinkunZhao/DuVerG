import json
import os
import yaml
from .base import BaseAgent


class CoderAgent(BaseAgent):
    def __init__(self, llm, name: str = "Coder"):
        super().__init__(name, llm)

        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)["coder"]

        knowledge_path = "config/task_knowledge.json"
        if os.path.exists(knowledge_path):
            with open(knowledge_path, "r", encoding="utf-8") as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = {}

    def generate_code(self, task_query, task_type, plan, graph_payload_hint, **kwargs):
        tkey = (task_type or "").lower()
        task_info = self.knowledge_base.get(tkey, self.knowledge_base.get("general", {}))

        diversity_hint = kwargs.get("diversity_hint", "")
        error_feedback = kwargs.get("error_feedback", "")

        knowledge_injection = f"""### Task-Specific Knowledge Injection:
- Definition: {task_info.get('definition', 'N/A')}
- Algorithm Guide: {task_info.get('algorithm_guideline', 'N/A')}
- Pitfalls to Avoid: {task_info.get('pitfalls', 'N/A')}
- Strict Output Format: {task_info.get('output_requirement', 'N/A')}
"""
        if diversity_hint:
            knowledge_injection += f"\n### Diversity Hint:\n{diversity_hint}\n"
        if error_feedback:
            knowledge_injection += f"\n### Previous Error Feedback:\n{error_feedback}\n"

        if "graph_file" in graph_payload_hint or "temporary JSON file" in graph_payload_hint:
            knowledge_injection += """
        ### CRITICAL CONSTRAINT:
        You MUST load the ENTIRE graph into memory using nx.node_link_graph().
        Do NOT use streaming or lazy loading. Do NOT use nx.read_edgelist() directly.
        The graph data is provided in the JSON format below - use it as-is.
        """

        plan_json = json.dumps(plan, ensure_ascii=False)

        warning_suppression = (
            "IMPORTANT: Start your code strictly with these lines to suppress dirty warnings:\n"
            "```python\n"
            "import warnings\n"
            "import logging\n"
            "warnings.filterwarnings('ignore')\n"
            "logging.getLogger().setLevel(logging.ERROR)\n"
            "```\n"
        )

        core_prompt = self.prompts["user_template"].format(
            query=task_query,
            task_type=task_type,
            plan_json=plan_json,
            graph_payload_hint=graph_payload_hint,
            error_feedback=error_feedback or "",
        )

        prompt = knowledge_injection + "\n\n" + warning_suppression + "\n" + core_prompt

        raw = self.llm.chat(
            messages=[
                {"role": "system", "content": self.prompts["system"]},
                {"role": "user", "content": prompt},
            ],
            json_mode=False,
            temperature=0.2,
        )
        return self.llm.extract_code(raw)
