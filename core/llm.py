import re
import json
from openai import OpenAI, APIConnectionError, InternalServerError, RateLimitError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LLMEngine:
    def __init__(self, cfg):
        self.client = OpenAI(api_key=cfg['api_key'], base_url=cfg['base_url'])
        self.model = cfg['model_name']
        self.request_timeout = cfg.get('timeout', 90)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @retry(retry=retry_if_exception_type((InternalServerError, RateLimitError, APIConnectionError, APITimeoutError)),
           stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=10))
    def chat(self, messages, json_mode=False, temperature=0.1):
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.request_timeout
        }
        if json_mode: params["response_format"] = {"type": "json_object"}

        try:
            resp = self.client.chat.completions.create(**params)
            if hasattr(resp, 'usage') and resp.usage:
                self.total_prompt_tokens += resp.usage.prompt_tokens
                self.total_completion_tokens += resp.usage.completion_tokens
            return resp.choices[0].message.content
        except Exception as e:
            print(f"⚠️ LLM Error: {e}")
            raise e

    def extract_code(self, text):
        pattern = r"```python\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match: return match.group(1).strip()
        if "import " in text and "print(" in text:
            return text.replace("```", "").strip()
        return None