import json
import os
from typing import Any, Dict, List


class JSONLogger:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._initialize_file()

    def _initialize_file(self) -> None:
        dir_path = os.path.dirname(self.file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if (not os.path.exists(self.file_path)) or os.path.getsize(self.file_path) == 0:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def log(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data: List[Dict[str, Any]] = json.load(f)
            if not isinstance(data, list):
                data = []
            data.append(record)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ JSONLogger Error: {e}")
