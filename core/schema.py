from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class GraphTask(BaseModel):
    id: str
    dataset_name: str
    query: str

    task_type: str

    graph_data: Optional[Dict[str, Any]] = None

    ground_truth: Optional[str] = None


class ExecutionStatus(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None
    runtime: float = 0.0