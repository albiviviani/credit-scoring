from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError


class BaseCfg(BaseModel):
    # Allow field names starting with 'model_' without warnings
    model_config = {"protected_namespaces": ()}


class TrainingConfig(BaseCfg):
    data_source: str = Field(default="openml")
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    seed: int = Field(default=42, ge=0)
    cv_splits: int = Field(default=5, ge=3, le=20)
    model_candidates: List[str] = Field(
        default_factory=lambda: ["logreg", "rf", "lgbm"]
    )


class EvaluateConfig(BaseCfg):
    model_path: str
    metrics_path: str = "artifacts/metrics.json"
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    seed: int = 42


class InferConfig(BaseCfg):
    model_path: str
    input_csv: str
    output_csv: str
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RecordsPayload(BaseCfg):
    records: List[Dict[str, Any]]

    @field_validator("records")
    @classmethod
    def non_empty(cls, v: List[Dict[str, Any]]):
        if not v:
            raise ValueError("records is empty")
        return v


def validate_required_columns(cols: List[str], required: List[str]) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValidationError.from_exception_data(
            "RecordsPayload",
            [
                {
                    "loc": ("records", "columns"),
                    "msg": f"missing required columns: {missing}",
                    "type": "value_error",
                }
            ],
        )
