"""Schema resolution helpers for datasets."""
import json
import os
from typing import Type

from pydantic import BaseModel
from app.backend.model.debt_collection import GraphNodeEnum


def ensure_demo_schema_exists() -> str:
    """Ensure the default demo schema exists and return its path."""
    os.makedirs("schemas", exist_ok=True)
    schema_path = "schemas/demo.json"
    if not os.path.exists(schema_path):
        demo_schema = {
            "Nodes": [
                "person",
                "location",
                "organization",
                "event",
                "object",
                "concept",
                "time_period",
                "creative_work",
                "biological_entity",
                "natural_phenomenon",
            ],
            "Relations": [
                "is_a",
                "part_of",
                "located_in",
                "created_by",
                "used_by",
                "participates_in",
                "related_to",
                "belongs_to",
                "influences",
                "precedes",
                "arrives_in",
                "comparable_to",
            ],
            "Attributes": [
                "name",
                "date",
                "size",
                "type",
                "description",
                "status",
                "quantity",
                "value",
                "position",
                "duration",
                "time",
            ],
        }
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(demo_schema, f, indent=2)
    return schema_path


def get_schema_path_for_dataset(dataset_name: str) -> str:
    """Return dataset-specific schema if present; otherwise fallback to demo."""
    if dataset_name and dataset_name != "demo":
        ds_schema = f"schemas/{dataset_name}/schema.json"
        if os.path.exists(ds_schema):
            return ds_schema
    return ensure_demo_schema_exists()

def serialize_pydantic_schema(model: Type[BaseModel]) -> str:
    """序列化 Pydantic schema，并把 GraphNodeEnum 的中文说明注入
    `x-enum-descriptions`，使 DeepSeek / GLM 在 prompt 中直接看到每个
    枚举值的适用场景与决策建议。"""
    schema = model.model_json_schema()

    for defn in schema.get("$defs", {}).values():
        values = defn.get("enum")
        if not values:
            continue
        descs = {v: d for v in values if (d := GraphNodeEnum.description_for(v))}
        if descs:
            defn["x-enum-descriptions"] = descs

    return json.dumps(schema, ensure_ascii=False, indent=2)