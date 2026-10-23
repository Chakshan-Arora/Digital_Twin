"""Export FastAPI OpenAPI JSON schema to docs/openapi.json."""

import json
from pathlib import Path

from hdt.api.app import app


def export_openapi() -> Path:
    """Generate OpenAPI schema and save to docs/openapi.json."""
    openapi_schema = app.openapi()
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = docs_dir / "openapi.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)
    return out_path


if __name__ == "__main__":
    out_file = export_openapi()
    print(f"OpenAPI schema successfully exported to {out_file}")
