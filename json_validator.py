import json
import sys
from pathlib import Path
from datetime import datetime

REQUIRED_TOP_LEVEL_KEYS = {"info", "images", "annotations", "categories"}

DEFAULT_INFO = {
    "description": "Auto-generated COCO dataset",
    "url": "",
    "version": "1.0",
    "year": datetime.now().year,
    "contributor": "",
    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

def validate_and_fix_coco_json(json_path: str):
    path = Path(json_path)
    if not path.exists():
        print(f"File not found: {json_path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            return

    changed = False

    if "info" not in data:
        print("`info` section is missing. Adding default `info`.")
        data["info"] = DEFAULT_INFO
        changed = True
    elif not isinstance(data["info"], dict):
        print("`info` exists but is not a dictionary. Replacing with default.")
        data["info"] = DEFAULT_INFO
        changed = True
    else:
        print("`info` is present and valid.")

    missing_keys = REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing_keys:
        print(f"Missing top-level keys: {missing_keys}")
    else:
        print("All required top-level keys are present.")

    def check_list(key, expected_keys):
        if key not in data:
            print(f"`{key}` not found.")
            return
        if not isinstance(data[key], list):
            print(f"`{key}` is not a list.")
            return
        if len(data[key]) == 0:
            print(f"`{key}` is an empty list.")
        for i, item in enumerate(data[key]):
            for k in expected_keys:
                if k not in item:
                    print(f"`{key}[{i}]` missing key: {k}")

    check_list("images", {"id", "file_name", "height", "width"})
    check_list("annotations", {"id", "image_id", "category_id", "bbox"})
    check_list("categories", {"id", "name"})

    if changed:
        backup_path = json_path + ".bak"
        Path(backup_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Fixed JSON saved as backup to: {backup_path}")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Overwrote original JSON with fixed version: {json_path}")

    print("Validation complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_and_fix_coco.py <path_to_json>")
    else:
        validate_and_fix_coco_json(sys.argv[1])
