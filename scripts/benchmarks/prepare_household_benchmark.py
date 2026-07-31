"""Prepare an evaluation-only manifest from the three household albums."""

import argparse
import json
from pathlib import Path


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_path(album):
    candidates = (album / "metadata.json", album / "metadata" / "metadata.json")
    return next((path for path in candidates if path.is_file()), None)


def _location(value):
    latitude = value.get("latitude")
    longitude = value.get("longitude")
    if latitude is None or longitude is None:
        return None
    return f"{float(latitude):.6f},{float(longitude):.6f}"


def _album_record(album):
    images_dir = album / "images"
    image_names = sorted(path.name for path in images_dir.iterdir() if path.is_file())
    metadata_path = _metadata_path(album)
    metadata = _read_json(metadata_path) if metadata_path else {}
    face_path = album / "faceid" / "face_info_cn.json"
    face_info = _read_json(face_path) if face_path.is_file() else {}
    queries = _read_json(album / "query.json") if (album / "query.json").is_file() else []
    image_set = set(image_names)
    metadata_set = set(metadata)
    face_map = face_info.get("image_to_face_ids", {})
    face_set = set(face_map)

    imports = []
    for file_name in image_names:
        item = metadata.get(file_name) or {}
        imports.append({
            "file_name": file_name,
            "relative_path": f"{album.name}/images/{file_name}",
            "captured_at": item.get("time") or None,
            "captured_location": _location(item),
            "source_album_id": album.name,
            "scope_id": album.name,
        })

    evaluation_queries = []
    for query in queries:
        ground_truth = [name for name in query.get("ground_truth", []) if name in image_set]
        evaluation_queries.append({
            "query_cn": query.get("query_cn", ""),
            "query_en": query.get("query_en", ""),
            "ground_truth": ground_truth,
            "ground_truth_count": len(ground_truth),
            "missing_ground_truth": [name for name in query.get("ground_truth", []) if name not in image_set],
            "dimensions": {
                key: query.get(key)
                for key in ("Location", "Time", "Person", "Object", "Concept", "Genre", "Source")
            },
        })

    return {
        "scope_id": album.name,
        "name": album.name,
        "import": {"files": imports},
        "diagnostics": {
            "metadata_path": str(metadata_path.relative_to(album)) if metadata_path else None,
            "metadata_missing_images": sorted(metadata_set - image_set),
            "face_missing_images": sorted(face_set - image_set),
            "images_without_metadata": sorted(image_set - metadata_set),
            "images_without_face_map": sorted(image_set - face_set),
        },
        "evaluation": {
            "face_id_to_nicknames": face_info.get("face_id_to_nicknames", {}),
            "image_to_face_ids": {name: face_map[name] for name in sorted(face_set & image_set)},
            "queries": evaluation_queries,
        },
    }


def prepare_benchmark(source_root):
    root = Path(source_root).expanduser().resolve()
    spaces = [_album_record(album) for album in sorted(root.iterdir()) if album.is_dir() and album.name.startswith("album")]
    if not spaces:
        raise ValueError(f"no album directories found under {root}")
    return {
        "version": 1,
        "source_root": str(root),
        "import_contract": "source image + capture time/location + album scope only",
        "spaces": spaces,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = prepare_benchmark(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "spaces": [{"id": item["scope_id"], "files": len(item["import"]["files"]), "queries": len(item["evaluation"]["queries"])} for item in manifest["spaces"]]
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
