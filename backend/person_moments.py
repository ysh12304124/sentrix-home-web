"""Evidence-bound person moment extraction from numbered photo previews.

The pipeline only writes PersonMoment rows that can trace back to an original
Asset, Observation and Event. Attribute language (health, income, religion,
ethnicity, politics, sexual orientation, age/generation/gender presentation)
is filtered before anything reaches the memory store.
"""

import tempfile
from pathlib import Path

PERSON_MOMENT_PROMPT = (
    "你是家庭相册中的人物瞬间观察器。图片中的 P1/P2 标签仅用于定位人物。\n"
    "只描述画面可见动作、互动、参与方式、表情状态和低风险角色线索。\n"
    "不得猜姓名；不得输出健康、收入、宗教、民族、政治、性取向；不得把年龄、\n"
    "代际或性别呈现保存为事实。看不清时留空。\n"
    "严格返回 JSON：\n"
    '{"moments":[{"label":"P1","action_text":"","interaction_labels":[],\n'
    '"interaction_text":"","participation_style":"","visible_affect":"",\n'
    '"social_role_cues":[],"narrative_note":"","confidence":0.0}]}'
)

PARTICIPATION_STYLES = {"组织", "照顾", "陪伴", "共同参与", "旁观", "拍摄", "无法判断"}

SENSITIVE_TERMS = {
    "健康", "生病", "疾病", "癌症", "抑郁", "焦虑",
    "收入", "富有", "贫穷", "有钱", "工资", "负债",
    "宗教", "佛教", "基督教", "伊斯兰",
    "民族", "汉族", "回族", "藏族", "维吾尔",
    "政治", "党员", "政见",
    "性取向", "同性恋",
    "年龄", "岁数", "老人", "中年", "青年", "老年",
    "性别", "男性", "女性", "辈分", "代际",
}

PERSONALITY_TERMS = {
    "内向", "外向", "开朗", "乐观", "悲观", "易怒",
    "敏感", "固执", "性格", "脾气", "温和", "暴躁",
}


def _contains_any(value, terms):
    return any(term in value for term in terms)


def normalize_person_moments(parsed, labels):
    labels = set(labels)
    moments = []
    raw_moments = parsed.get("moments") if isinstance(parsed, dict) else []
    if not isinstance(raw_moments, list):
        raw_moments = []
    for item in raw_moments:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if label not in labels:
            continue
        action = str(item.get("action_text") or "").strip()
        if not action:
            continue
        style = str(item.get("participation_style") or "").strip() or "无法判断"
        if style not in PARTICIPATION_STYLES:
            style = "无法判断"
        cues = []
        for cue in item.get("social_role_cues") or []:
            value = str(cue).strip()
            if value and not _contains_any(value, SENSITIVE_TERMS):
                cues.append(value)
        affect = str(item.get("visible_affect") or "").strip()
        if _contains_any(affect, SENSITIVE_TERMS | PERSONALITY_TERMS):
            affect = ""
        interaction_labels = []
        for raw in item.get("interaction_labels") or []:
            value = str(raw).strip()
            if value and value not in interaction_labels:
                interaction_labels.append(value)
        moments.append({
            "label": label,
            "action_text": action,
            "interaction_labels": interaction_labels,
            "interaction_text": str(item.get("interaction_text") or ""),
            "participation_style": style,
            "visible_affect": affect,
            "social_role_cues": list(dict.fromkeys(cues)),
            "narrative_note": str(item.get("narrative_note") or ""),
            "confidence": float(item.get("confidence") or 0),
        })
    return moments


def render_numbered_preview(asset_path, faces):
    """Draw high-contrast face boxes and P1/P2 labels on an EXIF-corrected image.

    The preview is written to a temporary JPEG and must be deleted by the caller;
    it is never saved as an Asset or long-lived file.
    """
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(asset_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_map = {}
    for index, face in enumerate(faces, start=1):
        label = f"P{index}"
        bbox = list(face.get("bbox") or [])
        if len(bbox) >= 4:
            x, y, width, height = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            draw.rectangle([x, y, x + width, y + height], outline="red", width=3)
            draw.text((x, max(0.0, y - 14)), label, fill="red")
        else:
            draw.text((10, 10 + index * 14), label, fill="red")
        label_map[label] = {
            "face_instance_id": face.get("face_instance_id"),
            "person_id": face.get("person_id"),
            "cluster_id": face.get("cluster_id"),
            "observation_id": face.get("observation_id"),
            "event_id": face.get("event_id"),
        }
    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    image.save(temp.name, format="JPEG", quality=90)
    temp.close()
    return temp.name, label_map


class PersonMomentExtractor:
    def __init__(self, store, gamma):
        self.store = store
        self.gamma = gamma

    def extract(self, scope_id, run_id, selections):
        by_asset = {}
        for selection in selections:
            by_asset.setdefault(selection["asset_id"], []).append(selection)
        moments_created = 0
        failures = 0
        for asset_id, items in by_asset.items():
            asset = self.store.get_asset(asset_id)
            if not asset or not asset.get("path"):
                continue
            faces = []
            for item in items:
                face = self.store.get_face_instance(item["face_instance_id"])
                bbox = (face or {}).get("bbox_json") or []
                faces.append({
                    "face_instance_id": item["face_instance_id"],
                    "person_id": item["person_id"],
                    "cluster_id": item["cluster_id"],
                    "observation_id": item["observation_id"],
                    "event_id": item["event_id"],
                    "bbox": bbox,
                })
            preview_path = None
            try:
                preview_path, label_map = render_numbered_preview(asset["path"], faces)
                context = {
                    "asset_id": asset_id,
                    "scope_id": scope_id,
                    "captured_at": asset.get("captured_at"),
                }
                result = self.gamma.analyze_person_moments(
                    preview_path, sorted(label_map), context
                )
                for moment in result.get("moments") if isinstance(result, dict) else []:
                    info = label_map.get(moment.get("label"))
                    if not info or not info.get("person_id"):
                        continue
                    targets = []
                    for target_label in moment.get("interaction_labels") or []:
                        target = label_map.get(target_label)
                        if target and target.get("person_id") and \
                                self._same_scope(target["person_id"], scope_id):
                            targets.append(target["person_id"])
                    payload = {
                        "person_id": info["person_id"],
                        "cluster_id": info["cluster_id"],
                        "event_id": info["event_id"],
                        "observation_id": info["observation_id"],
                        "asset_id": asset_id,
                        "face_instance_id": info["face_instance_id"],
                        "action_text": moment.get("action_text") or "",
                        "interaction_target_ids": targets,
                        "interaction_text": moment.get("interaction_text") or "",
                        "participation_style": moment.get("participation_style") or "无法判断",
                        "visible_affect": moment.get("visible_affect") or "",
                        "social_role_cues": moment.get("social_role_cues") or [],
                        "narrative_note": moment.get("narrative_note") or "",
                        "confidence": float(moment.get("confidence") or 0),
                        "model_name": getattr(self.gamma, "model", "unknown"),
                        "prompt_version": "person-moment-v1",
                        "run_id": run_id,
                    }
                    if self._same_scope(info["person_id"], scope_id):
                        self.store.upsert_person_moment(payload)
                        moments_created += 1
            except Exception:
                failures += 1
                self._record_failure(run_id, failures)
            finally:
                if preview_path:
                    Path(preview_path).unlink(missing_ok=True)
        return {"moments": moments_created, "failures": failures}

    def _same_scope(self, person_id, scope_id):
        return self.store._scope_for("entity", person_id) == scope_id

    def _record_failure(self, run_id, failures):
        run = self.store.get_person_insight_run(run_id)
        stats = dict(run.get("stats") or {}) if run else {}
        stats["moment_failures"] = failures
        self.store.update_person_insight_run(run_id, stats=stats)
