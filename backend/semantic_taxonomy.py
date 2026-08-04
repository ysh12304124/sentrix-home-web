from copy import deepcopy


OTHER = "其他或不确定"

PLACE_PRIMARY_TYPES = (
    "居住空间", "餐饮空间", "商业空间", "公园与花园", "滨水空间", "山地与自然景观",
    "街道与广场", "交通空间", "文化与展览", "运动与休闲", "演出与活动", "办公与学习",
    "医疗与公共服务", "宗教与纪念", "工业与工程", "住宿空间", "农场与乡村", OTHER,
)

OBJECT_PRIMARY_TYPES = (
    "食品与饮品", "餐具与容器", "电子设备", "家具与家居", "服饰与配件", "植物与花卉",
    "动物与宠物", "交通工具", "建筑与公共设施", "文字与标识", "玩具与娱乐", "书籍与文具",
    "礼物与纪念物", OTHER,
)

ATMOSPHERE_PRIMARY_TYPES = (
    "温馨", "热闹", "轻松", "平静", "安静", "活跃", "忙碌", "庄重", "节庆", "自然开阔", OTHER,
)

PLACE_DETAIL_TYPES = frozenset({
    "室内", "室外", "客厅", "卧室", "厨房", "阳台", "门口", "正餐", "咖啡或茶", "烘焙",
    "有餐桌", "露天座位", "商场", "超市", "市场或摊位", "展厅", "舞台", "候车区", "酒店房间",
    "多人停留", "开放空间", "自然环境",
})

OBJECT_DETAIL_TYPES = frozenset({
    "蛋糕", "水果", "饮料", "咖啡", "茶", "餐具", "杯子", "手机", "相机", "电脑", "桌面",
    "手持", "穿着", "摆放", "宠物", "汽车", "自行车", "书籍", "文具",
})

ATMOSPHERE_DETAIL_TYPES = frozenset({
    "明亮", "昏暗", "暖色光线", "冷色光线", "空间开阔", "空间拥挤", "多人聚集", "少人",
    "画面整洁", "画面繁杂", "庆祝活动", "观看活动", "休息状态", "自然光",
})

PLACE_PRIMARY_ALIASES = {
    "居住室内": "居住空间",
    "餐厅": "餐饮空间",
    "餐馆": "餐饮空间",
    "咖啡馆": "餐饮空间",
    "茶室": "餐饮空间",
    "园林公园": "公园与花园",
    "山地与自然": "山地与自然景观",
    "户外公共空间": "街道与广场",
    "交通出行空间": "交通空间",
    "文化展览": "文化与展览",
    "展览空间": "文化与展览",
    "演出活动空间": "演出与活动",
    "办公学习空间": "办公与学习",
    "工业与工程空间": "工业与工程",
}

OBJECT_PRIMARY_ALIASES = {
    "食物与饮品": "食品与饮品",
    "手机与移动设备": "电子设备",
    "展示与标识": "文字与标识",
    "餐具容器": "餐具与容器",
    "花卉与植物": "植物与花卉",
    "服饰与配件": "服饰与配件",
}

ATMOSPHERE_PRIMARY_ALIASES = {
    "欢快": "热闹",
    "喜悦": "热闹",
    "愉快": "轻松",
    "放松": "轻松",
    "宁静": "安静",
    "平和": "平静",
    "兴奋": "活跃",
}


def _text(value):
    return str(value or "").strip()


def _unique_text(values):
    result = []
    for value in values or []:
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return result


def _primary(value, choices, aliases):
    text = _text(value)
    normalized = aliases.get(text, text)
    return normalized if normalized in choices else OTHER


def _details(values, allowed, raw_labels):
    normalized = []
    for value in _unique_text(values):
        if value in allowed:
            normalized.append(value)
        else:
            raw_labels.append(value)
    return normalized


def normalize_semantic_analysis(analysis):
    """Normalize model semantic choices while preserving non-taxonomy labels."""
    source = deepcopy(analysis or {})
    semantic_input = source.get("semantic") if isinstance(source.get("semantic"), dict) else {}
    semantic_available = bool(
        semantic_input.get("available")
        if "available" in semantic_input
        else semantic_input or source.get("scene_type")
    )
    semantic = semantic_input
    raw_labels = {
        "place": _text(source.get("place")),
        "semantic_place": [],
        "objects": [],
        "atmosphere": [],
    }

    place_source = semantic.get("place") if isinstance(semantic.get("place"), dict) else {}
    place_value = place_source.get("primary") or source.get("scene_type")
    place_primary = _primary(place_value, PLACE_PRIMARY_TYPES, PLACE_PRIMARY_ALIASES)
    if _text(place_value) and _text(place_value) != place_primary:
        raw_labels["semantic_place"].append(_text(place_value))
    place_details_raw = []
    place_details = _details(place_source.get("details"), PLACE_DETAIL_TYPES, place_details_raw)
    raw_labels["semantic_place"].extend(place_details_raw)

    objects = []
    object_source = semantic.get("objects") if isinstance(semantic.get("objects"), list) else []
    for item in object_source:
        if not isinstance(item, dict):
            raw_labels["objects"].append(_text(item))
            continue
        detail_raw = []
        primary = _primary(item.get("primary"), OBJECT_PRIMARY_TYPES, OBJECT_PRIMARY_ALIASES)
        label = _text(item.get("label"))
        if _text(item.get("primary")) and _text(item.get("primary")) != primary:
            raw_labels["objects"].append(_text(item.get("primary")))
        details = _details(item.get("details"), OBJECT_DETAIL_TYPES, detail_raw)
        raw_labels["objects"].extend(detail_raw)
        if label or primary != OTHER or details:
            objects.append({"primary": primary, "label": label, "details": details})

    atmosphere_source = semantic.get("atmosphere") if isinstance(semantic.get("atmosphere"), dict) else {}
    atmosphere_raw = []
    atmosphere_labels = []
    for value in _unique_text(atmosphere_source.get("labels") or source.get("emotions")):
        primary = _primary(value, ATMOSPHERE_PRIMARY_TYPES, ATMOSPHERE_PRIMARY_ALIASES)
        if primary not in atmosphere_labels:
            atmosphere_labels.append(primary)
        if value != primary:
            atmosphere_raw.append(value)
    atmosphere_details_raw = []
    atmosphere_details = _details(atmosphere_source.get("details"), ATMOSPHERE_DETAIL_TYPES, atmosphere_details_raw)
    raw_labels["atmosphere"].extend(atmosphere_raw + atmosphere_details_raw)
    if not atmosphere_labels and (atmosphere_source.get("labels") or source.get("emotions")):
        atmosphere_labels = [OTHER]

    normalized = dict(source)
    normalized["semantic"] = {
        "available": semantic_available,
        "place": {"primary": place_primary, "details": place_details},
        "objects": objects,
        "atmosphere": {"labels": atmosphere_labels, "details": atmosphere_details},
    }
    normalized["scene_type"] = place_primary
    normalized["raw_labels"] = raw_labels
    return normalized
