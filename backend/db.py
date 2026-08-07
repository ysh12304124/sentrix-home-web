import json
import math
import re
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone, timedelta

from .semantic_taxonomy import ATMOSPHERE_PRIMARY_ALIASES, ATMOSPHERE_PRIMARY_TYPES, OTHER, normalize_semantic_analysis


def make_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_time(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def json_value(value, fallback):
    return json.dumps(value if value is not None else fallback, ensure_ascii=False)


def dedupe_json_values(values):
    """Deduplicate scalar or structured model outputs without losing dictionaries."""
    result = []
    seen = set()
    for value in values or []:
        try:
            marker = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


CLOTHING_NORMALIZATION = {
    "深色西装外套": "西装外套",
    "黑色西装外套": "西装外套",
    "西装": "西装外套",
    "西装外套": "西装外套",
    "深色领带": "领带",
    "红色领带": "领带",
    "蓝色领带": "领带",
    "条纹领带": "领带",
    "带花纹的领带": "领带",
    "有花纹的棕色系领带": "领带",
    "白色衬衫": "衬衫",
    "浅色衬衫": "衬衫",
    "浅蓝色衬衫": "衬衫",
}

OBJECT_CATEGORIES = {
    "食物": ("蛋糕", "餐", "饭", "水果", "面包", "饮料", "咖啡", "茶", "酒"),
    "交通工具": ("自行车", "汽车", "车辆", "火车", "飞机", "摩托", "轮船", "船"),
    "宠物": ("猫", "狗", "鸟", "兔", "宠物"),
    "电子设备": ("手机", "相机", "电脑", "屏幕", "耳机", "麦克风", "电视"),
    "文具与出版物": ("书", "笔", "证书", "毕业证", "杂志"),
    "服饰": ("衣", "帽", "鞋", "包", "眼镜", "领带"),
}

TRIP_MIN_GPS_DISPLACEMENT_KM = 50.0
DISPLAY_COORDINATE_RE = re.compile(
    r"(?:GPS(?:坐标)?|坐标|经纬度)?\s*[+-]?\d{1,3}(?:\.\d+)?\s*[,，]\s*[+-]?\d{1,3}(?:\.\d+)?"
)

MOOD_NORMALIZATION = {
    "平静": "平静",
    "宁静": "平静",
    "平和": "平静",
    "安详": "平静",
    "淡定": "平静",
    "表情平淡": "平静",
    "愉快": "喜悦",
    "愉悦": "喜悦",
    "喜悦": "喜悦",
    "温馨": "温馨",
    "微笑": "喜悦",
    "面带微笑": "喜悦",
    "欢快": "喜悦",
    "兴奋": "兴奋",
    "兴奋感": "兴奋",
    "轻松": "放松",
    "放松": "放松",
    "悠闲": "放松",
    "专注": "专注",
    "好奇": "好奇",
    "警觉": "警觉",
}

SEMANTIC_ENTITY_EQUIVALENTS = {
    "place": {
        "湖边": "滨水区域",
        "水边": "滨水区域",
        "河边": "滨水区域",
        "河岸": "滨水区域",
        "湖畔": "滨水区域",
        "博物馆": "展览空间",
        "展厅": "展览空间",
        "展览厅": "展览空间",
        "展览馆": "展览空间",
        "美术馆": "展览空间",
        "画廊": "展览空间",
    },
    "emotion": MOOD_NORMALIZATION,
}

SEMANTIC_PLACE_CONCEPTS = (
    ("滨水空间", ("湖", "河", "海", "港", "码头", "水域", "水边", "水池", "滨水")),
    ("展览空间", ("博物馆", "展厅", "展览", "美术馆", "画廊", "展柜", "科普馆")),
    ("餐饮空间", ("餐厅", "餐馆", "餐饮", "厨房", "餐桌", "咖啡", "烘焙", "快餐", "茶室")),
    ("园林与公园", ("园林", "公园", "花园", "植物园", "温室", "果园", "庭院")),
    ("商业空间", ("商店", "商场", "购物", "店铺", "超市", "摊位")),
    ("交通出行空间", ("机场", "地铁", "车厢", "停机坪", "登机桥", "车站", "高架", "公路")),
    ("演出活动空间", ("剧场", "舞台", "演艺", "活动现场")),
    ("居住室内空间", ("卧室", "客厅", "室内", "房间", "走廊", "门口", "室内环境")),
    ("户外公共空间", ("广场", "街道", "户外", "室外", "景区", "道路", "观景")),
)

SEMANTIC_OBJECT_CONCEPTS = (
    ("手机与移动设备", ("手机", "平板", "相机")),
    ("展示与标识", ("海报", "横幅", "宣传", "标牌", "展板", "告示", "路标", "灯牌", "菜单")),
    ("餐具与容器", ("碗", "杯", "盘", "勺", "叉", "筷", "锅", "茶具", "餐具", "容器")),
    ("食物与饮品", ("蛋糕", "甜点", "饮料", "面条", "沙拉", "肉", "虾", "玉米", "番茄", "薯条", "牛角", "饭", "菜")),
    ("花卉与植物", ("花", "树", "草", "绿植", "盆栽", "棕榈", "樱花", "梅花")),
    ("建筑与工程", ("建筑", "房屋", "大坝", "高塔", "电线杆", "厂房", "机械", "起重机", "桥", "围墙")),
    ("围栏与公共设施", ("围栏", "栏杆", "护栏", "路灯", "扶手", "井盖")),
    ("交通工具", ("汽车", "车辆", "自行车", "摩托", "飞机", "船", "轮渡", "公交", "三轮车")),
    ("服饰与配件", ("背包", "项链", "戒指", "眼镜", "手表", "鞋", "钥匙", "项圈")),
    ("毛绒与玩具", ("毛绒", "玩偶", "气球")),
)


def normalize_clothing(value):
    text = str(value or "").strip()
    return CLOTHING_NORMALIZATION.get(text, text)


def time_semantics(value):
    """Return deterministic calendar labels for an EXIF-backed timestamp."""
    if value.hour < 5:
        part_of_day = "凌晨"
    elif value.hour < 8:
        part_of_day = "清晨"
    elif value.hour < 12:
        part_of_day = "上午"
    elif value.hour < 17:
        part_of_day = "午后"
    elif value.hour < 20:
        part_of_day = "傍晚"
    else:
        part_of_day = "夜晚"
    season = "春" if value.month in (3, 4, 5) else "夏" if value.month in (6, 7, 8) else "秋" if value.month in (9, 10, 11) else "冬"
    return {"date": value.date().isoformat(), "year": value.year, "month": value.month, "season": season, "part_of_day": part_of_day}


def object_category(label):
    text = str(label or "").strip()
    for category, terms in OBJECT_CATEGORIES.items():
        if any(term in text for term in terms):
            return category
    return "其他"


def normalize_mood(value):
    return MOOD_NORMALIZATION.get(str(value or "").strip())


def parse_gps_place(value):
    """Parse an EXIF-style latitude,longitude place only when it is usable."""
    try:
        latitude, longitude = (float(part.strip()) for part in str(value or "").split(",", 1))
    except (TypeError, ValueError):
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    if latitude == 0 and longitude == 0:
        return None
    return latitude, longitude


def gps_distance_km(first, second):
    """Return the great-circle distance in kilometres between two GPS points."""
    latitude_one, longitude_one = map(math.radians, first)
    latitude_two, longitude_two = map(math.radians, second)
    delta_latitude = latitude_two - latitude_one
    delta_longitude = longitude_two - longitude_one
    a = math.sin(delta_latitude / 2) ** 2 + math.cos(latitude_one) * math.cos(latitude_two) * math.sin(delta_longitude / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


class MemoryStore:
    _schema_locks = {}
    _schema_locks_guard = threading.Lock()

    @classmethod
    def _schema_lock(cls, path):
        with cls._schema_locks_guard:
            return cls._schema_locks.setdefault(str(path), threading.Lock())

    def __init__(self, path):
        self.path = str(path)
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        with self._schema_lock(path):
            self.connection.execute("PRAGMA busy_timeout = 30000")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema()

    def get_setting(self, key, default=None):
        row = self._row("SELECT value FROM runtime_settings WHERE key = ?", (key,))
        return row["value"] if row else default

    def set_setting(self, key, value):
        self.connection.execute(
            "INSERT INTO runtime_settings(key,value,updated_at) VALUES (?,?,datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, str(value)),
        )
        self.connection.commit()

    def list_settings(self):
        rows = self._rows("SELECT key, value, updated_at FROM runtime_settings ORDER BY key")
        return [{"key": r["key"], "value": r["value"], "updated_at": r["updated_at"]} for r in rows]

    def close(self):
        self.connection.close()

    def apply_authorized_revision(self, *, proposal_id, confirmation_token, actor):
        """Plan §6 Memory Kernel entry point.

        Thin wrapper over :class:`MemoryCorrections.apply` so the plan-facing
        API surface lives on ``MemoryStore``.  The correction service still
        owns audit + revision log semantics.  Imported lazily to avoid a
        circular dependency between ``db.py`` and ``memory_corrections.py``.
        """
        from .memory_corrections import MemoryCorrections

        service = MemoryCorrections(self)
        return service.apply(proposal_id=proposal_id,
                              confirmation_token=confirmation_token,
                              actor=actor)

    def propose_memory_correction(self, *, scope_id, actor, target_type, target_id,
                                   changed_fields, evidence_ids=None, request_id=None):
        """Plan §6 propose entry — mirror of ``apply_authorized_revision``."""
        from .memory_corrections import MemoryCorrections

        service = MemoryCorrections(self)
        return service.propose(scope_id=scope_id, actor=actor,
                                target_type=target_type, target_id=target_id,
                                changed_fields=changed_fields,
                                evidence_ids=evidence_ids,
                                request_id=request_id)

    def _create_schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_spaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'household',
                source_path TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ingest_batches (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                batch_id TEXT,
                file_name TEXT NOT NULL,
                media_type TEXT NOT NULL,
                path TEXT NOT NULL,
                mime_type TEXT,
                size_bytes INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'queued',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                source_owner_id TEXT,
                source_owner_label TEXT,
                source_device_id TEXT,
                source_album_id TEXT,
                source_confidence REAL NOT NULL DEFAULT 0,
                content_sha256 TEXT,
                captured_at TEXT,
                captured_location TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                asset_id TEXT NOT NULL REFERENCES assets(id),
                captured_at TEXT,
                source_type TEXT NOT NULL,
                caption TEXT,
                activity TEXT,
                place TEXT,
                people_json TEXT NOT NULL DEFAULT '[]',
                objects_json TEXT NOT NULL DEFAULT '[]',
                ocr_text TEXT NOT NULL DEFAULT '',
                event_type TEXT,
                transcript TEXT,
                confidence REAL NOT NULL DEFAULT 0,
                raw_json TEXT NOT NULL DEFAULT '{}',
                canonical_json TEXT NOT NULL DEFAULT '{}',
                source_owner_id TEXT,
                inferred_captured_by TEXT,
                clothing_json TEXT NOT NULL DEFAULT '[]',
                spatial_relations_json TEXT NOT NULL DEFAULT '[]',
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                title TEXT NOT NULL,
                event_type TEXT,
                time_start TEXT,
                time_end TEXT,
                place TEXT,
                activity TEXT,
                summary TEXT,
                participants_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                aggregation_score REAL NOT NULL DEFAULT 0,
                aggregation_breakdown_json TEXT NOT NULL DEFAULT '{}',
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_revisions (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL REFERENCES events(id),
                field_name TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                source TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_observations (
                event_id TEXT NOT NULL REFERENCES events(id),
                observation_id TEXT NOT NULL REFERENCES observations(id),
                PRIMARY KEY(event_id, observation_id)
            );
            CREATE TABLE IF NOT EXISTS event_participants (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL REFERENCES events(id),
                person_id TEXT NOT NULL REFERENCES entities(id),
                role TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(event_id, person_id, role)
            );
            CREATE TABLE IF NOT EXISTS persons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                confidence REAL NOT NULL DEFAULT 0,
                source_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_entities (
                event_id TEXT NOT NULL REFERENCES events(id),
                entity_id TEXT NOT NULL REFERENCES entities(id),
                relation TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(event_id, entity_id, relation)
            );
            CREATE TABLE IF NOT EXISTS trips (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                time_start TEXT,
                time_end TEXT,
                event_ids_json TEXT NOT NULL DEFAULT '[]',
                place_names_json TEXT NOT NULL DEFAULT '[]',
                companion_ids_json TEXT NOT NULL DEFAULT '[]',
                trip_type TEXT,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trip_revisions (
                id TEXT PRIMARY KEY,
                trip_id TEXT NOT NULL REFERENCES trips(id),
                action TEXT NOT NULL,
                old_value_json TEXT NOT NULL DEFAULT '{}',
                new_value_json TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                entity_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                family_role TEXT,
                summary TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entity_revisions (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL REFERENCES entities(id),
                field_name TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                source TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entity_properties (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL REFERENCES entities(id),
                property_key TEXT NOT NULL,
                value_json TEXT NOT NULL DEFAULT 'null',
                source TEXT NOT NULL DEFAULT 'derived',
                confidence REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                supersedes_property_id TEXT REFERENCES entity_properties(id),
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entity_merge_candidates (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                entity_type TEXT NOT NULL,
                entity_ids_json TEXT NOT NULL DEFAULT '[]',
                suggested_name TEXT NOT NULL,
                rationale_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL DEFAULT 0,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'pending',
                target_entity_id TEXT REFERENCES entities(id),
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(scope_id, entity_type, entity_ids_json, suggested_name)
            );
            CREATE TABLE IF NOT EXISTS face_clusters (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                status TEXT NOT NULL DEFAULT 'pending',
                entity_id TEXT REFERENCES entities(id),
                representative_embedding_json TEXT NOT NULL DEFAULT '[]',
                member_count INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS face_instances (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL REFERENCES assets(id),
                observation_id TEXT NOT NULL REFERENCES observations(id),
                cluster_id TEXT REFERENCES face_clusters(id),
                bbox_json TEXT NOT NULL DEFAULT '[]',
                embedding_json TEXT NOT NULL DEFAULT '[]',
                detection_confidence REAL NOT NULL DEFAULT 0,
                quality REAL NOT NULL DEFAULT 0,
                pose_json TEXT NOT NULL DEFAULT '[]',
                area_ratio REAL NOT NULL DEFAULT 0,
                sharpness REAL NOT NULL DEFAULT 0,
                pose_bucket TEXT NOT NULL DEFAULT 'unknown',
                embedding_model TEXT NOT NULL DEFAULT 'unknown',
                embedding_version TEXT NOT NULL DEFAULT 'unknown',
                embedding_quality_signal REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS face_prototypes (
                id TEXT PRIMARY KEY,
                cluster_id TEXT NOT NULL REFERENCES face_clusters(id),
                face_instance_id TEXT NOT NULL REFERENCES face_instances(id),
                pose_bucket TEXT NOT NULL DEFAULT 'unknown',
                embedding_json TEXT NOT NULL DEFAULT '[]',
                quality REAL NOT NULL DEFAULT 0,
                model_name TEXT NOT NULL DEFAULT 'unknown',
                model_version TEXT NOT NULL DEFAULT 'unknown',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(cluster_id, pose_bucket)
            );
            CREATE TABLE IF NOT EXISTS person_appearance_evidence (
                id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL REFERENCES entities(id),
                face_instance_id TEXT NOT NULL REFERENCES face_instances(id),
                observation_id TEXT NOT NULL REFERENCES observations(id),
                asset_id TEXT NOT NULL REFERENCES assets(id),
                crop_bbox_json TEXT NOT NULL DEFAULT '[]',
                clothing_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0,
                model_name TEXT NOT NULL DEFAULT 'unknown',
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(person_id, face_instance_id, model_name)
            );
            CREATE TABLE IF NOT EXISTS entity_mentions (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL REFERENCES entities(id),
                observation_id TEXT NOT NULL REFERENCES observations(id),
                face_instance_id TEXT REFERENCES face_instances(id),
                confidence REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(entity_id, observation_id, face_instance_id)
            );
            CREATE TABLE IF NOT EXISTS entity_observations (
                entity_id TEXT NOT NULL REFERENCES entities(id),
                observation_id TEXT NOT NULL REFERENCES observations(id),
                confidence REAL NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'observation_extraction',
                created_at TEXT NOT NULL,
                PRIMARY KEY(entity_id, observation_id)
            );
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                subject_entity_id TEXT NOT NULL REFERENCES entities(id),
                predicate TEXT NOT NULL,
                object_entity_id TEXT NOT NULL REFERENCES entities(id),
                status TEXT NOT NULL DEFAULT 'pending',
                confidence REAL NOT NULL DEFAULT 0,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                supersedes_relationship_id TEXT REFERENCES relationships(id),
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_vectors (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                space TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                model_name TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(space, source_type, source_id, model_name)
            );
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                confidence REAL NOT NULL DEFAULT 0,
                valid_from TEXT,
                valid_to TEXT,
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                supersedes_fact_id TEXT REFERENCES facts(id),
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS semantic_profiles (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                person_id TEXT NOT NULL UNIQUE REFERENCES entities(id),
                summary_zh TEXT NOT NULL DEFAULT '',
                activity_summary_zh TEXT NOT NULL DEFAULT '',
                place_summary_zh TEXT NOT NULL DEFAULT '',
                appearance_summary_zh TEXT NOT NULL DEFAULT '',
                preference_summary_zh TEXT NOT NULL DEFAULT '',
                event_memory_json TEXT NOT NULL DEFAULT '[]',
                patterns_json TEXT NOT NULL DEFAULT '[]',
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS semantic_claims (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                person_id TEXT NOT NULL REFERENCES entities(id),
                dimension TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value_text TEXT NOT NULL,
                normalized_value_text TEXT,
                normalization_version TEXT NOT NULL DEFAULT 'zh-v1',
                support_count INTEGER NOT NULL DEFAULT 1,
                value_entity_id TEXT REFERENCES entities(id),
                valid_from TEXT,
                valid_to TEXT,
                supporting_event_ids_json TEXT NOT NULL DEFAULT '[]',
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'active',
                confidence REAL NOT NULL DEFAULT 0,
                confidence_source TEXT NOT NULL DEFAULT 'derived',
                supersedes_claim_id TEXT REFERENCES semantic_claims(id),
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS person_event_memory (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                person_id TEXT NOT NULL REFERENCES entities(id),
                event_id TEXT NOT NULL REFERENCES events(id),
                role TEXT NOT NULL DEFAULT 'visible_subject',
                activity_text TEXT NOT NULL DEFAULT '',
                place_text TEXT NOT NULL DEFAULT '',
                time_start TEXT,
                time_end TEXT,
                co_person_ids_json TEXT NOT NULL DEFAULT '[]',
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(person_id, event_id, role)
            );
            CREATE TABLE IF NOT EXISTS person_patterns (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                person_id TEXT NOT NULL REFERENCES entities(id),
                pattern_type TEXT NOT NULL,
                value_text TEXT NOT NULL,
                normalized_key TEXT NOT NULL,
                supporting_event_ids_json TEXT NOT NULL DEFAULT '[]',
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                support_count INTEGER NOT NULL DEFAULT 0,
                first_seen TEXT,
                last_seen TEXT,
                confidence REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(person_id, pattern_type, normalized_key)
            );
            CREATE TABLE IF NOT EXISTS query_gaps (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL DEFAULT 'home-default',
                query TEXT NOT NULL,
                missing_dimension TEXT NOT NULL,
                candidate_asset_ids_json TEXT NOT NULL DEFAULT '[]',
                resolution TEXT NOT NULL DEFAULT 'open',
                evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_feedback (
                id TEXT PRIMARY KEY,
                query_gap_id TEXT REFERENCES query_gaps(id),
                user_id TEXT,
                accepted_answer TEXT,
                correction TEXT,
                target_claim_id TEXT REFERENCES semantic_claims(id),
                target_entity_id TEXT REFERENCES entities(id),
                target_event_id TEXT REFERENCES events(id),
                target_property_key TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dialogue_states (
                conversation_id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                state_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rebuild_runs (
                id TEXT PRIMARY KEY,
                run_version TEXT NOT NULL,
                scope TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                stats_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stories (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                outline_json TEXT NOT NULL DEFAULT '[]',
                event_ids_json TEXT NOT NULL DEFAULT '[]',
                content TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runtime_settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS invites (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                expires_at TEXT
            );
            """
        )
        self._migrate_face_instances_cluster_nullable()
        self.connection.execute("INSERT OR IGNORE INTO runtime_settings(key,value) VALUES('vlm_backend','ollama_12b')")
        self._ensure_columns("assets", {
            "scope_id": "TEXT NOT NULL DEFAULT 'home-default'",
            "batch_id": "TEXT",
            "source_owner_id": "TEXT", "source_owner_label": "TEXT", "source_device_id": "TEXT", "source_album_id": "TEXT",
            "source_confidence": "REAL NOT NULL DEFAULT 0", "content_sha256": "TEXT", "captured_at": "TEXT", "captured_location": "TEXT",
        })
        self._ensure_columns("observations", {
            "scope_id": "TEXT NOT NULL DEFAULT 'home-default'",
            "canonical_json": "TEXT NOT NULL DEFAULT '{}'", "source_owner_id": "TEXT",
            "inferred_captured_by": "TEXT", "clothing_json": "TEXT NOT NULL DEFAULT '[]'",
            "spatial_relations_json": "TEXT NOT NULL DEFAULT '[]'", "revision": "INTEGER NOT NULL DEFAULT 1", "updated_at": "TEXT",
        })
        self._ensure_columns("face_instances", {
            "asset_id": "TEXT", "observation_id": "TEXT", "cluster_id": "TEXT",
            "bbox_json": "TEXT NOT NULL DEFAULT '[]'", "embedding_json": "TEXT NOT NULL DEFAULT '[]'",
            "detection_confidence": "REAL NOT NULL DEFAULT 0",
            "quality": "REAL NOT NULL DEFAULT 0", "pose_json": "TEXT NOT NULL DEFAULT '[]'",
            "area_ratio": "REAL NOT NULL DEFAULT 0", "sharpness": "REAL NOT NULL DEFAULT 0",
            "pose_bucket": "TEXT NOT NULL DEFAULT 'unknown'", "embedding_model": "TEXT NOT NULL DEFAULT 'unknown'",
            "embedding_version": "TEXT NOT NULL DEFAULT 'unknown'", "embedding_quality_signal": "REAL NOT NULL DEFAULT 0",
        })
        self._ensure_columns("events", {
            "scope_id": "TEXT NOT NULL DEFAULT 'home-default'",
            "title": "TEXT NOT NULL DEFAULT '未命名事件'", "event_type": "TEXT",
            "time_start": "TEXT", "time_end": "TEXT", "place": "TEXT", "activity": "TEXT",
            "summary": "TEXT", "participants_json": "TEXT NOT NULL DEFAULT '[]'",
            "confidence": "REAL NOT NULL DEFAULT 0", "status": "TEXT NOT NULL DEFAULT 'active'",
            "aggregation_score": "REAL NOT NULL DEFAULT 0",
            "aggregation_breakdown_json": "TEXT NOT NULL DEFAULT '{}'",
            "merged_into_event_id": "TEXT", "merge_reason": "TEXT",
            "cover_asset_id": "TEXT", "cover_selection_json": "TEXT NOT NULL DEFAULT '{}'",
            "revision": "INTEGER NOT NULL DEFAULT 1", "created_at": "TEXT", "updated_at": "TEXT",
        })
        self._ensure_columns("entities", {"scope_id": "TEXT NOT NULL DEFAULT 'home-default'"})
        self._ensure_columns("stories", {"tags_json": "TEXT NOT NULL DEFAULT '[]'"})
        self._ensure_columns("query_gaps", {"scope_id": "TEXT NOT NULL DEFAULT 'home-default'"})
        self._ensure_columns("memory_feedback", {
            "target_entity_id": "TEXT REFERENCES entities(id)", "target_event_id": "TEXT REFERENCES events(id)",
            "target_property_key": "TEXT",
        })
        self._ensure_columns("entity_merge_candidates", {"target_entity_id": "TEXT REFERENCES entities(id)"})
        self._ensure_columns("face_clusters", {"scope_id": "TEXT NOT NULL DEFAULT 'home-default'"})
        self._ensure_columns("relationships", {"scope_id": "TEXT NOT NULL DEFAULT 'home-default'"})
        self._ensure_columns("facts", {"scope_id": "TEXT NOT NULL DEFAULT 'home-default'"})
        self._ensure_columns("memory_vectors", {"scope_id": "TEXT NOT NULL DEFAULT 'home-default'"})
        self._ensure_columns("semantic_profiles", {
            "scope_id": "TEXT NOT NULL DEFAULT 'home-default'",
            "event_memory_json": "TEXT NOT NULL DEFAULT '[]'",
            "patterns_json": "TEXT NOT NULL DEFAULT '[]'",
        })
        self._ensure_columns("semantic_claims", {
            "scope_id": "TEXT NOT NULL DEFAULT 'home-default'",
            "confidence_source": "TEXT NOT NULL DEFAULT 'derived'",
            "normalized_value_text": "TEXT",
            "normalization_version": "TEXT NOT NULL DEFAULT 'zh-v1'",
            "support_count": "INTEGER NOT NULL DEFAULT 1",
        })
        self.connection.executescript(
            """
            INSERT OR IGNORE INTO memory_spaces(id, name, kind, created_at, updated_at)
            VALUES ('home-default', '默认家庭', 'household', datetime('now'), datetime('now'));
            CREATE INDEX IF NOT EXISTS idx_observations_asset ON observations(asset_id);
            CREATE INDEX IF NOT EXISTS idx_assets_batch ON assets(batch_id);
            CREATE INDEX IF NOT EXISTS idx_assets_content_sha256 ON assets(content_sha256);
            CREATE INDEX IF NOT EXISTS idx_events_time ON events(time_start);
            CREATE INDEX IF NOT EXISTS idx_event_revisions_event ON event_revisions(event_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_facts_subject_predicate ON facts(subject, predicate);
            CREATE INDEX IF NOT EXISTS idx_face_instances_cluster ON face_instances(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_face_prototypes_cluster ON face_prototypes(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_person_appearance_person ON person_appearance_evidence(person_id);
            CREATE INDEX IF NOT EXISTS idx_person_appearance_face ON person_appearance_evidence(face_instance_id);
            CREATE INDEX IF NOT EXISTS idx_entity_mentions_observation ON entity_mentions(observation_id);
            CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationship_subject_object ON relationships(subject_entity_id, object_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_scope ON relationships(scope_id);
            CREATE INDEX IF NOT EXISTS idx_facts_scope ON facts(scope_id);
            CREATE INDEX IF NOT EXISTS idx_memory_vectors_space ON memory_vectors(space);
            CREATE INDEX IF NOT EXISTS idx_memory_vectors_visual_asset ON memory_vectors(space, source_type, source_id, model_name);
            CREATE INDEX IF NOT EXISTS idx_event_observations_observation ON event_observations(observation_id);
            CREATE INDEX IF NOT EXISTS idx_event_participants_event ON event_participants(event_id);
            CREATE INDEX IF NOT EXISTS idx_event_participants_person ON event_participants(person_id);
            CREATE INDEX IF NOT EXISTS idx_event_entities_event ON event_entities(event_id);
            CREATE INDEX IF NOT EXISTS idx_event_entities_entity ON event_entities(entity_id);
            CREATE INDEX IF NOT EXISTS idx_trips_scope_status ON trips(scope_id, status, time_start);
            CREATE INDEX IF NOT EXISTS idx_trip_revisions_trip ON trip_revisions(trip_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_semantic_claims_person ON semantic_claims(person_id);
            CREATE INDEX IF NOT EXISTS idx_assets_scope ON assets(scope_id);
            CREATE INDEX IF NOT EXISTS idx_observations_scope ON observations(scope_id);
            CREATE INDEX IF NOT EXISTS idx_events_scope ON events(scope_id);
            CREATE INDEX IF NOT EXISTS idx_entities_scope ON entities(scope_id);
            CREATE INDEX IF NOT EXISTS idx_entity_properties_entity_key ON entity_properties(entity_id, property_key, updated_at);
            CREATE INDEX IF NOT EXISTS idx_entity_properties_status ON entity_properties(status);
            CREATE INDEX IF NOT EXISTS idx_entity_merge_candidates_scope_status ON entity_merge_candidates(scope_id, status, entity_type);
            CREATE INDEX IF NOT EXISTS idx_face_clusters_scope ON face_clusters(scope_id);
            CREATE INDEX IF NOT EXISTS idx_person_event_memory_person ON person_event_memory(person_id, event_id);
            CREATE INDEX IF NOT EXISTS idx_person_event_memory_scope ON person_event_memory(scope_id, event_id);
            CREATE INDEX IF NOT EXISTS idx_person_patterns_person ON person_patterns(person_id, pattern_type);
            CREATE INDEX IF NOT EXISTS idx_person_patterns_scope ON person_patterns(scope_id, pattern_type);
            CREATE INDEX IF NOT EXISTS idx_query_gaps_status ON query_gaps(status);
            """
        )
        self.connection.commit()
        self._migrate_legacy_persons()

    def _ensure_columns(self, table, columns):
        existing = {row["name"] for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, definition in columns.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _migrate_face_instances_cluster_nullable(self):
        """Allow low-confidence face evidence without inventing a person cluster.

        SQLite cannot drop a NOT NULL constraint in place. Older databases used
        a required `cluster_id`; rebuild the small derived table once while
        preserving its rows and child references.
        """
        columns = self.connection.execute("PRAGMA table_info(face_instances)").fetchall()
        cluster_column = next((row for row in columns if row["name"] == "cluster_id"), None)
        if not cluster_column or not cluster_column["notnull"]:
            return
        self.connection.commit()
        self.connection.execute("PRAGMA foreign_keys = OFF")
        try:
            self.connection.executescript(
                """
                CREATE TABLE face_instances_migrated (
                    id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL REFERENCES assets(id),
                    observation_id TEXT NOT NULL REFERENCES observations(id),
                    cluster_id TEXT REFERENCES face_clusters(id),
                    bbox_json TEXT NOT NULL DEFAULT '[]',
                    embedding_json TEXT NOT NULL DEFAULT '[]',
                    detection_confidence REAL NOT NULL DEFAULT 0,
                    quality REAL NOT NULL DEFAULT 0,
                    pose_json TEXT NOT NULL DEFAULT '[]',
                    area_ratio REAL NOT NULL DEFAULT 0,
                    sharpness REAL NOT NULL DEFAULT 0,
                    pose_bucket TEXT NOT NULL DEFAULT 'unknown',
                    embedding_model TEXT NOT NULL DEFAULT 'unknown',
                    embedding_version TEXT NOT NULL DEFAULT 'unknown',
                    embedding_quality_signal REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                INSERT INTO face_instances_migrated(
                    id, asset_id, observation_id, cluster_id, bbox_json,
                    embedding_json, detection_confidence, quality, pose_json,
                    area_ratio, sharpness, pose_bucket, embedding_model,
                    embedding_version, embedding_quality_signal, created_at
                ) SELECT
                    id, asset_id, observation_id, cluster_id, bbox_json,
                    embedding_json, detection_confidence, quality, pose_json,
                    area_ratio, sharpness, pose_bucket, embedding_model,
                    embedding_version, embedding_quality_signal, created_at
                FROM face_instances;
                DROP TABLE face_instances;
                ALTER TABLE face_instances_migrated RENAME TO face_instances;
                """
            )
            self.connection.commit()
        finally:
            self.connection.execute("PRAGMA foreign_keys = ON")
        violations = self.connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("face instance migration produced foreign-key violations")

    def _migrate_legacy_persons(self):
        """Keep the first prototype's person candidates visible in the native entity view."""
        rows = self._rows("SELECT * FROM persons")
        timestamp = now_iso()
        for person in rows:
            entity_id = f"entity_{person['id']}"
            self.connection.execute(
                """INSERT OR IGNORE INTO entities(
                    id, entity_type, canonical_name, status, family_role, summary, confidence, created_at, updated_at
                ) VALUES (?, 'person', ?, ?, NULL, ?, ?, ?, ?)""",
                (entity_id, person["name"], person["status"], "历史人物候选", float(person["confidence"] or 0), person["created_at"] or timestamp, person["updated_at"] or timestamp),
            )
        self.connection.commit()

    def count(self, table):
        if table not in {"memory_spaces", "assets", "observations", "events", "event_observations", "event_participants", "persons", "entities", "entity_revisions", "entity_merge_candidates", "face_clusters", "face_instances", "face_prototypes", "person_appearance_evidence", "entity_mentions", "relationships", "memory_vectors", "facts", "semantic_profiles", "semantic_claims", "person_event_memory", "person_patterns", "query_gaps", "memory_feedback", "dialogue_states", "rebuild_runs", "stories", "invites", "trips"}:
            raise ValueError("unsupported table")
        return self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def create_memory_space(self, scope_id, name, kind="household", source_path=None):
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO memory_spaces(id, name, kind, source_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
            name = CASE
                WHEN memory_spaces.name IS NULL OR trim(memory_spaces.name) = ''
                    OR memory_spaces.name = memory_spaces.id
                THEN excluded.name
                ELSE memory_spaces.name
            END,
            kind = excluded.kind,
            source_path = excluded.source_path, updated_at = excluded.updated_at""",
            (scope_id, name, kind, source_path, timestamp, timestamp),
        )
        self.connection.commit()
        return self._row("SELECT * FROM memory_spaces WHERE id = ?", (scope_id,))

    def list_memory_spaces(self, status="active"):
        if status:
            return self._rows("SELECT * FROM memory_spaces WHERE status = ? ORDER BY created_at", (status,))
        return self._rows("SELECT * FROM memory_spaces ORDER BY created_at")

    def _row(self, query, params=()):
        row = self.connection.execute(query, params).fetchone()
        return dict(row) if row else None

    def _rows(self, query, params=()):
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def _decode(self, row, fields):
        if not row:
            return row
        result = dict(row)
        for field in fields:
            try:
                result[field] = json.loads(result[field] or "[]")
            except (TypeError, json.JSONDecodeError):
                result[field] = []
        return result

    def create_asset(self, asset_id, file_name, media_type, path, mime_type=None, size_bytes=0, metadata=None, scope_id=None):
        metadata = metadata or {}
        scope_id = scope_id or metadata.get("scope_id") or "home-default"
        self.create_memory_space(scope_id, scope_id) if not self._row("SELECT id FROM memory_spaces WHERE id = ?", (scope_id,)) else None
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO assets(
                id, scope_id, batch_id, file_name, media_type, path, mime_type, size_bytes, metadata_json,
                source_owner_id, source_owner_label, source_device_id, source_album_id, source_confidence,
                content_sha256, captured_at, captured_location, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                asset_id, scope_id, metadata.get("batch_id"), file_name, media_type, path, mime_type, size_bytes, json_value(metadata, {}),
                metadata.get("source_owner_id"), metadata.get("source_owner_label"), metadata.get("source_device_id"), metadata.get("source_album_id"),
                float(metadata.get("source_confidence", 0) or 0), metadata.get("content_sha256") or metadata.get("sha256"), metadata.get("captured_at"), metadata.get("captured_location"),
                timestamp, timestamp,
            ),
        )
        self.connection.commit()
        return self.get_asset(asset_id)

    def get_asset(self, asset_id):
        return self._decode(self._row("SELECT * FROM assets WHERE id = ?", (asset_id,)), ["metadata_json"])

    def find_asset_by_hash(self, content_sha256, scope_id=None):
        if not content_sha256:
            return None
        if scope_id:
            row = self._row("SELECT id FROM assets WHERE content_sha256 = ? AND scope_id = ? ORDER BY created_at LIMIT 1", (content_sha256, scope_id))
        else:
            row = self._row("SELECT id FROM assets WHERE content_sha256 = ? ORDER BY created_at LIMIT 1", (content_sha256,))
        return self.get_asset(row["id"]) if row else None

    def is_confirmed_person_name(self, name, scope_id=None):
        value = str(name or "").strip()
        if not value:
            return False
        clauses = ["entity_type = 'person'", "status = 'confirmed'", "canonical_name = ?"]
        params = [value]
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        return bool(self._row(f"SELECT id FROM entities WHERE {' AND '.join(clauses)} LIMIT 1", params))

    def list_assets(self, media_type=None, status=None, limit=200, scope_id=None):
        clauses = []
        params = []
        if media_type:
            clauses.append("media_type = ?")
            params.append(media_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._rows(f"SELECT * FROM assets {where} ORDER BY created_at DESC LIMIT ?", params)
        return [self._decode(row, ["metadata_json"]) for row in rows]

    def update_asset(self, asset_id, status, metadata=None):
        metadata = metadata or {}
        current = self.get_asset(asset_id) or {}
        merged_metadata = {**(current.get("metadata_json") or {}), **metadata}
        self.connection.execute(
            """UPDATE assets SET status = ?, batch_id = ?, metadata_json = ?, source_owner_id = ?, source_owner_label = ?, source_device_id = ?,
                source_album_id = ?, source_confidence = ?, content_sha256 = ?, captured_at = ?, captured_location = ?, updated_at = ? WHERE id = ?""",
            (
                status, metadata.get("batch_id", current.get("batch_id")), json_value(merged_metadata, {}), metadata.get("source_owner_id", current.get("source_owner_id")),
                metadata.get("source_owner_label", current.get("source_owner_label")), metadata.get("source_device_id", current.get("source_device_id")), metadata.get("source_album_id", current.get("source_album_id")),
                float(metadata.get("source_confidence", current.get("source_confidence", 0)) or 0),
                metadata.get("content_sha256", metadata.get("sha256", current.get("content_sha256"))),
                metadata.get("captured_at", current.get("captured_at")), metadata.get("captured_location", current.get("captured_location")),
                now_iso(), asset_id,
            ),
        )
        self.connection.commit()
        return self.get_asset(asset_id)

    def create_ingest_batch(self, batch_id, scope_id="home-default"):
        timestamp = now_iso()
        self.connection.execute(
            """INSERT OR IGNORE INTO ingest_batches(id, scope_id, status, created_at, updated_at)
            VALUES (?, ?, 'open', ?, ?)""",
            (str(batch_id), scope_id or "home-default", timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_ingest_batch(batch_id)

    def get_ingest_batch(self, batch_id):
        return self._row("SELECT * FROM ingest_batches WHERE id = ?", (str(batch_id),))

    def complete_ingest_batch(self, batch_id):
        timestamp = now_iso()
        self.connection.execute(
            """UPDATE ingest_batches SET status = CASE WHEN status IN ('completed', 'summarizing') THEN status ELSE 'complete' END,
            updated_at = ?, completed_at = COALESCE(completed_at, ?) WHERE id = ?""",
            (timestamp, timestamp, str(batch_id)),
        )
        self.connection.commit()
        return self.get_ingest_batch(batch_id)

    def claim_ingest_batch_summary(self, batch_id):
        batch_id = str(batch_id)
        batch = self.get_ingest_batch(batch_id)
        if not batch or batch["status"] != "complete":
            return False
        pending = self._row(
            "SELECT COUNT(*) AS count FROM assets WHERE batch_id = ? AND status IN ('queued', 'processing', 'semantic_enriching')",
            (batch_id,),
        )
        if pending and pending["count"]:
            return False
        cursor = self.connection.execute(
            "UPDATE ingest_batches SET status = 'summarizing', updated_at = ? WHERE id = ? AND status = 'complete'",
            (now_iso(), batch_id),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def finish_ingest_batch(self, batch_id):
        self.connection.execute(
            "UPDATE ingest_batches SET status = 'completed', updated_at = ? WHERE id = ? AND status = 'summarizing'",
            (now_iso(), str(batch_id)),
        )
        self.connection.commit()
        return self.get_ingest_batch(batch_id)

    def batch_event_ids(self, batch_id):
        rows = self._rows(
            """SELECT DISTINCT eo.event_id FROM event_observations eo
            JOIN observations o ON o.id = eo.observation_id
            JOIN assets a ON a.id = o.asset_id
            WHERE a.batch_id = ? ORDER BY eo.event_id""",
            (str(batch_id),),
        )
        return [row["event_id"] for row in rows]

    def cleanup_asset_derivatives(self, asset_id):
        """Remove only derived records owned by one failed asset."""
        observation_rows = self._rows("SELECT id FROM observations WHERE asset_id = ?", (asset_id,))
        observation_ids = [row["id"] for row in observation_rows]
        if not observation_ids:
            return
        placeholders = ",".join("?" for _ in observation_ids)
        event_rows = self._rows(
            f"SELECT event_id FROM event_observations WHERE observation_id IN ({placeholders})",
            observation_ids,
        )
        event_ids = list(dict.fromkeys(row["event_id"] for row in event_rows))
        face_rows = self._rows(
            f"SELECT id FROM face_instances WHERE observation_id IN ({placeholders})",
            observation_ids,
        )
        face_ids = [row["id"] for row in face_rows]
        if face_ids:
            face_placeholders = ",".join("?" for _ in face_ids)
            self.connection.execute(f"DELETE FROM face_prototypes WHERE face_instance_id IN ({face_placeholders})", face_ids)
            self.connection.execute(f"DELETE FROM memory_vectors WHERE source_type = 'face_instance' AND source_id IN ({face_placeholders})", face_ids)
            self.connection.execute(f"DELETE FROM face_instances WHERE id IN ({face_placeholders})", face_ids)
        self.connection.execute(
            "DELETE FROM memory_vectors WHERE (source_type = 'asset' AND source_id = ?) OR metadata_json LIKE ?",
            (asset_id, f'%"asset_id": "{asset_id}"%'),
        )
        for observation_id in observation_ids:
            self.connection.execute("DELETE FROM facts WHERE evidence_ids_json LIKE ?", (f'%{observation_id}%',))
        self.connection.execute(f"DELETE FROM entity_observations WHERE observation_id IN ({placeholders})", observation_ids)
        self.connection.execute(f"DELETE FROM event_observations WHERE observation_id IN ({placeholders})", observation_ids)
        self.connection.execute(f"DELETE FROM entity_mentions WHERE observation_id IN ({placeholders})", observation_ids)
        self.connection.execute(f"DELETE FROM observations WHERE id IN ({placeholders})", observation_ids)
        for event_id in event_ids:
            if self.connection.execute("SELECT COUNT(*) FROM event_observations WHERE event_id = ?", (event_id,)).fetchone()[0] == 0:
                self.connection.execute("DELETE FROM event_entities WHERE event_id = ?", (event_id,))
                self.connection.execute("DELETE FROM event_participants WHERE event_id = ?", (event_id,))
                self.connection.execute("DELETE FROM event_revisions WHERE event_id = ?", (event_id,))
                self.connection.execute("DELETE FROM events WHERE id = ?", (event_id,))
        self.connection.commit()

    def add_observation(self, asset_id, data, scope_id=None):
        observation_id = data.get("id") or make_id("obs")
        asset = self.get_asset(asset_id) or {}
        scope_id = scope_id or data.get("scope_id") or asset.get("scope_id") or "home-default"
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO observations(
                id, scope_id, asset_id, captured_at, source_type, caption, activity, place,
                people_json, objects_json, ocr_text, event_type, transcript, confidence, raw_json,
                canonical_json, source_owner_id, inferred_captured_by, clothing_json, spatial_relations_json, revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                observation_id,
                scope_id,
                asset_id,
                data.get("captured_at"),
                data.get("source_type", "image"),
                data.get("caption", ""),
                data.get("activity", ""),
                data.get("place", ""),
                json_value(data.get("people"), []),
                json_value(data.get("objects"), []),
                data.get("ocr_text", ""),
                data.get("event_type", ""),
                data.get("transcript", ""),
                float(data.get("confidence", 0) or 0),
                json_value(data.get("raw"), {}),
                json_value(data.get("canonical"), {}),
                data.get("source_owner_id"),
                data.get("inferred_captured_by"),
                json_value(data.get("clothing"), []),
                json_value(data.get("spatial_relations"), []),
                int(data.get("revision", 1) or 1),
                timestamp,
                timestamp,
            ),
        )
        self.connection.commit()
        return self.get_observation(observation_id)

    def get_observation(self, observation_id):
        result = self._decode(self._row("SELECT * FROM observations WHERE id = ?", (observation_id,)), ["people_json", "objects_json", "raw_json", "canonical_json", "clothing_json", "spatial_relations_json"])
        if result:
            result["people"] = result.get("people_json", [])
            result["objects"] = result.get("objects_json", [])
            result["raw"] = result.get("raw_json", {})
            result["canonical"] = result.get("canonical_json", {})
            result["clothing"] = result.get("clothing_json", [])
            result["spatial_relations"] = result.get("spatial_relations_json", [])
        return result

    def enrich_observation(self, observation_id, details, source="agent_visual_refinement"):
        observation = self.get_observation(observation_id)
        if not observation:
            return None
        canonical = {**(observation.get("canonical") or {}), **{key: value for key, value in details.items() if value not in (None, "", [], {})}}
        assignments = ["canonical_json = ?", "revision = revision + 1"]
        params = [json_value(canonical, {})]
        for key, column in (("objects", "objects_json"), ("clothing", "clothing_json"), ("spatial_relations", "spatial_relations_json")):
            if key in details:
                assignments.append(f"{column} = ?")
                params.append(json_value(details[key], []))
        for key in ("caption", "activity", "place", "event_type", "ocr_text"):
            if details.get(key):
                assignments.append(f"{key} = ?")
                params.append(details[key])
        assignments.extend(["updated_at = ?"])
        params.extend([now_iso(), observation_id])
        self.connection.execute(f"UPDATE observations SET {', '.join(assignments)} WHERE id = ?", params)
        self.connection.commit()
        mentioned = self._rows("SELECT DISTINCT entity_id FROM entity_mentions WHERE observation_id = ?", (observation_id,))
        for item in mentioned:
            self.rebuild_person_memory(item["entity_id"])
        return self.get_observation(observation_id)

    def entity_mentions_for_observation(self, observation_id):
        """Entity mentions for an observation (Phase 12B-FC person bridging)."""
        return self._rows("SELECT entity_id, confidence FROM entity_mentions WHERE observation_id = ?",
                          (observation_id,))

    def list_observations(self, limit=100, scope_id=None):
        if scope_id:
            rows = self._rows("SELECT * FROM observations WHERE scope_id = ? ORDER BY created_at DESC LIMIT ?", (scope_id, limit))
        else:
            rows = self._rows("SELECT * FROM observations ORDER BY created_at DESC LIMIT ?", (limit,))
        values = []
        for row in rows:
            value = self._decode(row, ["people_json", "objects_json", "raw_json", "canonical_json", "clothing_json", "spatial_relations_json"])
            value["people"] = value.get("people_json", [])
            value["objects"] = value.get("objects_json", [])
            value["raw"] = value.get("raw_json", {})
            value["canonical"] = value.get("canonical_json", {})
            value["clothing"] = value.get("clothing_json", [])
            value["spatial_relations"] = value.get("spatial_relations_json", [])
            values.append(value)
        return values

    def _event_candidates(self, observation):
        scope_id = (self.get_asset(observation.get("asset_id")) or {}).get("scope_id") or "home-default"
        rows = self._rows("SELECT * FROM events WHERE status = 'active' AND scope_id = ? ORDER BY time_start DESC", (scope_id,))
        anchor = self._event_anchor(observation)
        captured = parse_time(anchor["captured_at"])
        candidates = []
        for row in rows:
            event_time = parse_time(row.get("time_start"))
            if captured and event_time and abs((captured - event_time).total_seconds()) > 6 * 3600:
                continue
            event_anchors = [self._event_anchor(self.get_observation(item["observation_id"])) for item in self._rows("SELECT observation_id FROM event_observations WHERE event_id = ?", (row["id"],))]
            score = self._event_candidate_score(observation, row, event_anchors)
            if score["total"] >= 0.50:
                candidate = dict(row)
                candidate["aggregation_score"] = score["total"]
                candidate["aggregation_breakdown"] = score
                candidates.append(candidate)
        candidates.sort(key=lambda item: (-item["aggregation_score"], item.get("time_start") or "", item["id"]))
        return candidates

    def _event_candidate_diagnostics(self, observation):
        scope_id = (self.get_asset(observation.get("asset_id")) or {}).get("scope_id") or "home-default"
        rows = self._rows("SELECT * FROM events WHERE status = 'active' AND scope_id = ? ORDER BY time_start DESC", (scope_id,))
        anchor = self._event_anchor(observation)
        captured = parse_time(anchor["captured_at"])
        diagnostics = []
        for row in rows:
            event_time = parse_time(row.get("time_start"))
            if captured and event_time and abs((captured - event_time).total_seconds()) > 6 * 3600:
                continue
            event_anchors = [self._event_anchor(self.get_observation(item["observation_id"])) for item in self._rows("SELECT observation_id FROM event_observations WHERE event_id = ?", (row["id"],))]
            score = self._event_candidate_score(observation, row, event_anchors)
            diagnostics.append({"event_id": row["id"], "score": score["total"], "breakdown": score})
        return sorted(diagnostics, key=lambda item: (-item["score"], item["event_id"]))

    @staticmethod
    def _tokens(value):
        if isinstance(value, (list, tuple, set)):
            values = value
        else:
            values = str(value or "").lower().replace("、", " ").replace("，", " ").split()
        return {str(item).strip().lower() for item in values if str(item).strip()}

    def _event_candidate_score(self, observation, event, event_anchors):
        anchor = self._event_anchor(observation)
        captured = parse_time(anchor["captured_at"])
        event_time = parse_time(event.get("time_start"))
        if captured and event_time:
            minutes = abs((captured - event_time).total_seconds()) / 60.0
            time_score = max(0.0, 1.0 - min(minutes, 360.0) / 360.0)
        else:
            time_score = 0.25
        locations = {item["location"] for item in event_anchors if item["location"]}
        visual_places = {item["visual_place"] for item in event_anchors if item["visual_place"]}
        if anchor["location"] and locations:
            _location_distances = [self._geo_distance_meters(anchor["location"], loc) for loc in locations]
            _location_distances = [d for d in _location_distances if d is not None]
            location_score = max((max(0.0, 1.0 - min(d, 1000.0) / 1000.0) for d in _location_distances), default=0.0)
        else:
            location_score = 0.0
        visual_place_score = 1.0 if anchor["visual_place"] and anchor["visual_place"] in visual_places else 0.0
        event_type = anchor["visual_event_type"]
        event_types = {item["visual_event_type"] for item in event_anchors if item["visual_event_type"]}
        event_type_score = 0.8 if event_type and event_type in event_types else 0.0
        activity = str(observation.get("activity") or "").lower()
        existing_activity = str(event.get("activity") or "").lower()
        activity_score = 0.9 if activity and existing_activity and (activity == existing_activity or activity in existing_activity or existing_activity in activity) else 0.0
        if activity and existing_activity and not activity_score and event_type and event_type not in event_types:
            activity_score = -0.8
        # 同时间(<60s)+同地点(<50m)时,不因activity冲突惩罚
        same_time_place = time_score >= 0.99 and location_score >= 0.95
        if same_time_place and activity_score < 0:
            activity_score = 0.0
        object_sets = [self._tokens(self.get_observation(item["observation_id"]).get("objects")) for item in self._rows("SELECT observation_id FROM event_observations WHERE event_id = ?", (event["id"],))]
        objects = self._tokens(observation.get("objects"))
        object_score = 0.8 if objects and any(objects.intersection(values) for values in object_sets) else 0.0
        people = self._confirmed_entity_ids_for_observation(observation.get("id"))
        existing_people = self._confirmed_entity_ids_for_event(event["id"])
        person_score = 0.8 if people and existing_people and people.intersection(existing_people) else 0.0
        visual_similarity, visual_available = self._event_visual_similarity(observation, event["id"])
        semantic_conflict = bool(
            event_type and event_types and event_type not in event_types
        )
        corroborated = bool(object_score or person_score)
        split_guard = None
        # Visual variation alone is normal for a family event: close-ups,
        # group shots, and different camera angles need not look alike. Split
        # only when independent semantic evidence also conflicts and no known
        # person/object bridges the candidate event.
        if visual_available and semantic_conflict and visual_similarity < 0.45 and not corroborated and not same_time_place:
            split_guard = "semantic_visual_conflict"
        visual_boost = (
            max(0.0, min(1.0, (visual_similarity - 0.70) / 0.30))
            if visual_available else 0.0
        )
        total = (
            0.25 * time_score + 0.25 * location_score + 0.15 * visual_place_score
            + 0.05 * event_type_score + 0.20 * activity_score
            + 0.05 * object_score + 0.20 * person_score + 0.05 * visual_boost
        )
        if split_guard:
            total *= 0.3
        return {
            "total": max(0.0, min(1.0, total)), "time": time_score, "location": location_score,
            "visual_place": visual_place_score, "event_type": event_type_score,
            "activity": activity_score, "objects": object_score, "confirmed_people": person_score,
            "visual_similarity": visual_similarity, "visual_available": visual_available,
            "visual_boost": visual_boost, "semantic_conflict": semantic_conflict,
            "split_guard": split_guard,
        }

    def _event_visual_similarity(self, observation, event_id):
        """Return the strongest asset-vector match for a candidate event."""
        asset_id = observation.get("asset_id")
        row = self._row(
            """SELECT vector_json, model_name FROM memory_vectors WHERE space = 'visual'
            AND source_type = 'asset' AND source_id = ? ORDER BY updated_at DESC LIMIT 1""",
            (asset_id,),
        )
        if not row:
            return 0.0, False
        try:
            incoming = json.loads(row["vector_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            return 0.0, False
        if not incoming:
            return 0.0, False
        rows = self._rows(
            """SELECT mv.vector_json FROM memory_vectors mv JOIN observations o
            ON o.asset_id = mv.source_id JOIN event_observations eo
            ON eo.observation_id = o.id
            WHERE mv.space = 'visual' AND mv.source_type = 'asset'
            AND mv.model_name = ? AND eo.event_id = ?""",
            (row["model_name"], event_id),
        )
        vectors = []
        for item in rows:
            try:
                values = json.loads(item["vector_json"] or "[]")
            except (TypeError, json.JSONDecodeError):
                continue
            if values and len(values) == len(incoming):
                vectors.append(values)
        if not vectors:
            return 0.0, False
        return max(self._cosine(incoming, values) for values in vectors), True

    @staticmethod
    def _geo_distance_meters(left, right):
        try:
            left_lat, left_lon = (float(value) for value in str(left).split(",")[:2])
            right_lat, right_lon = (float(value) for value in str(right).split(",")[:2])
        except (TypeError, ValueError):
            return None
        radius = 6371000.0
        lat1, lat2 = math.radians(left_lat), math.radians(right_lat)
        dlat = lat2 - lat1
        dlon = math.radians(right_lon - left_lon)
        haversine = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return radius * 2 * math.asin(math.sqrt(min(1.0, haversine)))

    def _event_asset_vectors(self, event):
        asset_ids = [
            (self.get_observation(observation_id) or {}).get("asset_id")
            for observation_id in event.get("observation_ids", [])
        ]
        asset_ids = [asset_id for asset_id in asset_ids if asset_id]
        if not asset_ids:
            return []
        placeholders = ",".join("?" for _ in asset_ids)
        rows = self._rows(
            f"SELECT vector_json FROM memory_vectors WHERE space = 'visual' AND source_type = 'asset' AND source_id IN ({placeholders})",
            asset_ids,
        )
        values = []
        for row in rows:
            try:
                vector = json.loads(row["vector_json"] or "[]")
            except (TypeError, json.JSONDecodeError):
                vector = []
            if vector:
                values.append(vector)
        return values

    def _event_face_clusters(self, event):
        observation_ids = event.get("observation_ids", [])
        if not observation_ids:
            return set()
        placeholders = ",".join("?" for _ in observation_ids)
        rows = self._rows(
            f"SELECT DISTINCT cluster_id FROM face_instances WHERE observation_id IN ({placeholders}) AND cluster_id IS NOT NULL",
            observation_ids,
        )
        return {row["cluster_id"] for row in rows if row["cluster_id"]}

    def _event_semantic_values(self, event):
        values = set()
        for observation_id in event.get("observation_ids", []):
            observation = self.get_observation(observation_id) or {}
            for key in ("activity", "event_type", "place"):
                value = str(observation.get(key) or "").strip().lower()
                if value:
                    values.add(value)
            for value in observation.get("objects") or []:
                value = str(value.get("name") if isinstance(value, dict) else value).strip().lower()
                if value and value not in {"人", "人物", "照片", "室内", "户外"}:
                    values.add(value)
        return values

    @staticmethod
    def _set_overlap(left, right):
        if not left or not right:
            return 0.0
        if left.intersection(right):
            return 1.0
        for left_value in left:
            for right_value in right:
                if len(left_value) >= 2 and len(right_value) >= 2 and (left_value in right_value or right_value in left_value):
                    return 0.7
        return 0.0

    def _event_pair_score(self, left, right):
        left_times = [parse_time((self.get_asset((self.get_observation(item) or {}).get("asset_id")) or {}).get("captured_at")) for item in left.get("observation_ids", [])]
        right_times = [parse_time((self.get_asset((self.get_observation(item) or {}).get("asset_id")) or {}).get("captured_at")) for item in right.get("observation_ids", [])]
        time_distances = [abs((left_time - right_time).total_seconds()) / 60.0 for left_time in left_times for right_time in right_times if left_time and right_time]
        time_score = max((max(0.0, 1.0 - min(value, 360.0) / 360.0) for value in time_distances), default=0.0)
        left_locations = [(self.get_asset((self.get_observation(item) or {}).get("asset_id")) or {}).get("captured_location") for item in left.get("observation_ids", [])]
        right_locations = [(self.get_asset((self.get_observation(item) or {}).get("asset_id")) or {}).get("captured_location") for item in right.get("observation_ids", [])]
        distances = [self._geo_distance_meters(a, b) for a in left_locations for b in right_locations if a and b]
        distances = [value for value in distances if value is not None]
        location_score = max((max(0.0, 1.0 - min(value, 1000.0) / 1000.0) for value in distances), default=0.0)
        left_vectors, right_vectors = self._event_asset_vectors(left), self._event_asset_vectors(right)
        visual_score = max((self._cosine(a, b) for a in left_vectors for b in right_vectors), default=0.0)
        semantic_score = self._set_overlap(self._event_semantic_values(left), self._event_semantic_values(right))
        face_overlap = bool(self._event_face_clusters(left).intersection(self._event_face_clusters(right)))
        merge = time_score >= 0.70 and (
            (location_score >= 0.65 and (visual_score >= 0.55 or semantic_score > 0 or face_overlap))
            or (visual_score >= 0.80 and (semantic_score > 0 or face_overlap))
            or (location_score >= 0.85 and semantic_score > 0)
        )
        return {
            "merge": merge, "time": time_score, "location": location_score,
            "visual_similarity": visual_score, "semantic": semantic_score,
            "face_overlap": face_overlap,
            "total": 0.35 * time_score + 0.25 * location_score + 0.25 * visual_score + 0.10 * semantic_score + (0.05 if face_overlap else 0.0),
        }

    def consolidate_events(self, scope_id=None):
        """Merge close event candidates after all observations have been imported."""
        merged = []
        while True:
            events = self.list_events(1000, scope_id)
            selected = None
            for index, left in enumerate(events):
                for right in events[index + 1:]:
                    score = self._event_pair_score(left, right)
                    if score["merge"] and (selected is None or score["total"] > selected[2]["total"]):
                        selected = (left, right, score)
            if selected is None:
                break
            left, right, score = selected
            left_count, right_count = len(left.get("observation_ids", [])), len(right.get("observation_ids", []))
            target, source = (left, right) if (left_count, left.get("time_start") or "") >= (right_count, right.get("time_start") or "") else (right, left)
            self._merge_events(target["id"], source["id"], reason={"type": "global_consolidation", **score})
            merged.append({"target_event_id": target["id"], "source_event_id": source["id"], **score})
        return merged

    def _confirmed_entity_ids_for_observation(self, observation_id):
        if not observation_id:
            return set()
        rows = self._rows(
            """SELECT DISTINCT em.entity_id FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            WHERE em.observation_id = ? AND e.entity_type = 'person' AND e.status = 'confirmed'""",
            (observation_id,),
        )
        return {row["entity_id"] for row in rows}

    def _confirmed_entity_ids_for_event(self, event_id):
        rows = self._rows(
            """SELECT DISTINCT em.entity_id FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            JOIN event_observations eo ON eo.observation_id = em.observation_id
            WHERE eo.event_id = ? AND e.entity_type = 'person' AND e.status = 'confirmed'""",
            (event_id,),
        )
        return {row["entity_id"] for row in rows}

    def _event_anchor(self, observation):
        observation = observation or {}
        asset = self.get_asset(observation.get("asset_id")) or {}
        return {
            "captured_at": asset.get("captured_at") or observation.get("captured_at"),
            "location": (asset.get("captured_location") or "").strip().lower(),
            "visual_place": (observation.get("place") or "").strip().lower(),
            "visual_event_type": (observation.get("event_type") or "").strip().lower(),
        }

    @staticmethod
    def _event_display_place(observation):
        """Return a semantic place label, keeping GPS only on the asset."""
        observation = observation or {}
        visual_place = str(observation.get("place") or "").strip()
        if visual_place and not DISPLAY_COORDINATE_RE.fullmatch(visual_place):
            return visual_place
        canonical = observation.get("canonical") if isinstance(observation.get("canonical"), dict) else {}
        semantic = canonical.get("semantic") if isinstance(canonical.get("semantic"), dict) else {}
        semantic_place = semantic.get("place") if isinstance(semantic.get("place"), dict) else {}
        primary = str(semantic_place.get("primary") or "").strip()
        return primary if primary and primary != OTHER else ""

    def merge_observation_into_event(self, observation):
        candidates = self._event_candidates(observation)
        event = candidates[0] if candidates else None
        diagnostics = self._event_candidate_diagnostics(observation)
        anchor = self._event_anchor(observation)
        confirmed_people = [
            {"entity_id": entity_id, "name": self.get_entity(entity_id)["canonical_name"], "status": "confirmed"}
            for entity_id in sorted(self._confirmed_entity_ids_for_observation(observation.get("id")))
        ]
        people = dedupe_json_values((json.loads(event["participants_json"]) if event else []) + confirmed_people)
        captured_at = anchor["captured_at"]
        event_place = self._event_display_place(observation) or "其他或不确定"
        event_type = "待判断"
        if event:
            start = min(filter(None, [event.get("time_start"), captured_at])) if any([event.get("time_start"), captured_at]) else None
            end = max(filter(None, [event.get("time_end"), captured_at])) if any([event.get("time_end"), captured_at]) else None
            summary = event.get("summary") or observation.get("caption") or observation.get("activity") or "家庭事件"
            if observation.get("caption") and observation["caption"] not in summary:
                summary = f"{summary}；{observation['caption']}"
            self.connection.execute(
                """UPDATE events SET time_start = ?, time_end = ?, place = ?, activity = ?, summary = ?,
                participants_json = ?, confidence = MAX(confidence, ?), revision = revision + 1, updated_at = ? WHERE id = ?""",
                (start, end, event_place, event.get("activity") or observation.get("activity"), summary, json_value(people, []), float(observation.get("confidence", 0) or 0), now_iso(), event["id"]),
            )
            event_id = event["id"]
        else:
            event_id = make_id("evt")
            title = "待总结事件"
            self.connection.execute(
                """INSERT INTO events(scope_id, id, title, event_type, time_start, time_end, place, activity, summary,
                participants_json, confidence, aggregation_breakdown_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ((self.get_asset(observation.get("asset_id")) or {}).get("scope_id") or "home-default", event_id, title, event_type, captured_at, captured_at, event_place, observation.get("activity"), observation.get("caption") or title, json_value(people, []), float(observation.get("confidence", 0) or 0), json_value({"selected": False, "ambiguity": len(diagnostics) > 1, "split_guard": next((item["breakdown"].get("split_guard") for item in diagnostics if item["breakdown"].get("split_guard")), None), "candidates": diagnostics[:5]}, {}), now_iso(), now_iso()),
            )
        self.connection.execute("INSERT OR IGNORE INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event_id, observation["id"]))
        if event:
            self.connection.execute(
                "UPDATE events SET aggregation_score = ?, aggregation_breakdown_json = ? WHERE id = ?",
                (float(event.get("aggregation_score", 0) or 0), json_value(event.get("aggregation_breakdown", {}), {}), event_id),
            )
        self.connection.commit()
        self.select_event_cover(event_id)
        self._refresh_event_participants([observation["id"]])
        return self.get_event(event_id)

    def get_event(self, event_id):
        row = self._decode(self._row("SELECT * FROM events WHERE id = ?", (event_id,)), ["participants_json", "aggregation_breakdown_json", "cover_selection_json"])
        if row:
            row["participants"] = row.get("participants_json", [])
            row["aggregation_breakdown"] = row.get("aggregation_breakdown_json", {})
            row["observation_ids"] = [item["observation_id"] for item in self._rows("SELECT observation_id FROM event_observations WHERE event_id = ?", (event_id,))]
            row["asset_ids"] = [
                item["asset_id"] for item in (
                    self.get_observation(observation_id) or {} for observation_id in row["observation_ids"]
                ) if item.get("asset_id")
            ]
            row["cover_asset_id"] = row.get("cover_asset_id") if row.get("cover_asset_id") in row["asset_ids"] else (row["asset_ids"][0] if row["asset_ids"] else None)
            row["cover_selection"] = row.get("cover_selection_json", {})
            row["participant_roles"] = self.list_event_participants(event_id)
        return row

    def select_event_cover(self, event_id):
        """Choose an event cover from its own image evidence, without replacing user choice."""
        event = self.get_event(event_id)
        if not event:
            return None
        selection = event.get("cover_selection") or {}
        if event.get("cover_asset_id") and selection.get("source") == "user":
            return event
        candidates = []
        for observation_id in event.get("observation_ids", []):
            observation = self.get_observation(observation_id) or {}
            asset = self.get_asset(observation.get("asset_id")) or {}
            if asset.get("media_type") != "image":
                continue
            candidates.append((
                float(observation.get("confidence", 0) or 0),
                str(observation.get("captured_at") or asset.get("captured_at") or ""),
                str(asset.get("id") or ""), observation, asset,
            ))
        if not candidates:
            return event
        _, _, _, observation, asset = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        selection = {
            "source": "derived", "asset_id": asset["id"], "evidence_observation_id": observation["id"],
            "criteria": {
                "media_type": "image", "observation_confidence": float(observation.get("confidence", 0) or 0),
                "captured_at": observation.get("captured_at") or asset.get("captured_at"), "candidate_count": len(candidates),
            },
        }
        self.connection.execute(
            "UPDATE events SET cover_asset_id = ?, cover_selection_json = ?, updated_at = ? WHERE id = ?",
            (asset["id"], json_value(selection, {}), now_iso(), event_id),
        )
        self.connection.commit()
        return self.get_event(event_id)

    def upsert_event_participant(self, event_id, person_id, role, evidence_ids=None, confidence=0.5):
        if not self.get_entity(person_id):
            return []
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        existing = self._row("SELECT * FROM event_participants WHERE event_id = ? AND person_id = ? AND role = ?", (event_id, person_id, role))
        timestamp = now_iso()
        if existing:
            old_evidence = json.loads(existing["evidence_ids_json"] or "[]")
            merged = list(dict.fromkeys(old_evidence + evidence_ids))
            self.connection.execute(
                "UPDATE event_participants SET evidence_ids_json = ?, confidence = MAX(confidence, ?), revision = revision + 1, updated_at = ? WHERE id = ?",
                (json_value(merged, []), float(confidence or 0), timestamp, existing["id"]),
            )
        else:
            self.connection.execute(
                """INSERT INTO event_participants(id, event_id, person_id, role, evidence_ids_json, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (make_id("event_person"), event_id, person_id, role, json_value(evidence_ids, []), float(confidence or 0), timestamp, timestamp),
            )
        self.connection.commit()
        self._maintain_event_cooccurrence_candidates(event_id)
        return self.list_event_participants(event_id)

    def _maintain_event_cooccurrence_candidates(self, event_id):
        """Suggest co-occurrence only; users decide whether it represents a real relationship."""
        participants = [
            item for item in self.list_event_participants(event_id)
            if item.get("role") == "visible_subject" and item.get("person_status") == "confirmed"
        ]
        for index, left in enumerate(participants):
            for right in participants[index + 1:]:
                subject_id, object_id = sorted((left["person_id"], right["person_id"]))
                evidence_ids = list(dict.fromkeys((left.get("evidence_ids_json") or []) + (right.get("evidence_ids_json") or [])))
                support_rows = self._rows(
                    """SELECT DISTINCT ep.event_id FROM event_participants ep
                    JOIN event_participants other ON other.event_id = ep.event_id
                    WHERE ep.person_id = ? AND other.person_id = ? AND ep.role = 'visible_subject' AND other.role = 'visible_subject'""",
                    (subject_id, object_id),
                )
                confidence = min(0.9, 0.45 + 0.1 * len(support_rows))
                self.create_relationship(subject_id, "共同出现", object_id, evidence_ids, confidence, "pending")

    def list_event_participants(self, event_id=None):
        clauses = []
        params = []
        if event_id:
            clauses.append("ep.event_id = ?")
            params.append(event_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._rows(
            f"""SELECT ep.*, e.canonical_name AS person_name, e.family_role, e.status AS person_status
            FROM event_participants ep JOIN entities e ON e.id = ep.person_id {where} ORDER BY ep.updated_at DESC""",
            params,
        )
        return [self._decode(row, ["evidence_ids_json"]) for row in rows]

    def refresh_event_summary(self, event_id):
        event = self._row("SELECT * FROM events WHERE id = ?", (event_id,))
        if not event:
            return None
        participants = self.list_event_participants(event_id)
        names = []
        for participant in participants:
            name = participant.get("person_name")
            if name and name not in names:
                names.append(name)
        activity = event.get("activity") or event.get("event_type") or "活动"
        place = event.get("place") or "某处"
        if names:
            summary = f"{'、'.join(names)}在{place}参与{activity}"
        else:
            summary = event.get("summary") or f"在{place}发生的{activity}"
        self.connection.execute("UPDATE events SET summary = ?, revision = revision + 1, updated_at = ? WHERE id = ?", (summary, now_iso(), event_id))
        self.connection.commit()
        return self.get_event(event_id)

    def get_semantic_profile(self, person_id):
        row = self._row("SELECT * FROM semantic_profiles WHERE person_id = ?", (person_id,))
        return self._decode(row, ["evidence_ids_json", "event_memory_json", "patterns_json"])

    def upsert_semantic_profile(self, person_id, fields, evidence_ids=None):
        existing = self.get_semantic_profile(person_id)
        timestamp = now_iso()
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        if existing:
            old_evidence = existing.get("evidence_ids_json", [])
            merged_evidence = list(dict.fromkeys(old_evidence + evidence_ids))
            assignments = []
            values = []
            for key in ("summary_zh", "activity_summary_zh", "place_summary_zh", "appearance_summary_zh", "preference_summary_zh", "event_memory_json", "patterns_json", "scope_id"):
                if key in fields:
                    assignments.append(f"{key} = ?")
                    values.append(json_value(fields[key], []) if key.endswith("_json") else fields[key])
            assignments.extend(["evidence_ids_json = ?", "revision = revision + 1", "updated_at = ?"])
            values.extend([json_value(merged_evidence, []), timestamp, existing["id"]])
            self.connection.execute(f"UPDATE semantic_profiles SET {', '.join(assignments)} WHERE id = ?", values)
        else:
            self.connection.execute(
                """INSERT INTO semantic_profiles(
                    id, scope_id, person_id, summary_zh, activity_summary_zh, place_summary_zh,
                    appearance_summary_zh, preference_summary_zh, event_memory_json, patterns_json,
                    evidence_ids_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    make_id("profile"), fields.get("scope_id", "home-default"), person_id, fields.get("summary_zh", ""),
                    fields.get("activity_summary_zh", ""), fields.get("place_summary_zh", ""), fields.get("appearance_summary_zh", ""),
                    fields.get("preference_summary_zh", ""), json_value(fields.get("event_memory_json", fields.get("event_memory", [])), []),
                    json_value(fields.get("patterns_json", fields.get("patterns", [])), []), json_value(evidence_ids, []), timestamp, timestamp,
                ),
            )
        self.connection.commit()
        return self.get_semantic_profile(person_id)

    def get_semantic_claim(self, claim_id):
        return self._decode(self._row("SELECT * FROM semantic_claims WHERE id = ?", (claim_id,)), ["supporting_event_ids_json", "evidence_ids_json"])

    def maintain_semantic_claim(self, person_id, dimension, predicate, value_text, evidence_ids=None, event_ids=None, confidence=0.5, valid_from=None, valid_to=None, confidence_source="derived"):
        predicate = {
            "参加过": "参与", "参加": "参与", "出席": "参与",
            "出现于": "出现在", "位于": "出现在", "穿着": "穿着",
            "拍摄了": "拍摄", "拍照": "拍摄", "家庭成员角色": "家庭角色",
        }.get(str(predicate).strip(), str(predicate).strip())
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        event_ids = list(dict.fromkeys(event_ids or []))
        normalized_value = normalize_clothing(value_text) if dimension == "clothing" else str(value_text or "").strip()
        entity = self.get_entity(person_id) or {}
        scope_id = entity.get("scope_id") or "home-default"
        matching = self._row(
            """SELECT * FROM semantic_claims WHERE person_id = ? AND dimension = ? AND predicate = ?
            AND COALESCE(normalized_value_text, value_text) = ?
            AND status IN ('active', 'pending') ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, revision DESC LIMIT 1""",
            (person_id, dimension, predicate, normalized_value),
        )
        timestamp = now_iso()
        if matching:
            merged_evidence = list(dict.fromkeys(json.loads(matching["evidence_ids_json"] or "[]") + evidence_ids))
            merged_events = list(dict.fromkeys(json.loads(matching["supporting_event_ids_json"] or "[]") + event_ids))
            self.connection.execute(
                """UPDATE semantic_claims SET evidence_ids_json = ?, supporting_event_ids_json = ?, confidence = MAX(confidence, ?), confidence_source = ?,
                normalized_value_text = ?, support_count = ?,
                valid_from = COALESCE(valid_from, ?), valid_to = CASE WHEN valid_to IS NULL THEN ? ELSE valid_to END,
                updated_at = ? WHERE id = ?""",
                (json_value(merged_evidence, []), json_value(merged_events, []), float(confidence or 0), confidence_source,
                 normalized_value, max(int(matching.get("support_count", 1) or 1), len(merged_events) or 1),
                 valid_from, valid_to, timestamp, matching["id"]),
            )
            self.connection.commit()
            return self.get_semantic_claim(matching["id"])
        active = self._row(
            """SELECT * FROM semantic_claims WHERE person_id = ? AND dimension = ? AND predicate = ? AND status = 'active'
            ORDER BY revision DESC LIMIT 1""",
            (person_id, dimension, predicate),
        )
        multi_valued_dimensions = {"activity", "place", "clothing", "capture", "attendance"}
        status = "active" if dimension in multi_valued_dimensions or not active else "pending"
        claim_id = make_id("claim")
        self.connection.execute(
            """INSERT INTO semantic_claims(
                id, scope_id, person_id, dimension, predicate, value_text, normalized_value_text,
                valid_from, valid_to, supporting_event_ids_json, evidence_ids_json, status, confidence,
                confidence_source, support_count, supersedes_claim_id, revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim_id, scope_id, person_id, dimension, predicate, value_text, normalized_value, valid_from, valid_to,
                json_value(event_ids, []), json_value(evidence_ids, []), status, float(confidence or 0), confidence_source,
                max(1, len(event_ids)), active["id"] if active and status == "pending" else None,
                (active["revision"] + 1) if active and status == "pending" else 1, timestamp, timestamp,
            ),
        )
        self.connection.commit()
        return self.get_semantic_claim(claim_id)

    def list_semantic_claims(self, person_id=None, limit=500, include_history=False):
        status_clause = "" if include_history else " AND status IN ('active', 'pending')"
        if person_id:
            rows = self._rows(f"SELECT * FROM semantic_claims WHERE person_id = ?{status_clause} ORDER BY updated_at DESC LIMIT ?", (person_id, limit))
        else:
            rows = self._rows(f"SELECT * FROM semantic_claims WHERE 1 = 1{status_clause} ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [self._decode(row, ["supporting_event_ids_json", "evidence_ids_json"]) for row in rows]

    def start_rebuild(self, run_version, scope):
        timestamp = now_iso()
        run_id = make_id("rebuild")
        self.connection.execute(
            "INSERT INTO rebuild_runs(id, run_version, scope, status, stats_json, created_at, updated_at) VALUES (?, ?, ?, 'running', '{}', ?, ?)",
            (run_id, run_version, scope, timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_rebuild(run_id)

    def finish_rebuild(self, run_id, status, stats=None, error=None):
        self.connection.execute("UPDATE rebuild_runs SET status = ?, stats_json = ?, error = ?, updated_at = ? WHERE id = ?", (status, json_value(stats, {}), error, now_iso(), run_id))
        self.connection.commit()
        return self.get_rebuild(run_id)

    def get_rebuild(self, run_id):
        return self._decode(self._row("SELECT * FROM rebuild_runs WHERE id = ?", (run_id,)), ["stats_json"])

    def create_query_gap(self, query, missing_dimension, candidate_asset_ids=None, evidence_ids=None, scope_id=None):
        gap_id = make_id("gap")
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO query_gaps(id, scope_id, query, missing_dimension, candidate_asset_ids_json, evidence_ids_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (gap_id, scope_id or "home-default", query, missing_dimension, json_value(candidate_asset_ids, []), json_value(evidence_ids, []), timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_query_gap(gap_id)

    def get_query_gap(self, gap_id):
        return self._decode(self._row("SELECT * FROM query_gaps WHERE id = ?", (gap_id,)), ["candidate_asset_ids_json", "evidence_ids_json"])

    def list_query_gaps(self, status=None, limit=200, scope_id=None):
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)
        rows = self._rows(f"SELECT * FROM query_gaps{where} ORDER BY updated_at DESC LIMIT ?", params)
        return [self._decode(row, ["candidate_asset_ids_json", "evidence_ids_json"]) for row in rows]

    def add_memory_feedback(self, gap_id=None, user_id=None, accepted_answer=None, correction=None, target_claim_id=None,
                            target_entity_id=None, target_event_id=None, target_property_key=None):
        if not gap_id and not any((target_claim_id, target_entity_id, target_event_id)):
            raise ValueError("feedback requires a query gap or an explicit memory target")
        if target_entity_id and not self.get_entity(target_entity_id):
            raise KeyError(target_entity_id)
        if target_event_id and not self.get_event(target_event_id):
            raise KeyError(target_event_id)
        if target_property_key and not target_entity_id:
            raise ValueError("a property feedback target requires an entity")
        feedback_id = make_id("feedback")
        self.connection.execute(
            """INSERT INTO memory_feedback(id, query_gap_id, user_id, accepted_answer, correction, target_claim_id,
            target_entity_id, target_event_id, target_property_key, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (feedback_id, gap_id, user_id, accepted_answer, correction, target_claim_id, target_entity_id,
             target_event_id, target_property_key, now_iso()),
        )
        if gap_id:
            self.connection.execute("UPDATE query_gaps SET status = 'resolved', resolution = ?, updated_at = ? WHERE id = ?", (correction or accepted_answer or "confirmed", now_iso(), gap_id))
        self.connection.commit()
        return self._row("SELECT * FROM memory_feedback WHERE id = ?", (feedback_id,))

    def get_dialogue_state(self, conversation_id, scope_id=None):
        row = self._row("SELECT * FROM dialogue_states WHERE conversation_id = ?", (conversation_id,))
        if not row or (scope_id and row.get("scope_id") != scope_id):
            return None
        try:
            return json.loads(row.get("state_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            return None

    def save_dialogue_state(self, conversation_id, scope_id, state):
        allowed = {"scope_id", "active_event_ids", "active_entity_ids", "evidence_ids", "unresolved_ambiguity"}
        value = {key: state.get(key) for key in allowed if key in state}
        self.connection.execute(
            """INSERT INTO dialogue_states(conversation_id, scope_id, state_json, updated_at) VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET scope_id = excluded.scope_id,
            state_json = excluded.state_json, updated_at = excluded.updated_at""",
            (conversation_id, scope_id or "home-default", json_value(value, {}), now_iso()),
        )
        self.connection.commit()
        return self.get_dialogue_state(conversation_id, scope_id)

    def list_person_event_memory(self, person_id, scope_id=None):
        clauses = ["person_id = ?", "status = 'active'"]
        params = [person_id]
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        rows = self._rows(
            f"SELECT * FROM person_event_memory WHERE {' AND '.join(clauses)} ORDER BY time_start DESC, updated_at DESC",
            params,
        )
        return [self._decode(row, ["co_person_ids_json", "evidence_ids_json"]) for row in rows]

    def list_person_patterns(self, person_id, scope_id=None):
        clauses = ["person_id = ?", "status = 'active'"]
        params = [person_id]
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        rows = self._rows(
            f"SELECT * FROM person_patterns WHERE {' AND '.join(clauses)} ORDER BY support_count DESC, last_seen DESC",
            params,
        )
        return [self._decode(row, ["supporting_event_ids_json", "evidence_ids_json"]) for row in rows]

    def get_person_memory(self, person_id, scope_id=None):
        profile = self.get_semantic_profile(person_id)
        return {
            "profile": profile,
            "event_memory": self.list_person_event_memory(person_id, scope_id),
            "patterns": self.list_person_patterns(person_id, scope_id),
            "claims": self.list_semantic_claims(person_id),
        }

    def _rebuild_person_event_memory(self, person_id, event_ids, mentions_by_event):
        self.connection.execute("DELETE FROM person_event_memory WHERE person_id = ?", (person_id,))
        entity = self.get_entity(person_id) or {}
        scope_id = entity.get("scope_id") or "home-default"
        for event_id in event_ids:
            event = self.get_event(event_id)
            if not event:
                continue
            evidence_ids = list(dict.fromkeys(mentions_by_event.get(event_id, [])))
            roles = self._rows(
                "SELECT role, evidence_ids_json FROM event_participants WHERE event_id = ? AND person_id = ?",
                (event_id, person_id),
            )
            if roles:
                role_values = roles
                for role_row in roles:
                    evidence_ids.extend(json.loads(role_row["evidence_ids_json"] or "[]"))
            else:
                role_values = [{"role": "visible_subject", "evidence_ids_json": "[]"}]
            co_people = [
                item["person_id"] for item in self._rows(
                    "SELECT person_id FROM event_participants WHERE event_id = ? AND person_id != ? AND role = 'visible_subject'",
                    (event_id, person_id),
                )
            ]
            for role_row in role_values:
                evidence = list(dict.fromkeys(evidence_ids + json.loads(role_row["evidence_ids_json"] or "[]")))
                self.connection.execute(
                    """INSERT INTO person_event_memory(
                        id, scope_id, person_id, event_id, role, activity_text, place_text,
                        time_start, time_end, co_person_ids_json, evidence_ids_json, confidence,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        make_id("pem"), scope_id, person_id, event_id, role_row["role"],
                        event.get("activity") or event.get("event_type") or "",
                        event.get("place") or "", event.get("time_start"), event.get("time_end"),
                        json_value(list(dict.fromkeys(co_people)), []), json_value(evidence, []),
                        float(event.get("confidence", 0.5) or 0.5), now_iso(), now_iso(),
                    ),
                )

    def _rebuild_person_patterns(self, person_id):
        self.connection.execute("DELETE FROM person_patterns WHERE person_id = ?", (person_id,))
        entity = self.get_entity(person_id) or {}
        scope_id = entity.get("scope_id") or "home-default"
        grouped = {}

        def add_pattern(pattern_type, value_text, event_id, evidence_ids, when, confidence):
            value_text = str(value_text or "").strip()
            if not value_text:
                return
            normalized = normalize_clothing(value_text) if pattern_type == "clothing" else value_text
            key = (pattern_type, normalized)
            item = grouped.setdefault(key, {"events": [], "evidence": [], "times": [], "confidence": 0.0})
            if event_id and event_id not in item["events"]:
                item["events"].append(event_id)
            item["evidence"].extend(evidence_ids or [])
            if when:
                item["times"].append(when)
            item["confidence"] = max(item["confidence"], float(confidence or 0))

        for row in self.list_person_event_memory(person_id):
            for event_id in [row["event_id"]]:
                add_pattern("activity", row["activity_text"], event_id, row["evidence_ids_json"], row.get("time_start"), row.get("confidence"))
                add_pattern("place", row["place_text"], event_id, row["evidence_ids_json"], row.get("time_start"), row.get("confidence"))
            for co_person_id in row["co_person_ids_json"]:
                co_person = self.get_entity(co_person_id)
                add_pattern("co_person", co_person["canonical_name"] if co_person else co_person_id, row["event_id"], row["evidence_ids_json"], row.get("time_start"), row.get("confidence"))

        for appearance in self.list_person_appearance_evidence(person_id):
            event_ids = [row["event_id"] for row in self._rows("SELECT event_id FROM event_observations WHERE observation_id = ?", (appearance["observation_id"],))]
            for clothing in appearance.get("clothing_json", []):
                for event_id in event_ids or [None]:
                    add_pattern("clothing", clothing, event_id, [appearance["id"]], appearance.get("created_at"), appearance.get("confidence"))

        for (pattern_type, normalized), item in grouped.items():
            times = sorted(item["times"])
            events = list(dict.fromkeys(item["events"]))
            evidence = list(dict.fromkeys(item["evidence"]))
            self.connection.execute(
                """INSERT INTO person_patterns(
                    id, scope_id, person_id, pattern_type, value_text, normalized_key,
                    supporting_event_ids_json, evidence_ids_json, support_count, first_seen,
                    last_seen, confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    make_id("pattern"), scope_id, person_id, pattern_type, normalized, normalized,
                    json_value(events, []), json_value(evidence, []), max(len(events), len(evidence), 1),
                    times[0] if times else None, times[-1] if times else None, item["confidence"], now_iso(), now_iso(),
                ),
            )

    def rebuild_person_memory(self, person_id):
        entity = self.get_entity(person_id)
        if not entity:
            return None
        mentions = self._rows("SELECT DISTINCT observation_id FROM entity_mentions WHERE entity_id = ?", (person_id,))
        observation_ids = [row["observation_id"] for row in mentions]
        mentions_by_event = {}
        for observation_id in observation_ids:
            for row in self._rows("SELECT event_id FROM event_observations WHERE observation_id = ?", (observation_id,)):
                mentions_by_event.setdefault(row["event_id"], []).append(observation_id)
        event_ids = self.entity_event_ids(person_id)
        event_ids.extend(
            row["event_id"] for row in self._rows("SELECT DISTINCT event_id FROM event_participants WHERE person_id = ?", (person_id,))
            if row["event_id"] not in event_ids
        )
        event_ids = list(dict.fromkeys(event_ids))
        self._rebuild_person_event_memory(person_id, event_ids, mentions_by_event)
        # Derived claims are a rebuildable projection. User-confirmed identity
        # claims and their audit history remain intact.
        self.connection.execute(
            "UPDATE semantic_claims SET status = 'superseded', updated_at = ? WHERE person_id = ? AND confidence_source = 'derived' AND status IN ('active', 'pending')",
            (now_iso(), person_id),
        )
        activities = []
        places = []
        event_rows = [self.get_event(event_id) for event_id in event_ids]
        event_rows = [event for event in event_rows if event]
        all_evidence_ids = list(observation_ids)
        for event in event_rows:
            # An event establishes attendance, activity, and place. Appearance
            # claims must be grounded in observations where this person was
            # actually identified, never in every person visible at the event.
            event_observation_ids = list(dict.fromkeys(mentions_by_event.get(event["id"], [])))
            participation = self._row(
                "SELECT evidence_ids_json FROM event_participants WHERE event_id = ? AND person_id = ? AND role = 'visible_subject'",
                (event["id"], person_id),
            )
            if not event_observation_ids and participation:
                event_observation_ids = json.loads(participation["evidence_ids_json"] or "[]")
            all_evidence_ids.extend(event_observation_ids)
            event_confidence = float(event.get("confidence", 0.5) or 0.5)
            activity = str(event.get("activity") or "").strip()
            if activity and activity not in activities:
                activities.append(activity)
                self.maintain_semantic_claim(
                    person_id, "activity", "参与", activity, event_observation_ids, [event["id"]], event_confidence,
                    event.get("time_start"), event.get("time_end"), confidence_source="derived",
                )
            place = str(event.get("place") or "").strip()
            if place and place not in places:
                places.append(place)
                self.maintain_semantic_claim(
                    person_id, "place", "出现在", place, event_observation_ids, [event["id"]], event_confidence,
                    event.get("time_start"), event.get("time_end"), confidence_source="derived",
                )
        participation_rows = self._rows("SELECT * FROM event_participants WHERE person_id = ?", (person_id,))
        for participation in participation_rows:
            event = self.get_event(participation["event_id"])
            if not event:
                continue
            evidence_ids = json.loads(participation["evidence_ids_json"] or "[]")
            if participation["role"] == "captured_by":
                self.maintain_semantic_claim(
                    person_id, "capture", "拍摄", event.get("title") or event.get("event_type") or "家庭事件",
                    evidence_ids, [event["id"]], participation.get("confidence", 0.5), event.get("time_start"), event.get("time_end"),
                    confidence_source="derived",
                )
                all_evidence_ids.extend(evidence_ids)
        appearance_evidence = self.list_person_appearance_evidence(person_id)
        clothing_values = []
        for appearance in appearance_evidence:
            evidence_id = appearance["id"]
            observation_id = appearance["observation_id"]
            event_ids_for_observation = [
                row["event_id"]
                for row in self._rows("SELECT event_id FROM event_observations WHERE observation_id = ?", (observation_id,))
            ]
            event_ids_for_observation = [event_id for event_id in event_ids_for_observation if event_id in event_ids]
            observation = self.get_observation(observation_id) or {}
            valid_from = observation.get("captured_at")
            for clothing in appearance.get("clothing_json", []):
                if clothing not in clothing_values:
                    clothing_values.append(clothing)
                self.maintain_semantic_claim(
                    person_id, "clothing", "穿着", clothing, [evidence_id], event_ids_for_observation,
                    appearance.get("confidence", 0.5), valid_from, confidence_source="derived",
                )
            all_evidence_ids.append(evidence_id)
        role = entity.get("family_role")
        if role:
            self.maintain_semantic_claim(person_id, "identity", "家庭角色", role, observation_ids, event_ids, 1.0, confidence_source="user")
        self._rebuild_person_patterns(person_id)
        event_memory = self.list_person_event_memory(person_id)
        patterns = self.list_person_patterns(person_id)
        summary_parts = []
        if observation_ids:
            summary_parts.append(f"出现在{len(observation_ids)}条观察中")
        if event_ids:
            summary_parts.append(f"关联{len(event_ids)}个事件")
        common_places = [value for value, _ in Counter(places).most_common(3)]
        common_activities = [value for value, _ in Counter(activities).most_common(3)]
        if common_places:
            summary_parts.append("常见地点：" + "、".join(common_places))
        if common_activities:
            summary_parts.append("常见活动：" + "、".join(common_activities))
        summary = f"{entity['canonical_name']}已确认，" + "；".join(summary_parts or ["等待新的家庭记忆证据"] ) + "。"
        activity_summary = "；".join(
            f"{event.get('time_start') or '时间未知'} 在{event.get('place') or '地点未知'}：{event.get('activity') or event.get('title') or '家庭记录'}"
            for event in event_rows[:12]
        )
        profile = self.upsert_semantic_profile(person_id, {
            "summary_zh": summary,
            "activity_summary_zh": activity_summary or "暂无已确认活动",
            "place_summary_zh": "、".join(places[:12]),
            "appearance_summary_zh": "、".join(clothing_values[:12]) or "暂无可归属的人物级衣物证据",
            "scope_id": entity.get("scope_id") or "home-default",
            "event_memory_json": event_memory,
            "patterns_json": patterns,
        }, list(dict.fromkeys(all_evidence_ids)))
        for event_id in event_ids:
            self.refresh_event_summary(event_id)
        return {
            "profile": profile,
            "claims": self.list_semantic_claims(person_id),
            "event_memory": event_memory,
            "patterns": patterns,
            "event_ids": event_ids,
            "observation_ids": observation_ids,
        }

    def list_events(self, limit=100, scope_id=None):
        if scope_id:
            rows = self._rows("SELECT * FROM events WHERE status = 'active' AND scope_id = ? ORDER BY time_start DESC LIMIT ?", (scope_id, limit))
        else:
            rows = self._rows("SELECT * FROM events WHERE status = 'active' ORDER BY time_start DESC LIMIT ?", (limit,))
        return [self.get_event(row["id"]) for row in rows]

    def list_trips(self, scope_id=None, status=None):
        clauses = []
        params = []
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._rows("SELECT * FROM trips" + where + " ORDER BY time_start DESC, updated_at DESC", params)
        return [self._decode(row, ["event_ids_json", "place_names_json", "companion_ids_json", "evidence_ids_json"]) for row in rows]

    def get_trip(self, trip_id):
        return self._decode(self._row("SELECT * FROM trips WHERE id = ?", (trip_id,)), ["event_ids_json", "place_names_json", "companion_ids_json", "evidence_ids_json"])

    def list_trip_revisions(self, trip_id):
        return [self._decode(row, ["old_value_json", "new_value_json"]) for row in self._rows(
            "SELECT * FROM trip_revisions WHERE trip_id = ? ORDER BY created_at DESC", (trip_id,),
        )]

    def get_trip_detail(self, trip_id):
        trip = self.get_trip(trip_id)
        if not trip:
            return None
        return {
            "trip": trip,
            "events": [event for event_id in trip["event_ids_json"] if (event := self.get_event(event_id))],
            "revisions": self.list_trip_revisions(trip_id),
        }

    def _record_trip_revision(self, trip_id, action, old_value, new_value, source="user"):
        self.connection.execute(
            """INSERT INTO trip_revisions(id, trip_id, action, old_value_json, new_value_json, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (make_id("trip_revision"), trip_id, action, json_value(old_value, {}), json_value(new_value, {}), source, now_iso()),
        )

    def confirm_trip(self, trip_id, name, trip_type=None):
        trip = self.get_trip(trip_id)
        if not trip:
            raise KeyError(trip_id)
        if trip["status"] != "pending":
            raise ValueError("only pending trip candidates can be confirmed")
        name = str(name or "").strip()
        if not name:
            raise ValueError("trip name is required")
        old_value = {"status": trip["status"], "name": trip["name"], "trip_type": trip.get("trip_type")}
        new_value = {"status": "active", "name": name, "trip_type": str(trip_type or "").strip() or None}
        self.connection.execute(
            "UPDATE trips SET status = 'active', name = ?, trip_type = ?, revision = revision + 1, updated_at = ? WHERE id = ?",
            (new_value["name"], new_value["trip_type"], now_iso(), trip_id),
        )
        self._record_trip_revision(trip_id, "confirmed", old_value, new_value)
        self.connection.commit()
        return self.get_trip(trip_id)

    def reject_trip(self, trip_id):
        trip = self.get_trip(trip_id)
        if not trip:
            raise KeyError(trip_id)
        if trip["status"] != "pending":
            raise ValueError("only pending trip candidates can be rejected")
        self.connection.execute("UPDATE trips SET status = 'rejected', revision = revision + 1, updated_at = ? WHERE id = ?", (now_iso(), trip_id))
        self._record_trip_revision(trip_id, "rejected", {"status": "pending"}, {"status": "rejected"})
        self.connection.commit()
        return self.get_trip(trip_id)

    def _upsert_trip_candidate(self, scope_id, events):
        event_ids = [event["id"] for event in events]
        places = list(dict.fromkeys(str(event.get("place") or "").strip() for event in events if str(event.get("place") or "").strip()))
        evidence_ids = []
        companion_ids = []
        for event in events:
            evidence_ids.extend(event.get("observation_ids") or [])
            companion_ids.extend(item["person_id"] for item in self.list_event_participants(event["id"]) if item.get("person_status") == "confirmed")
        evidence_ids = list(dict.fromkeys(evidence_ids))
        companion_ids = list(dict.fromkeys(companion_ids))
        start, end = events[0].get("time_start"), events[-1].get("time_start")
        name = f"{str(start or '')[:10]} 至 {str(end or '')[:10]} 的行程候选"
        confidence = min(0.9, 0.45 + 0.08 * len(events) + 0.04 * len(places))
        existing = self._row("SELECT * FROM trips WHERE scope_id = ? AND event_ids_json = ? AND status IN ('pending', 'active', 'rejected')", (scope_id, json_value(event_ids, [])))
        timestamp = now_iso()
        if existing and existing["status"] == "pending":
            self.connection.execute(
                "UPDATE trips SET name = ?, time_start = ?, time_end = ?, place_names_json = ?, companion_ids_json = ?, evidence_ids_json = ?, confidence = MAX(confidence, ?), updated_at = ? WHERE id = ?",
                (name, start, end, json_value(places, []), json_value(companion_ids, []), json_value(evidence_ids, []), confidence, timestamp, existing["id"]),
            )
            self.connection.commit()
            return next(item for item in self.list_trips(scope_id, "pending") if item["id"] == existing["id"])
        if existing:
            return None
        trip_id = make_id("trip")
        self.connection.execute(
            """INSERT INTO trips(id, scope_id, name, status, time_start, time_end, event_ids_json, place_names_json,
            companion_ids_json, evidence_ids_json, confidence, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trip_id, scope_id, name, start, end, json_value(event_ids, []), json_value(places, []), json_value(companion_ids, []), json_value(evidence_ids, []), confidence, timestamp, timestamp),
        )
        self.connection.commit()
        return next(item for item in self.list_trips(scope_id, "pending") if item["id"] == trip_id)

    def derive_trip_candidates(self, scope_id=None):
        events = self.list_events(100000, scope_id)
        by_scope = {}
        for event in events:
            captured = parse_time(event.get("time_start"))
            if captured:
                by_scope.setdefault(event.get("scope_id") or "home-default", []).append((captured, event))
        candidates = []
        for current_scope, rows in by_scope.items():
            sequence = []
            for captured, event in sorted(rows, key=lambda item: item[0]):
                if sequence and captured - sequence[-1][0] > timedelta(days=3):
                    candidates.extend(self._derive_trip_candidates_for_sequence(current_scope, [item[1] for item in sequence]))
                    sequence = []
                sequence.append((captured, event))
            candidates.extend(self._derive_trip_candidates_for_sequence(current_scope, [item[1] for item in sequence]))
        return candidates

    def _derive_trip_candidates_for_sequence(self, scope_id, events):
        if len(events) < 2:
            return []
        start = parse_time(events[0].get("time_start"))
        end = parse_time(events[-1].get("time_start"))
        places = {str(event.get("place") or "").strip() for event in events if str(event.get("place") or "").strip()}
        gps_places = [parse_gps_place(event.get("place")) for event in events]
        gps_places = [place for place in gps_places if place]
        material_displacement = any(
            gps_distance_km(first, second) >= TRIP_MIN_GPS_DISPLACEMENT_KM
            for index, first in enumerate(gps_places)
            for second in gps_places[index + 1:]
        )
        cross_day = bool(start and end and start.date() != end.date())
        within_duration = bool(start and end and end - start <= timedelta(days=10))
        if not (cross_day and within_duration and len(places) >= 2 and material_displacement):
            return []
        candidate = self._upsert_trip_candidate(scope_id, events)
        return [candidate] if candidate else []

    def get_event_detail(self, event_id):
        event = self.get_event(event_id)
        if not event:
            return None
        observations = []
        for observation_id in event["observation_ids"]:
            observation = self.get_observation(observation_id)
            if observation:
                observation["asset"] = self.get_asset(observation["asset_id"])
                observations.append(observation)
        facts = [fact for fact in self.list_facts(500) if any(item["id"] in fact["evidence_ids_json"] for item in observations)]
        revisions = self._rows("SELECT * FROM event_revisions WHERE event_id = ? ORDER BY created_at DESC", (event_id,))
        return {"event": event, "observations": observations, "facts": facts, "entities": self.list_event_entities(event_id), "event_revisions": revisions}

    def upsert_event_entity(self, event_id, entity_id, relation, evidence_ids=None, confidence=0.0):
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        existing = self._row(
            "SELECT * FROM event_entities WHERE event_id = ? AND entity_id = ? AND relation = ?",
            (event_id, entity_id, relation),
        )
        timestamp = now_iso()
        if existing:
            merged = list(dict.fromkeys(json.loads(existing["evidence_ids_json"] or "[]") + evidence_ids))
            self.connection.execute(
                "UPDATE event_entities SET evidence_ids_json = ?, confidence = MAX(confidence, ?), updated_at = ? WHERE event_id = ? AND entity_id = ? AND relation = ?",
                (json_value(merged, []), float(confidence or 0), timestamp, event_id, entity_id, relation),
            )
        else:
            self.connection.execute(
                "INSERT INTO event_entities(event_id, entity_id, relation, evidence_ids_json, confidence, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, entity_id, relation, json_value(evidence_ids, []), float(confidence or 0), timestamp, timestamp),
            )
        self.connection.commit()
        return self.list_event_entities(event_id)

    def list_event_entities(self, event_id):
        rows = self._rows(
            """SELECT ee.*, e.id AS entity_id_value, e.entity_type, e.canonical_name, e.status, e.summary
            FROM event_entities ee JOIN entities e ON e.id = ee.entity_id
            WHERE ee.event_id = ? ORDER BY CASE e.entity_type WHEN 'person' THEN 0 WHEN 'place' THEN 1 WHEN 'time' THEN 2 WHEN 'object' THEN 3 WHEN 'emotion' THEN 4 ELSE 5 END, e.canonical_name""",
            (event_id,),
        )
        values = []
        for row in rows:
            value = self._decode(row, ["evidence_ids_json"])
            value["id"] = value.pop("entity_id_value")
            value["evidence_count"] = len(value["evidence_ids_json"])
            values.append(self.public_entity(value))
        for participant in self.list_event_participants(event_id):
            entity = self.get_entity(participant["person_id"])
            if not entity:
                continue
            values.append({
                **self.public_entity(entity), "event_id": event_id, "entity_id": entity["id"],
                "relation": "参与", "role": participant["role"], "confidence": participant["confidence"],
                "evidence_ids_json": participant["evidence_ids_json"], "evidence_count": len(participant["evidence_ids_json"]),
            })
        return sorted(values, key=lambda item: ({"person": 0, "place": 1, "time": 2, "object": 3, "emotion": 4}.get(item["entity_type"], 5), item["canonical_name"]))

    def create_event(self, data):
        event_id = data.get("id") or make_id("evt")
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO events(id, title, event_type, time_start, time_end, place, activity, summary,
            participants_json, confidence, status, revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, ?)""",
            (event_id, data.get("title") or "未命名事件", data.get("event_type", "人工事件"), data.get("time_start"), data.get("time_end"), data.get("place", ""), data.get("activity", ""), data.get("summary", ""), json_value(data.get("participants"), []), float(data.get("confidence", 1) or 1), timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_event(event_id)

    def update_event(self, event_id, fields):
        event = self.get_event(event_id)
        if not event:
            return None
        allowed = {"title", "event_type", "time_start", "time_end", "place", "activity", "summary", "status", "cover_asset_id"}
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return event
        cover_asset_id = values.get("cover_asset_id")
        if cover_asset_id and cover_asset_id not in event["asset_ids"]:
            raise ValueError("event cover must be an asset already linked to this event")
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = list(values.values()) + [now_iso(), event_id]
        self.connection.execute(f"UPDATE events SET {assignments}, revision = revision + 1, updated_at = ? WHERE id = ?", params)
        if cover_asset_id:
            selection = {"source": "user", "asset_id": cover_asset_id, "criteria": {"reason": "user_selected"}}
            self.connection.execute(
                "UPDATE events SET cover_selection_json = ? WHERE id = ?",
                (json_value(selection, {}), event_id),
            )
        for key, value in values.items():
            if event.get(key) != value:
                self.connection.execute(
                    "INSERT INTO event_revisions(id, event_id, field_name, old_value, new_value, source, created_at) VALUES (?, ?, ?, ?, ?, 'user', ?)",
                    (make_id("event_revision"), event_id, key, str(event.get(key) or ""), str(value or ""), now_iso()),
                )
        self.connection.commit()
        return self.get_event(event_id)

    def _fact_row(self, row):
        return self._decode(row, ["evidence_ids_json"])

    def get_fact(self, fact_id):
        return self._fact_row(self._row("SELECT * FROM facts WHERE id = ?", (fact_id,)))

    def list_facts(self, limit=200, scope_id=None):
        rows = self._rows(
            "SELECT * FROM facts" + (" WHERE scope_id = ?" if scope_id else "") + " ORDER BY updated_at DESC LIMIT ?",
            (scope_id, limit) if scope_id else (limit,),
        )
        return [self._fact_row(row) for row in rows]

    def maintain_fact(self, subject, predicate, object_value, evidence_ids, confidence=0.5, scope_id=None):
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        if not scope_id and evidence_ids:
            observation = self.get_observation(evidence_ids[0])
            scope_id = (observation or {}).get("scope_id")
        scope_id = scope_id or "home-default"
        active = self._row(
            "SELECT * FROM facts WHERE scope_id = ? AND subject = ? AND predicate = ? AND status = 'active' ORDER BY revision DESC LIMIT 1",
            (scope_id, subject, predicate),
        )
        if active and active["object"] == object_value:
            existing = json.loads(active["evidence_ids_json"] or "[]")
            merged = list(dict.fromkeys(existing + evidence_ids))
            self.connection.execute("UPDATE facts SET evidence_ids_json = ?, confidence = MAX(confidence, ?), updated_at = ? WHERE id = ?", (json_value(merged, []), float(confidence), now_iso(), active["id"]))
            self.connection.commit()
            return self.get_fact(active["id"])
        status = "pending" if active else "active"
        fact_id = make_id("fact")
        revision = (active["revision"] + 1) if active else 1
        self.connection.execute(
            """INSERT INTO facts(id, scope_id, subject, predicate, object, status, confidence, evidence_ids_json,
            supersedes_fact_id, revision, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fact_id, scope_id, subject, predicate, object_value, status, float(confidence), json_value(evidence_ids, []), active["id"] if active else None, revision, now_iso(), now_iso()),
        )
        self.connection.commit()
        return self.get_fact(fact_id)

    def confirm_fact(self, fact_id):
        fact = self.get_fact(fact_id)
        if not fact:
            raise KeyError(fact_id)
        self.connection.execute("UPDATE facts SET status = 'superseded', updated_at = ? WHERE subject = ? AND predicate = ? AND status = 'active' AND id != ?", (now_iso(), fact["subject"], fact["predicate"], fact_id))
        self.connection.execute("UPDATE facts SET status = 'active', updated_at = ? WHERE id = ?", (now_iso(), fact_id))
        self.connection.commit()
        return self.get_fact(fact_id)

    def purge_unanchored_facts(self, scope_id):
        """Remove model facts that cannot be anchored to a confirmed person."""
        cursor = self.connection.execute(
            """DELETE FROM facts
            WHERE scope_id = ? AND NOT EXISTS (
                SELECT 1 FROM entities
                WHERE entities.scope_id = facts.scope_id
                  AND entities.entity_type = 'person'
                  AND entities.status = 'confirmed'
                  AND entities.canonical_name = facts.subject
            )""",
            (scope_id,),
        )
        self.connection.commit()
        return cursor.rowcount

    def reject_fact(self, fact_id):
        self.connection.execute("UPDATE facts SET status = 'retracted', updated_at = ? WHERE id = ?", (now_iso(), fact_id))
        self.connection.commit()
        return self.get_fact(fact_id)

    def list_persons(self):
        rows = self._rows("SELECT * FROM persons ORDER BY updated_at DESC")
        return [self._decode(row, ["source_json"]) for row in rows]

    def get_person(self, person_id):
        return self._decode(self._row("SELECT * FROM persons WHERE id = ?", (person_id,)), ["source_json"])

    def update_person(self, person_id, name=None, status=None):
        person = self.get_person(person_id)
        if not person:
            return None
        next_name = name or person["name"]
        next_status = status or person["status"]
        timestamp = now_iso()
        self.connection.execute("UPDATE persons SET name = ?, status = ?, updated_at = ? WHERE id = ?", (next_name, next_status, timestamp, person_id))
        entity_id = f"entity_{person_id}"
        entity = self.get_entity(entity_id)
        if not entity:
            self.connection.execute(
                """INSERT INTO entities(
                    id, entity_type, canonical_name, status, family_role, summary,
                    confidence, created_at, updated_at
                ) VALUES (?, 'person', ?, ?, NULL, '由旧人物接口同步', ?, ?, ?)""",
                (entity_id, next_name, next_status, float(person.get("confidence", 0) or 0), timestamp, timestamp),
            )
            entity = self.get_entity(entity_id)
        if entity:
            self.connection.execute(
                """UPDATE entities SET canonical_name = ?, status = ?, confidence = MAX(confidence, ?),
                updated_at = ? WHERE id = ?""",
                (next_name, next_status, float(person.get("confidence", 0) or 0), timestamp, entity_id),
            )
            if entity.get("canonical_name") != next_name:
                self.connection.execute(
                    """INSERT INTO entity_revisions(
                        id, entity_id, field_name, old_value, new_value, source, created_at
                    ) VALUES (?, ?, 'canonical_name', ?, ?, 'legacy_person_api', ?)""",
                    (make_id("entity_revision"), entity_id, entity.get("canonical_name"), next_name, timestamp),
                )
        self.connection.commit()
        return self.get_person(person_id)

    def upsert_person(self, name, confidence=0, status="pending", source=None):
        existing = self._row("SELECT * FROM persons WHERE name = ?", (name,))
        if existing:
            self.connection.execute("UPDATE persons SET confidence = MAX(confidence, ?), status = ?, source_json = ?, updated_at = ? WHERE id = ?", (float(confidence), status if status == "confirmed" else existing["status"], json_value(source, {}), now_iso(), existing["id"]))
            self.connection.commit()
            return self._row("SELECT * FROM persons WHERE id = ?", (existing["id"],))
        person_id = make_id("person")
        self.connection.execute("INSERT INTO persons(id, name, status, confidence, source_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (person_id, name, status, float(confidence), json_value(source, {}), now_iso(), now_iso()))
        self.connection.commit()
        return self._row("SELECT * FROM persons WHERE id = ?", (person_id,))

    @staticmethod
    def _normalise_vector(vector):
        values = [float(value) for value in (vector or [])]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else []

    @staticmethod
    def _cosine(left, right):
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    def upsert_vector(self, space, source_type, source_id, vector, model_name, metadata=None):
        values = self._normalise_vector(vector)
        if not values:
            return None
        metadata = dict(metadata or {})
        scope_id = metadata.get("scope_id")
        if not scope_id and metadata.get("asset_id"):
            scope_id = (self.get_asset(metadata["asset_id"]) or {}).get("scope_id")
        if not scope_id and source_type == "asset":
            scope_id = (self.get_asset(source_id) or {}).get("scope_id")
        if not scope_id and source_type == "observation":
            scope_id = (self.get_observation(source_id) or {}).get("scope_id")
        if not scope_id and source_type == "event":
            scope_id = (self.get_event(source_id) or {}).get("scope_id")
        metadata["scope_id"] = scope_id or "home-default"
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO memory_vectors(id, scope_id, space, source_type, source_id, vector_json, model_name, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(space, source_type, source_id, model_name) DO UPDATE SET
                scope_id = excluded.scope_id, vector_json = excluded.vector_json, metadata_json = excluded.metadata_json, updated_at = excluded.updated_at""",
            (make_id("vec"), metadata["scope_id"], space, source_type, source_id, json_value(values, []), model_name, json_value(metadata, {}), timestamp, timestamp),
        )
        self.connection.commit()
        return self._row("SELECT * FROM memory_vectors WHERE space = ? AND source_type = ? AND source_id = ? AND model_name = ?", (space, source_type, source_id, model_name))

    def search_vectors(self, space, vector, limit=10, scope_id=None):
        query = self._normalise_vector(vector)
        if not query:
            return []
        results = []
        rows = self._rows(
            "SELECT * FROM memory_vectors WHERE space = ?" + (" AND scope_id = ?" if scope_id else ""),
            (space, scope_id) if scope_id else (space,),
        )
        for row in rows:
            try:
                candidate = json.loads(row["vector_json"] or "[]")
                score = self._cosine(query, candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            result = self._decode(row, ["metadata_json"])
            result["score"] = score
            results.append(result)
        return sorted(results, key=lambda item: item["score"], reverse=True)[:max(1, limit)]

    def create_entity(self, name, entity_type="person", status="pending", family_role=None, confidence=0.0, summary="", scope_id="home-default"):
        entity_id = make_id("entity")
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO entities(scope_id, id, entity_type, canonical_name, status, family_role, summary, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (scope_id or "home-default", entity_id, entity_type, name, status, family_role, summary, float(confidence or 0), timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_entity(entity_id)

    def _find_or_create_entity(self, name, entity_type, scope_id, confidence, summary):
        name = str(name or "").strip()
        if not name:
            return None
        existing = self._row(
            """SELECT * FROM entities WHERE scope_id = ? AND entity_type = ? AND canonical_name = ?
            AND status != 'rejected' ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'confirmed' THEN 1 ELSE 2 END, updated_at DESC LIMIT 1""",
            (scope_id or "home-default", entity_type, name),
        )
        if existing:
            self.connection.execute(
                "UPDATE entities SET confidence = MAX(confidence, ?), updated_at = ? WHERE id = ?",
                (float(confidence or 0), now_iso(), existing["id"]),
            )
            self.connection.commit()
            return self.get_entity(existing["id"])
        return self.create_entity(name, entity_type, "pending", confidence=confidence, summary=summary, scope_id=scope_id)

    def maintain_observation_entities(self, observation_id, event_id=None):
        """Create evidence-backed non-person entities from existing observations."""
        observation = self.get_observation(observation_id)
        if not observation:
            return []
        asset = self.get_asset(observation["asset_id"]) or {}
        scope_id = observation.get("scope_id") or asset.get("scope_id") or "home-default"
        raw = observation.get("raw") or {}
        extracted = raw.get("gamma") if isinstance(raw.get("gamma"), dict) else raw
        canonical = observation.get("canonical") or {}
        analysis = {**extracted, **canonical}
        normalized = normalize_semantic_analysis(analysis)
        semantic = normalized.get("semantic") or {}
        semantic_place = semantic.get("place") if isinstance(semantic.get("place"), dict) else {}
        semantic_objects = semantic.get("objects") if isinstance(semantic.get("objects"), list) else []
        semantic_atmosphere = semantic.get("atmosphere") if isinstance(semantic.get("atmosphere"), dict) else {}
        captured_at = parse_time(observation.get("captured_at") or asset.get("captured_at"))
        captured_day = captured_at.date().isoformat() if captured_at else ""
        gps_place = parse_gps_place(asset.get("captured_location"))
        visual_place = str(observation.get("place") or "").strip()
        has_semantic_place = bool(semantic.get("available"))
        selected_scene = str(semantic_place.get("primary") or "").strip() if has_semantic_place else ""
        # GPS remains a location anchor only. A semantic primary is the stable
        # place entity; free text is retained as evidence, never as a GPS name.
        place_name = selected_scene or visual_place or "其他或不确定"
        scene_type = selected_scene or (visual_place if not has_semantic_place else "")
        object_specs = [
            item for item in semantic_objects
            if isinstance(item, dict) and (item.get("label") or item.get("primary"))
        ]
        object_names = [str(item.get("label") or item.get("primary")).strip() for item in object_specs]
        if not object_names:
            object_names = [str(item).strip() for item in (observation.get("objects") or []) if str(item).strip()]
        atmosphere_names = [str(item).strip() for item in (semantic_atmosphere.get("labels") or []) if str(item).strip()]
        if not atmosphere_names:
            atmosphere_names = [
                normalize_mood(value)
                for value in (extracted.get("emotions") or raw.get("emotions") or [])
                if normalize_mood(value) in ATMOSPHERE_PRIMARY_TYPES
            ]
        raw_atmosphere_labels = [
            str(item).strip() for item in (normalized.get("raw_labels") or {}).get("atmosphere", []) if str(item).strip()
        ]
        values = [
            ("place", place_name, "由图片观察或采集地点维护"),
            ("object", object_names, "由图片观察到的物体"),
            ("atmosphere", atmosphere_names, "由图片观察到的画面氛围"),
            ("time", captured_day, "由原始拍摄时间维护") if captured_day else ("time", [], ""),
        ]
        entities = []
        place_entity = None
        time_entity = None
        for entity_type, names, summary in values:
            for name in names if isinstance(names, list) else [names]:
                entity = self._find_or_create_entity(name, entity_type, scope_id, observation.get("confidence", 0), summary)
                if not entity:
                    continue
                self.connection.execute(
                    """INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at)
                    VALUES (?, ?, ?, 'observation_extraction', ?) ON CONFLICT(entity_id, observation_id)
                    DO UPDATE SET confidence = MAX(confidence, excluded.confidence)""",
                    (entity["id"], observation_id, float(observation.get("confidence", 0) or 0), now_iso()),
                )
                if entity_type == "place":
                    place_entity = entity
                if entity_type == "time":
                    time_entity = entity
                entities.append(entity)
        evidence_ids = [observation_id, event_id] if event_id else [observation_id]
        if place_entity:
            if gps_place:
                self.maintain_entity_property(
                    place_entity["id"], "geo", {"latitude": gps_place[0], "longitude": gps_place[1]},
                    observation.get("confidence", 0), evidence_ids, "asset_gps",
                )
            if scene_type:
                self.maintain_entity_property(
                    place_entity["id"], "scene_type", scene_type,
                    observation.get("confidence", 0), evidence_ids, "observation_extraction",
                )
            if has_semantic_place:
                self.maintain_entity_property(
                    place_entity["id"], "semantic_primary", selected_scene or "其他或不确定",
                    observation.get("confidence", 0), evidence_ids, "semantic_taxonomy_v1",
                )
                self.maintain_entity_property_values(
                    place_entity["id"], "semantic_details", semantic_place.get("details") or [],
                    observation.get("confidence", 0), evidence_ids, "semantic_taxonomy_v1",
                )
            if visual_place and (visual_place != scene_type or has_semantic_place):
                self.maintain_entity_property_values(
                    place_entity["id"], "visual_place_descriptions", [visual_place],
                    observation.get("confidence", 0), evidence_ids, "observation_extraction",
                )
            # A scene-type backfill must replace the former free-text place
            # projection for this observation. The original text remains on
            # the selected scene entity, while stale place cards and event
            # links no longer remain visible beside it.
            stale_places = self._rows(
                """SELECT eob.entity_id FROM entity_observations eob JOIN entities e ON e.id = eob.entity_id
                WHERE eob.observation_id = ? AND e.entity_type = 'place' AND eob.entity_id != ?
                AND eob.source = 'observation_extraction'""",
                (observation_id, place_entity["id"]),
            )
            for stale in stale_places:
                stale_id = stale["entity_id"]
                self.connection.execute(
                    "DELETE FROM entity_observations WHERE entity_id = ? AND observation_id = ? AND source = 'observation_extraction'",
                    (stale_id, observation_id),
                )
                if event_id:
                    link = self._row("SELECT * FROM event_entities WHERE event_id = ? AND entity_id = ?", (event_id, stale_id))
                    if link:
                        linked_evidence = [item for item in json.loads(link["evidence_ids_json"] or "[]") if item != observation_id]
                        if linked_evidence:
                            self.connection.execute(
                                "UPDATE event_entities SET evidence_ids_json = ?, updated_at = ? WHERE event_id = ? AND entity_id = ?",
                                (json_value(linked_evidence, []), now_iso(), event_id, stale_id),
                            )
                        else:
                            self.connection.execute("DELETE FROM event_entities WHERE event_id = ? AND entity_id = ?", (event_id, stale_id))
        for entity in entities:
            if entity["entity_type"] != "object":
                continue
            self.maintain_entity_property(
                entity["id"], "label", entity["canonical_name"], observation.get("confidence", 0), evidence_ids,
                    "observation_extraction",
                )
            self.maintain_entity_property(
                entity["id"], "category", object_category(entity["canonical_name"]), observation.get("confidence", 0), evidence_ids,
                "object_taxonomy_v1",
            )
            spec = next((item for item in object_specs if str(item.get("label") or item.get("primary")).strip() == entity["canonical_name"]), None)
            if spec:
                self.maintain_entity_property(
                    entity["id"], "semantic_primary", spec.get("primary") or "其他或不确定",
                    observation.get("confidence", 0), evidence_ids, "semantic_taxonomy_v1",
                )
                self.maintain_entity_property_values(
                    entity["id"], "semantic_details", spec.get("details") or [],
                    observation.get("confidence", 0), evidence_ids, "semantic_taxonomy_v1",
                )
        for entity in entities:
            if entity["entity_type"] not in {"emotion", "atmosphere"}:
                continue
            raw_moods = [
                str(value).strip()
                for value in (extracted.get("emotions") or raw.get("emotions") or [])
                if normalize_mood(value) == entity["canonical_name"] and str(value).strip()
            ]
            property_key = "atmosphere_label" if entity["entity_type"] == "atmosphere" else "mood_label"
            self.maintain_entity_property(entity["id"], property_key, entity["canonical_name"], observation.get("confidence", 0), evidence_ids, "semantic_taxonomy_v1" if entity["entity_type"] == "atmosphere" else "mood_normalization_v1")
            if entity["entity_type"] == "atmosphere":
                self.maintain_entity_property_values(entity["id"], "semantic_details", semantic_atmosphere.get("details") or [], observation.get("confidence", 0), evidence_ids, "semantic_taxonomy_v1")
            self.maintain_entity_property_values(
                entity["id"], "raw_atmosphere_labels" if entity["entity_type"] == "atmosphere" else "raw_mood_labels", raw_atmosphere_labels if entity["entity_type"] == "atmosphere" else raw_moods, observation.get("confidence", 0), evidence_ids,
                "semantic_taxonomy_v1" if entity["entity_type"] == "atmosphere" else "mood_normalization_v1",
            )
        if time_entity and captured_at:
            semantics = time_semantics(captured_at)
            for key in ("date", "year", "month", "season"):
                self.maintain_entity_property(
                    time_entity["id"], key, semantics[key], observation.get("confidence", 0), evidence_ids,
                    "timestamp_derivation",
                )
            self.maintain_entity_property_values(
                time_entity["id"], "part_of_day", [semantics["part_of_day"]],
                observation.get("confidence", 0), evidence_ids, "timestamp_derivation",
            )
        for entity in entities:
            if event_id:
                relation = {"place": "地点", "time": "时间", "object": "包含物件", "emotion": "情感氛围", "atmosphere": "画面氛围"}.get(entity["entity_type"], "关联实体")
                self.upsert_event_entity(event_id, entity["id"], relation, [observation_id], observation.get("confidence", 0))
            if place_entity and entity["id"] != place_entity["id"]:
                self.create_relationship(
                    entity["id"], "出现在", place_entity["id"], [observation_id, event_id] if event_id else [observation_id],
                    observation.get("confidence", 0),
                )
            if time_entity and entity["id"] != time_entity["id"]:
                self.create_relationship(
                    entity["id"], "记录于", time_entity["id"], [observation_id, event_id] if event_id else [observation_id],
                    observation.get("confidence", 0),
                )
        self.connection.commit()
        return entities

    def reindex_observation_entities(self, scope_id=None):
        """Idempotently rebuild evidence-backed non-person entity links."""
        rows = self._rows(
            "SELECT id FROM observations" + (" WHERE scope_id = ?" if scope_id else ""),
            (scope_id,) if scope_id else (),
        )
        total = 0
        for row in rows:
            event_row = self._row("SELECT event_id FROM event_observations WHERE observation_id = ? LIMIT 1", (row["id"],))
            total += len(self.maintain_observation_entities(row["id"], event_row["event_id"] if event_row else None))
        mood_cleanup = self.normalize_legacy_mood_entities(scope_id)
        return {"observations": len(rows), "entity_links": total, "normalized_moods": mood_cleanup["normalized"], "retired_unclassified_moods": mood_cleanup["retired"], "scope_id": scope_id}

    def normalize_legacy_mood_entities(self, scope_id=None):
        """Retire prior model-only mood labels after moving every evidence edge."""
        clauses = ["entity_type = 'emotion'", "status != 'rejected'"]
        params = []
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        legacy_entities = self._rows("SELECT * FROM entities WHERE " + " AND ".join(clauses), params)
        normalized_count = 0
        retired_count = 0
        for legacy in legacy_entities:
            normalized_name = normalize_mood(legacy["canonical_name"])
            if self._row("SELECT 1 FROM entity_properties WHERE entity_id = ? AND source = 'user' LIMIT 1", (legacy["id"],)):
                continue
            atmosphere_name = ATMOSPHERE_PRIMARY_ALIASES.get(normalized_name, normalized_name)
            if atmosphere_name not in ATMOSPHERE_PRIMARY_TYPES:
                self.connection.execute("DELETE FROM event_entities WHERE entity_id = ?", (legacy["id"],))
                self.connection.execute("DELETE FROM entity_observations WHERE entity_id = ?", (legacy["id"],))
                self.connection.execute("DELETE FROM relationships WHERE subject_entity_id = ? OR object_entity_id = ?", (legacy["id"], legacy["id"]))
                self.connection.execute("UPDATE entities SET status = 'rejected', summary = ?, updated_at = ? WHERE id = ?", ("原始模型标签未进入受控氛围词表，保留在观察证据中", now_iso(), legacy["id"]))
                self.connection.commit()
                retired_count += 1
                continue
            target = self._find_or_create_entity(
                atmosphere_name, "atmosphere", legacy.get("scope_id"), legacy.get("confidence", 0), "由图片观察到的画面氛围",
            )
            evidence_rows = self._rows("SELECT * FROM entity_observations WHERE entity_id = ?", (legacy["id"],))
            if not evidence_rows:
                for observation in self._rows("SELECT id, raw_json FROM observations"):
                    try:
                        payload = json.loads(observation.get("raw_json") or "{}")
                    except (TypeError, json.JSONDecodeError):
                        payload = {}
                    gamma = payload.get("gamma") if isinstance(payload.get("gamma"), dict) else payload
                    labels = gamma.get("emotions") if isinstance(gamma, dict) else []
                    if legacy["canonical_name"] in (labels or []):
                        evidence_rows.append({
                            "observation_id": observation["id"],
                            "confidence": legacy.get("confidence", 0),
                            "source": "legacy_mood_migration",
                            "created_at": now_iso(),
                        })
            evidence_ids = [row["observation_id"] for row in evidence_rows]
            for evidence in evidence_rows:
                self.connection.execute(
                    """INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at)
                    VALUES (?, ?, ?, ?, ?) ON CONFLICT(entity_id, observation_id)
                    DO UPDATE SET confidence = MAX(confidence, excluded.confidence)""",
                    (target["id"], evidence["observation_id"], evidence["confidence"], evidence["source"], evidence["created_at"]),
                )
            for link in self._rows("SELECT * FROM event_entities WHERE entity_id = ?", (legacy["id"],)):
                self.upsert_event_entity(link["event_id"], target["id"], link["relation"], json.loads(link["evidence_ids_json"] or "[]"), link["confidence"])
            for relationship in self._rows(
                "SELECT * FROM relationships WHERE (subject_entity_id = ? OR object_entity_id = ?) AND status != 'retracted'",
                (legacy["id"], legacy["id"]),
            ):
                subject_id = target["id"] if relationship["subject_entity_id"] == legacy["id"] else relationship["subject_entity_id"]
                object_id = target["id"] if relationship["object_entity_id"] == legacy["id"] else relationship["object_entity_id"]
                self.create_relationship(
                    subject_id, relationship["predicate"], object_id, json.loads(relationship["evidence_ids_json"] or "[]"),
                    relationship["confidence"], relationship["status"],
                )
            self.maintain_entity_property(target["id"], "atmosphere_label", atmosphere_name, legacy.get("confidence", 0), evidence_ids, "semantic_taxonomy_v1")
            self.maintain_entity_property_values(target["id"], "raw_atmosphere_labels", [legacy["canonical_name"]], legacy.get("confidence", 0), evidence_ids, "semantic_taxonomy_v1")
            self._record_entity_revision(target["id"], "atmosphere_normalization", legacy["canonical_name"], atmosphere_name, "semantic_taxonomy_v1", evidence_ids)
            self.connection.execute("DELETE FROM event_entities WHERE entity_id = ?", (legacy["id"],))
            self.connection.execute("DELETE FROM entity_observations WHERE entity_id = ?", (legacy["id"],))
            self.connection.execute("DELETE FROM relationships WHERE subject_entity_id = ? OR object_entity_id = ?", (legacy["id"], legacy["id"]))
            self.connection.execute("UPDATE entities SET status = 'rejected', updated_at = ? WHERE id = ?", (now_iso(), legacy["id"]))
            self.connection.commit()
            normalized_count += 1
        return {"normalized": normalized_count, "retired": retired_count}

    def list_entities(self, status=None, scope_id=None, public=True):
        params = [status] if status else []
        where = "WHERE status = ?" if status else "WHERE status NOT IN ('rejected', 'superseded')"
        if scope_id:
            where += " AND scope_id = ?"
            params.append(scope_id)
        entities = self._rows(f"SELECT * FROM entities {where} ORDER BY updated_at DESC", params)
        for entity in entities:
            entity["cluster_count"] = self.connection.execute("SELECT COUNT(*) FROM face_clusters WHERE entity_id = ?", (entity["id"],)).fetchone()[0]
            entity["mention_count"] = self.connection.execute("SELECT COUNT(*) FROM entity_mentions WHERE entity_id = ?", (entity["id"],)).fetchone()[0]
            entity["evidence_count"] = self.connection.execute("SELECT COUNT(*) FROM entity_observations WHERE entity_id = ?", (entity["id"],)).fetchone()[0]
            entity["relationship_count"] = self.connection.execute("SELECT COUNT(*) FROM relationships WHERE (subject_entity_id = ? OR object_entity_id = ?) AND status != 'retracted'", (entity["id"], entity["id"])).fetchone()[0]
            cluster_rows = self._rows("SELECT member_count, confidence, status FROM face_clusters WHERE entity_id = ?", (entity["id"],))
            if entity["entity_type"] == "person":
                entity["reviewable"] = entity["status"] == "confirmed" or any(
                    row["status"] != "rejected" and int(row.get("member_count", 0) or 0) > 0
                    for row in cluster_rows
                )
                entity["single_sample"] = entity["status"] != "confirmed" and all(
                    int(row.get("member_count", 0) or 0) <= 1 for row in cluster_rows
                )
            else:
                entity["reviewable"] = entity["evidence_count"] > 0
            avatar = self._row(
                """SELECT fi.id FROM face_instances fi JOIN face_clusters fc ON fc.id = fi.cluster_id
                WHERE fc.entity_id = ? AND fc.status != 'rejected' ORDER BY fi.detection_confidence DESC, fi.created_at ASC LIMIT 1""",
                (entity["id"],),
            )
            entity["avatar_face_instance_id"] = avatar["id"] if avatar else None
            preview = self._row(
                """SELECT a.id AS asset_id, a.file_name, a.media_type
                FROM entity_observations eob JOIN observations o ON o.id = eob.observation_id
                JOIN assets a ON a.id = o.asset_id
                WHERE eob.entity_id = ? ORDER BY o.captured_at DESC, o.id DESC LIMIT 1""",
                (entity["id"],),
            )
            entity["preview_asset_id"] = preview["asset_id"] if preview else None
            entity["preview_file_name"] = preview["file_name"] if preview else None
            entity["preview_media_type"] = preview["media_type"] if preview else None
        return [self.public_entity(entity) for entity in entities] if public else entities

    def _semantic_entity_key(self, entity_type, name, entity=None):
        """Return an explainable semantic concept for automatic grouping."""
        label = re.sub(r"\s+", "", str(name or "").strip())
        properties = {
            item["property_key"]: item
            for item in self.list_entity_properties((entity or {}).get("id"))
        } if (entity or {}).get("id") else {}
        primary = properties.get("semantic_primary", {}).get("value")
        if primary and entity_type in {"place", "object", "atmosphere"}:
            return str(primary), {"strategy": "semantic_primary", "matched_label": str(primary)}
        if entity_type == "emotion":
            entity_type = "atmosphere"
        equivalents = SEMANTIC_ENTITY_EQUIVALENTS.get(entity_type, {})
        if label in equivalents:
            return equivalents[label], {"strategy": "controlled_equivalence", "matched_label": label}
        for term, normalized in equivalents.items():
            if term in label:
                return normalized, {"strategy": "controlled_equivalence", "matched_label": term}
        concepts = SEMANTIC_PLACE_CONCEPTS if entity_type == "place" else SEMANTIC_OBJECT_CONCEPTS if entity_type == "object" else ()
        for normalized, terms in concepts:
            matched = [term for term in terms if term in label]
            if matched:
                return normalized, {"strategy": "semantic_concept", "matched_label": matched[0]}
        return label, {"strategy": "exact_label", "matched_label": label}

    @staticmethod
    def _merge_candidate_row(row):
        value = dict(row)
        for key, fallback in (("entity_ids_json", []), ("rationale_json", {}), ("evidence_ids_json", [])):
            try:
                value[key.replace("_json", "")] = json.loads(value.pop(key) or json_value(fallback, fallback))
            except (TypeError, json.JSONDecodeError):
                value[key.replace("_json", "")] = fallback
        return value

    def list_entity_merge_candidates(self, scope_id=None, status="pending"):
        clauses, params = [], []
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._rows(
            "SELECT * FROM entity_merge_candidates" + where + " ORDER BY confidence DESC, created_at DESC",
            params,
        )
        return [self._merge_candidate_row(row) for row in rows]

    def list_semantic_entity_groups(self, scope_id=None):
        """Return a read-only semantic projection for browsing and recall.

        A group never replaces its members.  This keeps original labels,
        user-maintained IDs and Observation links stable while avoiding a UI
        made of many cards that describe the same concept.
        """
        grouped = {}
        for entity in self.list_entities(scope_id=scope_id, public=False):
            source_entity_type = entity.get("entity_type")
            entity_type = "atmosphere" if source_entity_type == "emotion" else source_entity_type
            if source_entity_type == "person" or entity.get("status") in {"rejected", "superseded"}:
                continue
            label, rationale = self._semantic_entity_key(source_entity_type, entity.get("canonical_name"), entity)
            key = (entity.get("scope_id") or "home-default", entity_type, label)
            group = grouped.setdefault(key, {
                "id": "semantic_group:" + ":".join((entity.get("scope_id") or "home-default", entity_type, label)),
                "scope_id": entity.get("scope_id") or "home-default",
                "entity_type": entity_type,
                "canonical_name": label,
                "members": [],
                "member_entity_ids": [],
                "source_labels": [],
                "semantic_details": [],
                "evidence_count": 0,
                "relationship_count": 0,
                "confidence": 0.0,
                "preview_asset_id": None,
                "preview_file_name": None,
                "preview_media_type": None,
                "rationale": {"strategy": rationale["strategy"], "normalized_key": label},
            })
            group["members"].append(self.public_entity(entity))
            group["member_entity_ids"].append(entity["id"])
            group["source_labels"].append(entity["canonical_name"])
            properties = {item["property_key"]: item for item in self.list_entity_properties(entity["id"])}
            details = properties.get("semantic_details", {}).get("value") or []
            if not isinstance(details, list):
                details = [details]
            group["semantic_details"] = list(dict.fromkeys(group["semantic_details"] + [str(item) for item in details if str(item).strip()]))
            group["evidence_count"] += int(entity.get("evidence_count", 0) or 0)
            group["relationship_count"] += int(entity.get("relationship_count", 0) or 0)
            group["confidence"] = max(group["confidence"], float(entity.get("confidence", 0) or 0))
            if not group["preview_asset_id"] and entity.get("preview_asset_id"):
                group["preview_asset_id"] = entity["preview_asset_id"]
                group["preview_file_name"] = entity.get("preview_file_name")
                group["preview_media_type"] = entity.get("preview_media_type")
        values = list(grouped.values())
        for group in values:
            group["members"].sort(key=lambda item: (-int(item.get("evidence_count", 0) or 0), item.get("canonical_name", "")))
            group["member_entity_ids"].sort()
            group["source_labels"] = [item["canonical_name"] for item in group["members"]]
            group["is_semantic_cluster"] = len(group["members"]) > 1
        return sorted(values, key=lambda item: (item["entity_type"], -item["evidence_count"], item["canonical_name"]))

    def get_semantic_entity_group(self, group_id, scope_id=None):
        for group in self.list_semantic_entity_groups(scope_id):
            if group["id"] != group_id:
                continue
            details = [self.get_entity_detail(entity_id) for entity_id in group["member_entity_ids"]]
            observations, events, relationships = [], [], []
            seen_observations, seen_events, seen_relationships = set(), set(), set()
            for detail in details:
                for observation in detail.get("observations", []):
                    if observation["id"] not in seen_observations:
                        observations.append(observation); seen_observations.add(observation["id"])
                for event in detail.get("events", []):
                    if event["id"] not in seen_events:
                        events.append(event); seen_events.add(event["id"])
                for relationship in detail.get("relationships", []):
                    if relationship["id"] not in seen_relationships:
                        relationships.append(relationship); seen_relationships.add(relationship["id"])
            return {"group": group, "members": details, "observations": observations, "events": events, "relationships": relationships}
        return None

    def derive_entity_merge_candidates(self, scope_id=None):
        """Create review-only semantic merge candidates within a MemorySpace.

        This never mutates entities, their links, or their names.  A user must
        explicitly accept a candidate before any stable identity can change.
        """
        allowed_types = {"place", "object", "emotion"}
        entities = self.list_entities(scope_id=scope_id, public=False)
        groups = {}
        for entity in entities:
            entity_type = entity.get("entity_type")
            if entity_type not in allowed_types or entity.get("status") == "rejected":
                continue
            key, rationale = self._semantic_entity_key(entity_type, entity.get("canonical_name"), entity)
            if rationale["strategy"] == "exact_label":
                continue
            groups.setdefault((entity.get("scope_id") or "home-default", entity_type, key), []).append((entity, rationale))
        candidates = []
        for (candidate_scope, entity_type, suggested_name), members in groups.items():
            if len(members) < 2:
                continue
            entity_ids = sorted(member[0]["id"] for member in members)
            evidence_ids = []
            source_labels = []
            for entity, _ in members:
                source_labels.append(entity["canonical_name"])
                evidence_ids.extend(row["observation_id"] for row in self._rows(
                    "SELECT observation_id FROM entity_observations WHERE entity_id = ? ORDER BY observation_id", (entity["id"],)
                ))
            evidence_ids = list(dict.fromkeys(evidence_ids))
            rationale = {
                "strategy": "controlled_equivalence",
                "source_labels": source_labels,
                "normalized_key": suggested_name,
                "automatic_merge": False,
            }
            encoded_ids = json_value(entity_ids, [])
            existing = self._row(
                "SELECT * FROM entity_merge_candidates WHERE scope_id = ? AND entity_type = ? AND entity_ids_json = ? AND suggested_name = ?",
                (candidate_scope, entity_type, encoded_ids, suggested_name),
            )
            timestamp = now_iso()
            if existing:
                if existing["status"] == "rejected":
                    continue
                self.connection.execute(
                    "UPDATE entity_merge_candidates SET evidence_ids_json = ?, confidence = MAX(confidence, ?), rationale_json = ?, updated_at = ? WHERE id = ?",
                    (json_value(evidence_ids, []), 0.8, json_value(rationale, {}), timestamp, existing["id"]),
                )
                candidate_id = existing["id"]
            else:
                candidate_id = make_id("entity_merge")
                self.connection.execute(
                    """INSERT INTO entity_merge_candidates(id, scope_id, entity_type, entity_ids_json, suggested_name,
                    rationale_json, confidence, evidence_ids_json, status, revision, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 1, ?, ?)""",
                    (candidate_id, candidate_scope, entity_type, encoded_ids, suggested_name, json_value(rationale, {}), 0.8,
                     json_value(evidence_ids, []), timestamp, timestamp),
                )
            candidates.append(self._merge_candidate_row(self._row("SELECT * FROM entity_merge_candidates WHERE id = ?", (candidate_id,))))
        self.connection.commit()
        return candidates

    def confirm_entity_merge_candidate(self, candidate_id, target_entity_id):
        """Apply an explicit user-approved non-person semantic entity merge."""
        candidate = self._row("SELECT * FROM entity_merge_candidates WHERE id = ?", (candidate_id,))
        target = self.get_entity(target_entity_id)
        if not candidate:
            raise KeyError(candidate_id)
        if candidate["status"] != "pending":
            raise ValueError("only pending entity merge candidates can be confirmed")
        entity_ids = json.loads(candidate["entity_ids_json"] or "[]")
        if target_entity_id not in entity_ids or not target:
            raise ValueError("target entity must belong to this merge candidate")
        if target["entity_type"] == "person" or target["entity_type"] != candidate["entity_type"]:
            raise ValueError("person entities and mismatched entity types cannot be semantically merged")
        if target.get("scope_id") != candidate.get("scope_id"):
            raise ValueError("entity merge must remain within one memory space")
        timestamp = now_iso()
        sources = [self.get_entity(entity_id) for entity_id in entity_ids if entity_id != target_entity_id]
        sources = [entity for entity in sources if entity and entity.get("status") != "superseded"]
        for source in sources:
            if source["entity_type"] != target["entity_type"] or source.get("scope_id") != target.get("scope_id"):
                raise ValueError("candidate contains an incompatible source entity")
            for evidence in self._rows("SELECT * FROM entity_observations WHERE entity_id = ?", (source["id"],)):
                self.connection.execute(
                    """INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at)
                    VALUES (?, ?, ?, ?, ?) ON CONFLICT(entity_id, observation_id)
                    DO UPDATE SET confidence = MAX(confidence, excluded.confidence)""",
                    (target_entity_id, evidence["observation_id"], evidence["confidence"], "user_semantic_merge", timestamp),
                )
            for link in self._rows("SELECT * FROM event_entities WHERE entity_id = ?", (source["id"],)):
                self.upsert_event_entity(link["event_id"], target_entity_id, link["relation"], json.loads(link["evidence_ids_json"] or "[]"), link["confidence"])
            for relationship in self._rows(
                "SELECT * FROM relationships WHERE (subject_entity_id = ? OR object_entity_id = ?) AND status != 'retracted'",
                (source["id"], source["id"]),
            ):
                subject_id = target_entity_id if relationship["subject_entity_id"] == source["id"] else relationship["subject_entity_id"]
                object_id = target_entity_id if relationship["object_entity_id"] == source["id"] else relationship["object_entity_id"]
                if subject_id != object_id:
                    self.create_relationship(subject_id, relationship["predicate"], object_id, json.loads(relationship["evidence_ids_json"] or "[]"), relationship["confidence"], relationship["status"])
            self._record_entity_revision(source["id"], "semantic_merge_target", source["canonical_name"], target_entity_id, "user_semantic_merge", json.loads(candidate["evidence_ids_json"] or "[]"))
            self.connection.execute("DELETE FROM event_entities WHERE entity_id = ?", (source["id"],))
            self.connection.execute("DELETE FROM entity_observations WHERE entity_id = ?", (source["id"],))
            self.connection.execute("DELETE FROM relationships WHERE subject_entity_id = ? OR object_entity_id = ?", (source["id"], source["id"]))
            self.connection.execute("UPDATE entities SET status = 'superseded', updated_at = ? WHERE id = ?", (timestamp, source["id"]))
        self._record_entity_revision(target_entity_id, "semantic_merge_sources", None, json_value([item["id"] for item in sources], []), "user_semantic_merge", json.loads(candidate["evidence_ids_json"] or "[]"))
        self.connection.execute(
            "UPDATE entity_merge_candidates SET status = 'confirmed', target_entity_id = ?, revision = revision + 1, updated_at = ? WHERE id = ?",
            (target_entity_id, timestamp, candidate_id),
        )
        self.connection.commit()
        return self._merge_candidate_row(self._row("SELECT * FROM entity_merge_candidates WHERE id = ?", (candidate_id,)))

    def reject_entity_merge_candidate(self, candidate_id):
        candidate = self._row("SELECT * FROM entity_merge_candidates WHERE id = ?", (candidate_id,))
        if not candidate:
            raise KeyError(candidate_id)
        if candidate["status"] != "pending":
            raise ValueError("only pending entity merge candidates can be rejected")
        self.connection.execute(
            "UPDATE entity_merge_candidates SET status = 'rejected', revision = revision + 1, updated_at = ? WHERE id = ?",
            (now_iso(), candidate_id),
        )
        self.connection.commit()
        return self._merge_candidate_row(self._row("SELECT * FROM entity_merge_candidates WHERE id = ?", (candidate_id,)))

    def get_entity(self, entity_id):
        return self._row("SELECT * FROM entities WHERE id = ?", (entity_id,))

    def person_aliases(self, entity_id):
        """Current user-maintained alias list for a person entity, or []."""
        row = self._current_entity_property(entity_id, "aliases")
        if not row:
            return []
        value = self._property_row(row).get("value")
        return list(value) if isinstance(value, list) else ([value] if value else [])

    def find_confirmed_person_by_name(self, scope_id, name):
        """Return the confirmed person entity in scope_id whose canonical name or
        an alias matches `name`. Returns {entity_id, via_alias} or None."""
        name = str(name or "").strip()
        if not name:
            return None
        row = self._row(
            """SELECT * FROM entities WHERE scope_id = ? AND entity_type = 'person' AND status = 'confirmed'
            AND canonical_name = ? ORDER BY updated_at DESC LIMIT 1""",
            (scope_id, name),
        )
        if row:
            return {"entity_id": row["id"], "via_alias": False}
        for candidate in self._rows(
            """SELECT id FROM entities WHERE scope_id = ? AND entity_type = 'person' AND status = 'confirmed'""",
            (scope_id,),
        ):
            if name in self.person_aliases(candidate["id"]):
                return {"entity_id": candidate["id"], "via_alias": True}
        return None

    def set_person_aliases(self, entity_id, aliases):
        """Replace the user-maintained alias list. Old values keep revision audit."""
        cleaned = list(dict.fromkeys(str(item or "").strip() for item in (aliases or []) if str(item or "").strip()))
        if not self.get_entity(entity_id):
            return []
        self.set_entity_property(entity_id, "aliases", cleaned)
        return self.person_aliases(entity_id)

    def rename_person(self, entity_id, new_name):
        """Rename a confirmed person globally: update canonical_name, fold the old
        name into aliases, audit the change, and rebuild person projections."""
        entity = self.get_entity(entity_id)
        if not entity or entity.get("entity_type") != "person":
            return None
        new_name = str(new_name or "").strip()
        if not new_name:
            raise ValueError("person name is required")
        old_name = entity["canonical_name"]
        timestamp = now_iso()
        if old_name != new_name:
            if old_name:
                aliases = self.person_aliases(entity_id)
                if old_name not in aliases:
                    aliases.append(old_name)
                self.set_entity_property(entity_id, "aliases", aliases)
            self.connection.execute(
                "UPDATE entities SET canonical_name = ?, updated_at = ? WHERE id = ?",
                (new_name, timestamp, entity_id),
            )
            self._record_entity_revision(entity_id, "canonical_name", old_name, new_name, "user_rename")
            self.connection.commit()
        memory = self.rebuild_person_memory(entity_id)
        self.connection.commit()
        detail = {**self.get_entity_detail(entity_id), "semantic_profile": memory["profile"] if memory else None, "semantic_claims": memory["claims"] if memory else []}
        if memory:
            detail["event_memory"] = memory["event_memory"]
            detail["patterns"] = memory["patterns"]
        return detail

    def get_face_instance(self, instance_id):
        result = self._decode(
            self._row(
                """SELECT fi.*, a.path AS asset_path, a.file_name, a.mime_type, a.media_type
                FROM face_instances fi JOIN assets a ON a.id = fi.asset_id WHERE fi.id = ?""",
                (instance_id,),
            ),
            ["bbox_json", "embedding_json", "pose_json"],
        )
        if result:
            result["pose"] = result.get("pose_json", [])
            asset = self.get_asset(result["asset_id"])
            result["scope_id"] = (asset or {}).get("scope_id") or "home-default"
        return result

    def record_person_appearance_evidence(self, person_id, face_instance_id, crop_bbox, clothing, confidence, model_name):
        """Persist target-person appearance evidence, never scene-level labels."""
        person = self.get_entity(person_id)
        instance = self.get_face_instance(face_instance_id)
        if not person or person.get("status") != "confirmed":
            raise ValueError("appearance evidence requires a confirmed person")
        if not instance:
            raise ValueError("face instance not found")
        linked = self._row(
            "SELECT 1 FROM entity_mentions WHERE entity_id = ? AND face_instance_id = ?",
            (person_id, face_instance_id),
        )
        if not linked:
            raise ValueError("face instance is not confirmed for this person")
        values = []
        for item in clothing or []:
            value = str(item or "").strip()
            if value:
                values.append(value)
        timestamp = now_iso()
        existing = self._row(
            "SELECT * FROM person_appearance_evidence WHERE person_id = ? AND face_instance_id = ? AND model_name = ?",
            (person_id, face_instance_id, str(model_name or "unknown")),
        )
        if existing:
            self.connection.execute(
                """UPDATE person_appearance_evidence SET crop_bbox_json = ?, clothing_json = ?, confidence = ?,
                status = 'confirmed', updated_at = ? WHERE id = ?""",
                (json_value(crop_bbox, []), json_value(list(dict.fromkeys(values)), []), float(confidence or 0), timestamp, existing["id"]),
            )
            evidence_id = existing["id"]
        else:
            evidence_id = make_id("appearance")
            self.connection.execute(
                """INSERT INTO person_appearance_evidence(
                    id, person_id, face_instance_id, observation_id, asset_id, crop_bbox_json,
                    clothing_json, confidence, model_name, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evidence_id, person_id, face_instance_id, instance["observation_id"], instance["asset_id"],
                    json_value(crop_bbox, []), json_value(list(dict.fromkeys(values)), []), float(confidence or 0),
                    str(model_name or "unknown"), timestamp, timestamp,
                ),
            )
        self.connection.commit()
        return self.get_person_appearance_evidence(evidence_id)

    def get_person_appearance_evidence(self, evidence_id):
        return self._decode(
            self._row("SELECT * FROM person_appearance_evidence WHERE id = ?", (evidence_id,)),
            ["crop_bbox_json", "clothing_json"],
        )

    def list_person_appearance_evidence(self, person_id, include_empty=False, limit=500):
        where = "pae.person_id = ? AND pae.status = 'confirmed'"
        if not include_empty:
            where += " AND pae.clothing_json != '[]'"
        rows = self._rows(
            f"""SELECT pae.*, a.file_name, a.media_type FROM person_appearance_evidence pae
            JOIN assets a ON a.id = pae.asset_id WHERE {where}
            ORDER BY pae.confidence DESC, pae.updated_at DESC LIMIT ?""",
            (person_id, limit),
        )
        return [self._decode(row, ["crop_bbox_json", "clothing_json"]) for row in rows]

    def list_face_prototypes(self, cluster_id=None):
        params = [cluster_id] if cluster_id else []
        where = "WHERE cluster_id = ?" if cluster_id else ""
        rows = self._rows(f"SELECT * FROM face_prototypes {where} ORDER BY quality DESC, pose_bucket ASC", params)
        return [self._decode(row, ["embedding_json"]) for row in rows]

    def entity_event_ids(self, entity_id):
        rows = self._rows(
            """SELECT DISTINCT event_id FROM (
                SELECT eo.event_id FROM entity_mentions em
                JOIN event_observations eo ON eo.observation_id = em.observation_id
                WHERE em.entity_id = ?
                UNION
                SELECT event_id FROM event_participants WHERE person_id = ?
                UNION
                SELECT eo.event_id FROM entity_observations eob
                JOIN event_observations eo ON eo.observation_id = eob.observation_id
                WHERE eob.entity_id = ?
            )""",
            (entity_id, entity_id, entity_id),
        )
        return [row["event_id"] for row in rows]

    def get_entity_detail(self, entity_id):
        entity = self.get_entity(entity_id)
        if not entity:
            return None
        avatar = self._row(
            """SELECT fi.id FROM face_instances fi JOIN face_clusters fc ON fc.id = fi.cluster_id
            WHERE fc.entity_id = ? AND fc.status != 'rejected' ORDER BY fi.detection_confidence DESC, fi.created_at ASC LIMIT 1""",
            (entity_id,),
        )
        entity["avatar_face_instance_id"] = avatar["id"] if avatar else None
        clusters = self._rows("SELECT * FROM face_clusters WHERE entity_id = ? ORDER BY updated_at DESC", (entity_id,))
        for cluster in clusters:
            cluster["samples"] = self._rows(
                """SELECT fi.id, fi.asset_id, fi.observation_id, fi.bbox_json, fi.detection_confidence, a.file_name, a.media_type
                FROM face_instances fi JOIN assets a ON a.id = fi.asset_id WHERE fi.cluster_id = ? ORDER BY fi.created_at DESC LIMIT 12""",
                (cluster["id"],),
            )
        relationships = self.list_relationships(entity_id)
        facts = [fact for fact in self.list_facts(1000) if fact["subject"] == entity["canonical_name"] or fact["object"] == entity["canonical_name"]]
        event_ids = self.entity_event_ids(entity_id)
        observations = self._rows(
            """SELECT o.* FROM entity_observations eob JOIN observations o ON o.id = eob.observation_id
            WHERE eob.entity_id = ? ORDER BY o.captured_at DESC LIMIT 100""",
            (entity_id,),
        )
        evidence_observations = []
        for row in observations:
            observation = self.get_observation(row["id"])
            if observation:
                observation["asset"] = self.get_asset(observation["asset_id"])
                evidence_observations.append(observation)
        entity["evidence_count"] = len(evidence_observations)
        evidence_files = {}
        for observation in evidence_observations:
            asset = observation.get("asset") or {}
            evidence_files[observation["id"]] = {
                "evidence_id": observation["id"], "asset_id": observation["asset_id"],
                "file_name": asset.get("file_name") or observation["asset_id"], "kind": "observation",
            }
        for appearance in self.list_person_appearance_evidence(entity_id, include_empty=True):
            evidence_files[appearance["id"]] = {
                "evidence_id": appearance["id"], "asset_id": appearance["asset_id"],
                "file_name": appearance.get("file_name") or appearance["asset_id"], "kind": "appearance",
            }
        return {
            "entity": entity,
            "properties": self.list_entity_properties(entity_id),
            "property_history": self.list_entity_properties(entity_id, include_history=True),
            "clusters": clusters,
            "relationships": relationships,
            "facts": facts,
            "profile": self.get_semantic_profile(entity_id),
            "claims": self.list_semantic_claims(entity_id, 500),
            "appearance_evidence": self.list_person_appearance_evidence(entity_id, include_empty=True),
            "events": [self.get_event(event_id) for event_id in event_ids if self.get_event(event_id)],
            "observations": evidence_observations,
            "evidence_files": list(evidence_files.values()),
        }

    def create_face_cluster(self, embedding, confidence=0.0, scope_id="home-default"):
        cluster_id = make_id("cluster")
        timestamp = now_iso()
        values = self._normalise_vector(embedding)
        self.connection.execute(
            """INSERT INTO face_clusters(scope_id, id, representative_embedding_json, member_count, confidence, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?, ?)""",
            (scope_id or "home-default", cluster_id, json_value(values, []), float(confidence or 0), timestamp, timestamp),
        )
        self.connection.commit()
        entity = self.create_entity(f"待确认人物簇 · {cluster_id}", "person", "pending", None, float(confidence or 0), "由 buffalo_l embedding 生成，等待用户确认", scope_id=scope_id)
        self.connection.execute("UPDATE face_clusters SET entity_id = ? WHERE id = ?", (entity["id"], cluster_id))
        self.connection.commit()
        return self._row("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,))

    def _record_entity_revision(self, entity_id, field_name, old_value, new_value, source, evidence_ids=None):
        self.connection.execute(
            """INSERT INTO entity_revisions(id, entity_id, field_name, old_value, new_value, source, evidence_ids_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (make_id("entity_revision"), entity_id, field_name, old_value, new_value, source, json_value(evidence_ids, []), now_iso()),
        )

    @staticmethod
    def _property_row(row):
        if not row:
            return row
        value = dict(row)
        try:
            value["value"] = json.loads(value.pop("value_json") or "null")
        except (TypeError, json.JSONDecodeError):
            value["value"] = None
        try:
            value["evidence_ids"] = json.loads(value.pop("evidence_ids_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            value["evidence_ids"] = []
        return value

    def list_entity_properties(self, entity_id, include_history=False):
        status_clause = "" if include_history else " AND status IN ('active', 'pending')"
        rows = self._rows(
            "SELECT * FROM entity_properties WHERE entity_id = ?" + status_clause + " ORDER BY property_key, revision DESC, updated_at DESC",
            (entity_id,),
        )
        return [self._property_row(row) for row in rows]

    def public_entity(self, entity):
        """Return a standard-list-safe entity projection without changing its stable identity."""
        if not entity or entity.get("entity_type") != "place":
            return entity
        value = dict(entity)
        properties = {item["property_key"]: item for item in self.list_entity_properties(entity["id"])}
        if properties.get("private_flag", {}).get("value") is True:
            value["canonical_name"] = str(properties.get("alias", {}).get("value") or "私密地点")
            value["private"] = True
        else:
            value["private"] = False
        return value

    def _current_entity_property(self, entity_id, property_key):
        return self._row(
            """SELECT * FROM entity_properties WHERE entity_id = ? AND property_key = ?
            AND status IN ('active', 'pending') ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END,
            revision DESC, updated_at DESC LIMIT 1""",
            (entity_id, property_key),
        )

    def maintain_entity_property(self, entity_id, property_key, value, confidence=0.0, evidence_ids=None, source="derived"):
        """Record a derived value without replacing a user-maintained current value."""
        if not self.get_entity(entity_id):
            raise KeyError(entity_id)
        property_key = str(property_key or "").strip()
        if not property_key:
            raise ValueError("property_key is required")
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        current = self._current_entity_property(entity_id, property_key)
        encoded = json_value(value, None)
        if current and current["source"] == "user":
            return self._property_row(current)
        if current and current["value_json"] == encoded:
            self.connection.execute(
                "UPDATE entity_properties SET confidence = MAX(confidence, ?), evidence_ids_json = ?, updated_at = ? WHERE id = ?",
                (float(confidence or 0), json_value(list(dict.fromkeys(json.loads(current["evidence_ids_json"] or "[]") + evidence_ids)), []), now_iso(), current["id"]),
            )
            self.connection.commit()
            return self._property_row(self._row("SELECT * FROM entity_properties WHERE id = ?", (current["id"],)))
        timestamp = now_iso()
        status = "active" if not current else "pending"
        property_id = make_id("entity_property")
        self.connection.execute(
            """INSERT INTO entity_properties(id, entity_id, property_key, value_json, source, confidence, status,
            evidence_ids_json, supersedes_property_id, revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (property_id, entity_id, property_key, encoded, source, float(confidence or 0), status,
             json_value(evidence_ids, []), current["id"] if current else None,
             int(current["revision"] or 0) + 1 if current else 1, timestamp, timestamp),
        )
        self.connection.commit()
        return self._property_row(self._row("SELECT * FROM entity_properties WHERE id = ?", (property_id,)))

    def maintain_entity_property_values(self, entity_id, property_key, values, confidence=0.0, evidence_ids=None, source="derived"):
        """Merge deterministic multi-value observations without overriding user values."""
        current = self._current_entity_property(entity_id, property_key)
        if current and current["source"] == "user":
            return self._property_row(current)
        existing = self._property_row(current).get("value", []) if current else []
        existing = existing if isinstance(existing, list) else [existing]
        merged = list(dict.fromkeys([item for item in existing + list(values or []) if item]))
        if not current:
            return self.maintain_entity_property(entity_id, property_key, merged, confidence, evidence_ids, source)
        old_evidence = json.loads(current["evidence_ids_json"] or "[]")
        self.connection.execute(
            "UPDATE entity_properties SET value_json = ?, confidence = MAX(confidence, ?), evidence_ids_json = ?, updated_at = ? WHERE id = ?",
            (json_value(merged, []), float(confidence or 0), json_value(list(dict.fromkeys(old_evidence + list(evidence_ids or []))), []), now_iso(), current["id"]),
        )
        self.connection.commit()
        return self._property_row(self._row("SELECT * FROM entity_properties WHERE id = ?", (current["id"],)))

    def set_entity_property(self, entity_id, property_key, value, evidence_ids=None):
        """Make a user value current and preserve all older revisions for audit."""
        if not self.get_entity(entity_id):
            raise KeyError(entity_id)
        property_key = str(property_key or "").strip()
        if not property_key:
            raise ValueError("property_key is required")
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        current = self._current_entity_property(entity_id, property_key)
        timestamp = now_iso()
        if current:
            self.connection.execute(
                "UPDATE entity_properties SET status = 'superseded', updated_at = ? WHERE entity_id = ? AND property_key = ? AND status IN ('active', 'pending')",
                (timestamp, entity_id, property_key),
            )
        property_id = make_id("entity_property")
        self.connection.execute(
            """INSERT INTO entity_properties(id, entity_id, property_key, value_json, source, confidence, status,
            evidence_ids_json, supersedes_property_id, revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'user', 1, 'active', ?, ?, ?, ?, ?)""",
            (property_id, entity_id, property_key, json_value(value, None), json_value(evidence_ids, []),
             current["id"] if current else None, int(current["revision"] or 0) + 1 if current else 1, timestamp, timestamp),
        )
        self._record_entity_revision(
            entity_id, f"property:{property_key}", current["value_json"] if current else None,
            json_value(value, None), "user", evidence_ids,
        )
        if property_key in ("canonical_name", "family_role"):
            column = "canonical_name" if property_key == "canonical_name" else "family_role"
            self.connection.execute(
                f"UPDATE entities SET {column} = ?, updated_at = ? WHERE id = ?",
                (str(value) if value is not None else None, timestamp, entity_id),
            )
            if property_key == "canonical_name" and value:
                new_name = str(value)
                for cluster in self._rows("SELECT id FROM face_clusters WHERE entity_id = ?", (entity_id,)):
                    short_id = (cluster.get("id") or "").replace("cluster_", "")[:8]
                    if not short_id:
                        continue
                    old_placeholder = f"待命名成员#{short_id}"
                    if old_placeholder == new_name:
                        continue
                    self.connection.execute(
                        "UPDATE observations SET caption = REPLACE(caption, ?, ?) WHERE caption LIKE ?",
                        (old_placeholder, new_name, f"%{old_placeholder}%"),
                    )
                    self.connection.execute(
                        "UPDATE events SET summary = REPLACE(summary, ?, ?) WHERE summary LIKE ?",
                        (old_placeholder, new_name, f"%{old_placeholder}%"),
                    )
        self.connection.commit()
        return self._property_row(self._row("SELECT * FROM entity_properties WHERE id = ?", (property_id,)))

    def merge_face_clusters(self, target_cluster_id, source_cluster_id, source="user_merge"):
        if target_cluster_id == source_cluster_id:
            return self._row("SELECT * FROM face_clusters WHERE id = ?", (target_cluster_id,))
        target = self._row("SELECT * FROM face_clusters WHERE id = ?", (target_cluster_id,))
        other = self._row("SELECT * FROM face_clusters WHERE id = ?", (source_cluster_id,))
        if not target or not other or target["status"] == "rejected" or other["status"] == "rejected":
            return None
        if (target.get("scope_id") or "home-default") != (other.get("scope_id") or "home-default"):
            raise ValueError("face clusters must belong to the same memory space")
        target_entity = self.get_entity(target["entity_id"]) if target.get("entity_id") else None
        source_entity = self.get_entity(other["entity_id"]) if other.get("entity_id") else None
        if target_entity and source_entity and target_entity["id"] != source_entity["id"] and target_entity["status"] == "confirmed" and source_entity["status"] == "confirmed" and source != "user_merge":
            raise ValueError("two confirmed people cannot be merged automatically")
        self.connection.execute("UPDATE face_instances SET cluster_id = ? WHERE cluster_id = ?", (target_cluster_id, source_cluster_id))
        self.connection.execute("UPDATE face_clusters SET status = 'rejected', member_count = 0, updated_at = ?, revision = revision + 1 WHERE id = ?", (now_iso(), source_cluster_id))
        self._refresh_face_prototypes(target_cluster_id)
        if target.get("entity_id"):
            self._record_entity_revision(target["entity_id"], "face_cluster_merge", source_cluster_id, target_cluster_id, source, [])
        self.connection.commit()
        return self._row("SELECT * FROM face_clusters WHERE id = ?", (target_cluster_id,))

    def split_face_instance(self, cluster_id, face_instance_id, source="user_split"):
        cluster = self._row("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,))
        instance = self._row("SELECT * FROM face_instances WHERE id = ? AND cluster_id = ?", (face_instance_id, cluster_id))
        if not cluster or not instance:
            return None
        new_cluster = self.create_face_cluster(
            json.loads(instance["embedding_json"] or "[]"),
            float(instance.get("quality", 0) or 0),
            scope_id=cluster.get("scope_id") or "home-default",
        )
        self.connection.execute("UPDATE face_instances SET cluster_id = ? WHERE id = ?", (new_cluster["id"], face_instance_id))
        self._refresh_face_prototypes(cluster_id)
        self._refresh_face_prototypes(new_cluster["id"])
        if cluster.get("entity_id"):
            self._record_entity_revision(cluster["entity_id"], "face_cluster_split", cluster_id, new_cluster["id"], source, [face_instance_id])
        self.connection.commit()
        return self._row("SELECT * FROM face_clusters WHERE id = ?", (new_cluster["id"],))

    def add_face_instance(self, asset_id, observation_id, face, threshold=0.30, model_name="buffalo_l"):
        asset = self.get_asset(asset_id) or {}
        scope_id = asset.get("scope_id") or "home-default"
        embedding = self._normalise_vector(face.get("embedding"))
        if not embedding or face.get("identity_ready") is False:
            return None
        pose = [float(value) for value in (face.get("pose") or [])]
        pose_bucket_value = face.get("pose_bucket")
        quality = face.get("quality")
        if quality is None:
            from .face_embeddings import compute_face_quality, pose_bucket

            pose_bucket_value = pose_bucket_value or pose_bucket(pose)
            quality = compute_face_quality(
                face.get("confidence", 0), face.get("area_ratio", 0), face.get("sharpness", 0), pose
            )
        else:
            quality = float(quality)
            if not pose_bucket_value:
                from .face_embeddings import pose_bucket

                pose_bucket_value = pose_bucket(pose)
        embedding_model = str(face.get("embedding_model") or model_name or "unknown")
        embedding_version = str(face.get("embedding_version") or "legacy")
        identity_eligible = face.get("identity_eligible", True) is not False
        if not identity_eligible:
            instance_id = make_id("face")
            self.connection.execute(
                """INSERT INTO face_instances(
                    id, asset_id, observation_id, cluster_id, bbox_json, embedding_json,
                    detection_confidence, quality, pose_json, area_ratio, sharpness,
                    pose_bucket, embedding_model, embedding_version, embedding_quality_signal, created_at
                ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    instance_id, asset_id, observation_id, json_value(face.get("bbox"), []), json_value(embedding, []),
                    float(face.get("confidence", 0) or 0), float(quality or 0), json_value(pose, []), float(face.get("area_ratio", 0) or 0),
                    float(face.get("sharpness", 0) or 0), str(pose_bucket_value or "unknown"), embedding_model, embedding_version,
                    float(face.get("quality_signal", 0) or 0), now_iso(),
                ),
            )
            self.connection.commit()
            self.upsert_vector(
                "visual", "face_instance", instance_id, embedding, embedding_model,
                {"asset_id": asset_id, "observation_id": observation_id, "model_version": embedding_version,
                 "quality": float(quality or 0), "pose_bucket": pose_bucket_value, "identity_eligible": False},
            )
            return {"id": instance_id, "cluster_id": None, "score": 0.0, "embedding": embedding}
        clusters = self._rows("SELECT * FROM face_clusters WHERE scope_id = ? AND status IN ('pending', 'confirmed')", (scope_id,))
        best = None
        best_score = 0.0
        if quality >= 0.30:
            for cluster in clusters:
                prototypes = self.list_face_prototypes(cluster["id"])
                cluster_signatures = {
                    (item.get("model_name") or "unknown", item.get("model_version") or "unknown")
                    for item in prototypes
                }
                embedding_signature = (embedding_model, embedding_version)
                if cluster_signatures and embedding_signature not in cluster_signatures:
                    continue
                representatives = [
                    item["embedding_json"] if isinstance(item["embedding_json"], list)
                    else json.loads(item["embedding_json"] or "[]")
                    for item in prototypes
                ]
                if not representatives:
                    representatives = [json.loads(cluster["representative_embedding_json"] or "[]")]
                score = max((self._cosine(embedding, representative) for representative in representatives), default=0.0)
                if score > best_score:
                    best, best_score = cluster, score
        if not best or best_score < threshold:
            best = self.create_face_cluster(embedding, quality, scope_id=scope_id)
        instance_id = make_id("face")
        self.connection.execute(
            """INSERT INTO face_instances(
                id, asset_id, observation_id, cluster_id, bbox_json, embedding_json,
                detection_confidence, quality, pose_json, area_ratio, sharpness,
                pose_bucket, embedding_model, embedding_version, embedding_quality_signal, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                instance_id, asset_id, observation_id, best["id"], json_value(face.get("bbox"), []), json_value(embedding, []),
                float(face.get("confidence", 0) or 0), float(quality or 0), json_value(pose, []), float(face.get("area_ratio", 0) or 0),
                float(face.get("sharpness", 0) or 0), str(pose_bucket_value or "unknown"), embedding_model, embedding_version,
                float(face.get("quality_signal", 0) or 0), now_iso(),
            ),
        )
        self._refresh_face_prototypes(best["id"])
        self.connection.commit()
        self.upsert_vector("visual", "face_instance", instance_id, embedding, embedding_model, {"cluster_id": best["id"], "asset_id": asset_id, "observation_id": observation_id, "model_version": embedding_version, "quality": float(quality or 0), "pose_bucket": pose_bucket_value})
        entity = self.get_entity(best.get("entity_id")) if best.get("entity_id") else None
        if entity and entity.get("status") == "confirmed":
            self._link_confirmed_entity_mention(entity, observation_id, instance_id, face.get("confidence"))
        return {"id": instance_id, "cluster_id": best["id"], "score": best_score, "embedding": embedding}

    def _link_confirmed_entity_mention(self, entity, observation_id, face_instance_id, confidence=0.0):
        self.connection.execute(
            """INSERT OR IGNORE INTO entity_mentions(
                id, entity_id, observation_id, face_instance_id, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (make_id("mention"), entity["id"], observation_id, face_instance_id, float(confidence or 0), now_iso()),
        )
        self._add_entity_to_observation(observation_id, entity)
        self.connection.commit()

    def _refresh_face_prototypes(self, cluster_id):
        rows = self._rows(
            "SELECT * FROM face_instances WHERE cluster_id = ? ORDER BY quality DESC, created_at ASC",
            (cluster_id,),
        )
        best_by_view = {}
        for row in rows:
            view = row.get("pose_bucket") or "unknown"
            if view not in best_by_view:
                best_by_view[view] = row
        timestamp = now_iso()
        self.connection.execute("DELETE FROM face_prototypes WHERE cluster_id = ?", (cluster_id,))
        for row in best_by_view.values():
            self.connection.execute(
                """INSERT INTO face_prototypes(
                    id, cluster_id, face_instance_id, pose_bucket, embedding_json,
                    quality, model_name, model_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    make_id("prototype"), cluster_id, row["id"], row.get("pose_bucket") or "unknown",
                    row["embedding_json"], float(row.get("quality", 0) or 0), row.get("embedding_model") or "unknown",
                    row.get("embedding_version") or "unknown", timestamp, timestamp,
                ),
            )
        representative = max(best_by_view.values(), key=lambda item: float(item.get("quality", 0) or 0), default=None)
        self.connection.execute(
            """UPDATE face_clusters SET representative_embedding_json = ?, member_count = ?,
            confidence = MAX(confidence, ?), revision = revision + 1, updated_at = ? WHERE id = ?""",
            (
                representative["embedding_json"] if representative else "[]", len(rows),
                float(representative.get("quality", 0) or 0) if representative else 0, timestamp, cluster_id,
            ),
        )

    def recluster_faces(self, threshold=0.30, minimum_quality=0.55, scope_id=None):
        """Globally regroup faces with quality-aware multi-view prototypes."""
        from .face_clustering import FaceClusterer, FaceSample

        instances = self._rows(
            """SELECT fi.id, fi.cluster_id, fi.embedding_json, fi.quality, fi.detection_confidence,
            fi.pose_bucket, fi.embedding_model, fi.embedding_version
            FROM face_instances fi JOIN assets a ON a.id = fi.asset_id
            WHERE fi.embedding_json != '[]' AND fi.cluster_id IS NOT NULL AND fi.quality >= ?"""
            + (" AND a.scope_id = ?" if scope_id else ""),
            (minimum_quality, scope_id) if scope_id else (minimum_quality,),
        )
        if not instances:
            return {"instances": 0, "clusters": 0, "threshold": threshold, "minimum_quality": minimum_quality}
        samples = []
        for row in instances:
            embedding = json.loads(row["embedding_json"] or "[]")
            quality = float(row.get("quality", 0) or 0) or float(row.get("detection_confidence", 0) or 0)
            samples.append(
                FaceSample(
                    row["id"], embedding, quality=quality,
                    pose_bucket=row.get("pose_bucket") or "unknown",
                    model_name=row.get("embedding_model") or "unknown",
                    model_version=row.get("embedding_version") or "unknown",
                    protected_cluster=(
                        row["cluster_id"]
                        if (self._row("SELECT status FROM face_clusters WHERE id = ?", (row["cluster_id"],)) or {}).get("status") == "confirmed"
                        else None
                    ),
                )
            )
        result = FaceClusterer(match_threshold=threshold, minimum_quality=minimum_quality).fit(samples)
        timestamp = now_iso()
        active_cluster_ids = set()
        for cluster in result.clusters:
            member_ids = [item.id for item in cluster.members]
            old_cluster_ids = [row["cluster_id"] for row in instances if row["id"] in member_ids]
            confirmed_cluster_ids = [
                old_id for old_id in dict.fromkeys(old_cluster_ids)
                if (self._row("SELECT status FROM face_clusters WHERE id = ?", (old_id,)) or {}).get("status") == "confirmed"
            ]
            keep_id = confirmed_cluster_ids[0] if confirmed_cluster_ids else old_cluster_ids[0]
            active_cluster_ids.add(keep_id)
            confirmed = bool(confirmed_cluster_ids)
            self.connection.executemany(
                "UPDATE face_instances SET cluster_id = ? WHERE id = ?",
                [(keep_id, instance_id) for instance_id in member_ids],
            )
            self.connection.execute(
                """UPDATE face_clusters SET member_count = ?, revision = revision + 1,
                updated_at = ?, status = ? WHERE id = ?""",
                (len(member_ids), timestamp, "confirmed" if confirmed else "pending", keep_id),
            )
            self._refresh_face_prototypes(keep_id)
            for old_cluster_id in set(old_cluster_ids):
                if old_cluster_id == keep_id:
                    continue
                self.connection.execute(
                    """UPDATE face_clusters SET status = 'rejected', member_count = 0,
                    updated_at = ?, revision = revision + 1 WHERE id = ?""",
                    (timestamp, old_cluster_id),
                )
            for instance_id in member_ids:
                vector = self._row(
                    "SELECT metadata_json FROM memory_vectors WHERE source_type = 'face_instance' AND source_id = ? ORDER BY updated_at DESC LIMIT 1",
                    (instance_id,),
                )
                metadata = json.loads(vector["metadata_json"] or "{}") if vector else {}
                metadata["cluster_id"] = keep_id
                self.connection.execute(
                    "UPDATE memory_vectors SET metadata_json = ?, updated_at = ? WHERE source_type = 'face_instance' AND source_id = ?",
                    (json_value(metadata, {}), timestamp, instance_id),
                )
        stale_ids = {
            row["cluster_id"] for row in instances if row["cluster_id"] not in active_cluster_ids
        }
        for cluster_id in stale_ids:
            self.connection.execute(
                "UPDATE face_clusters SET status = 'rejected', member_count = 0, updated_at = ?, revision = revision + 1 WHERE id = ?",
                (timestamp, cluster_id),
            )
        self._sync_face_entity_statuses(timestamp)
        self.connection.commit()
        return {
            "instances": len(instances), "clusters": len(result.clusters),
            "threshold": threshold, "minimum_quality": minimum_quality,
            "noise": sum(1 for cluster in result.clusters if cluster.noise),
        }

    def _sync_face_entity_statuses(self, timestamp=None):
        """Hide pending entities whose candidate face clusters were retired."""
        timestamp = timestamp or now_iso()
        entities = self._rows("SELECT id, status FROM entities WHERE entity_type = 'person'")
        for entity in entities:
            active_count = self.connection.execute(
                "SELECT COUNT(*) FROM face_clusters WHERE entity_id = ? AND status IN ('pending', 'confirmed') AND member_count > 0",
                (entity["id"],),
            ).fetchone()[0]
            if active_count == 0 and entity["status"] != "confirmed":
                self.connection.execute(
                    "UPDATE entities SET status = 'rejected', summary = ?, updated_at = ? WHERE id = ?",
                    ("该人物候选已因全局重聚类失效，保留历史审计记录", timestamp, entity["id"]),
                )

    def list_face_clusters(self, status=None):
        params = [status] if status else []
        where = "WHERE fc.status = ?" if status else "WHERE fc.status != 'rejected'"
        rows = self._rows(f"""SELECT fc.*, e.canonical_name, e.family_role, e.status AS entity_status
            FROM face_clusters fc LEFT JOIN entities e ON e.id = fc.entity_id {where} ORDER BY fc.updated_at DESC""", params)
        for row in rows:
            row["reviewable"] = row.get("status") != "rejected" and int(row.get("member_count", 0) or 0) > 0
            row["single_sample"] = row.get("status") == "pending" and int(row.get("member_count", 0) or 0) <= 1
            row["samples"] = self._rows("""SELECT fi.id, fi.asset_id, fi.observation_id, fi.bbox_json, fi.detection_confidence, a.file_name, a.media_type
                FROM face_instances fi JOIN assets a ON a.id = fi.asset_id WHERE fi.cluster_id = ? ORDER BY fi.created_at DESC LIMIT 12""", (row["id"],))
        return rows

    def confirm_face_cluster(self, cluster_id, name, family_role=None):
        cluster = self._row("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,))
        if not cluster:
            return None
        scope_id = cluster.get("scope_id") or "home-default"
        existing = self.find_confirmed_person_by_name(scope_id, name)
        if existing:
            return self._merge_cluster_into_person(cluster_id, existing["entity_id"])
        entity = self.get_entity(cluster["entity_id"]) if cluster["entity_id"] else None
        if not entity:
            entity = self.create_entity(name, "person", "confirmed", family_role, 1.0, "用户确认的人物实体", scope_id=scope_id)
        else:
            self.connection.execute("UPDATE entities SET canonical_name = ?, status = 'confirmed', family_role = ?, confidence = MAX(confidence, 1), updated_at = ? WHERE id = ?", (name, family_role, now_iso(), entity["id"]))
            entity = self.get_entity(entity["id"])
        self.connection.execute("UPDATE face_clusters SET status = 'confirmed', entity_id = ?, updated_at = ?, revision = revision + 1 WHERE id = ?", (entity["id"], now_iso(), cluster_id))
        instances = self._rows("SELECT * FROM face_instances WHERE cluster_id = ?", (cluster_id,))
        observation_ids = []
        for instance in instances:
            observation_ids.append(instance["observation_id"])
            self._link_confirmed_entity_mention(entity, instance["observation_id"], instance["id"], instance["detection_confidence"])
        if family_role:
            self.maintain_fact(name, "家庭角色", family_role, list(dict.fromkeys(observation_ids)), confidence=1.0)
        self.connection.execute("UPDATE entities SET summary = ?, updated_at = ? WHERE id = ?", (f"已确认人物，出现在 {len(set(observation_ids))} 条观察中", now_iso(), entity["id"]))
        self.connection.commit()
        self._refresh_event_participants(observation_ids)
        self.resegment_events_for_confirmed_entity(entity["id"])
        memory = self.rebuild_person_memory(entity["id"])
        self.connection.commit()
        if memory:
            detail = {**self.get_entity_detail(entity["id"]), "semantic_profile": memory["profile"], "semantic_claims": memory["claims"]}
            detail["event_memory"] = memory["event_memory"]
            detail["patterns"] = memory["patterns"]
            detail["refresh_counts"] = {
                "observations": len(memory["observation_ids"]),
                "events": len(memory["event_ids"]),
                "patterns": len(memory["patterns"]),
                "claims": len(memory["claims"]),
                "appearance": len(self.list_person_appearance_evidence(entity["id"], include_empty=True)),
            }
            return detail
        return self.get_entity_detail(entity["id"])

    def auto_confirm_clusters(self, scope_id=None, min_members=2, min_confidence=0.5):
        """Auto-confirm stable pending clusters with a placeholder name.

        Selects pending clusters with member_count >= min_members and average
        detection_confidence >= min_confidence. Each is confirmed with a
        placeholder name "待命名成员#<short_id>" that the user can rename from
        the dashboard. Returns the list of confirmed cluster summaries.
        """
        params = [min_members]
        where = "WHERE status = 'pending' AND member_count >= ?"
        if scope_id:
            where += " AND scope_id = ?"
            params.append(scope_id)
        rows = self._rows(f"SELECT id, confidence, member_count FROM face_clusters {where} ORDER BY updated_at ASC", params)
        confirmed = []
        for row in rows:
            avg_confidence = float(row["confidence"] or 0)
            if avg_confidence < min_confidence:
                continue
            short_id = row["id"].replace("cluster_", "")[:8]
            placeholder = f"待命名成员#{short_id}"
            try:
                self.confirm_face_cluster(row["id"], placeholder)
                confirmed.append({"cluster_id": row["id"], "name": placeholder, "member_count": int(row["member_count"] or 0), "avg_confidence": round(avg_confidence, 3)})
            except Exception as error:
                confirmed.append({"cluster_id": row["id"], "error": str(error)})
        return confirmed

    def _merge_cluster_into_person(self, cluster_id, target_entity_id):
        """Merge a just-confirmed cluster into an existing confirmed person:
        move its face instances and mentions to the target, retire the cluster,
        rebuild the target's memory, and return a merged marker detail."""
        cluster = self._row("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,))
        target = self.get_entity(target_entity_id)
        if not cluster or not target:
            return None
        if (cluster.get("scope_id") or "home-default") != (target.get("scope_id") or "home-default"):
            raise ValueError("face cluster and person must belong to the same memory space")
        instances = self._rows("SELECT * FROM face_instances WHERE cluster_id = ?", (cluster_id,))
        observation_ids = []
        target_cluster = self._row(
            """SELECT id FROM face_clusters WHERE entity_id = ? AND status = 'confirmed'
            ORDER BY member_count DESC, updated_at DESC LIMIT 1""",
            (target_entity_id,),
        )
        for instance in instances:
            observation_ids.append(instance["observation_id"])
            if target_cluster:
                self.connection.execute(
                    "UPDATE face_instances SET cluster_id = ? WHERE id = ?",
                    (target_cluster["id"], instance["id"]),
                )
            self._link_confirmed_entity_mention(target, instance["observation_id"], instance["id"], instance["detection_confidence"])
        self.connection.execute(
            "UPDATE face_clusters SET status = 'rejected', entity_id = ?, member_count = 0, updated_at = ?, revision = revision + 1 WHERE id = ?",
            (target_entity_id, now_iso(), cluster_id),
        )
        if target_cluster:
            self._refresh_face_prototypes(target_cluster["id"])
        self._record_entity_revision(target_entity_id, "face_cluster_merge", cluster_id, target_cluster["id"] if target_cluster else None, "user_name_merge", [])
        self.connection.commit()
        self._refresh_event_participants(observation_ids)
        self.resegment_events_for_confirmed_entity(target_entity_id)
        memory = self.rebuild_person_memory(target_entity_id)
        self.connection.commit()
        detail = {**self.get_entity_detail(target_entity_id), "semantic_profile": memory["profile"] if memory else None, "semantic_claims": memory["claims"] if memory else []}
        if memory:
            detail["event_memory"] = memory["event_memory"]
            detail["patterns"] = memory["patterns"]
            detail["refresh_counts"] = {
                "observations": len(memory["observation_ids"]),
                "events": len(memory["event_ids"]),
                "patterns": len(memory["patterns"]),
                "claims": len(memory["claims"]),
                "appearance": len(self.list_person_appearance_evidence(target_entity_id, include_empty=True)),
            }
        detail["merged_into"] = target["id"]
        detail["canonical_name"] = target["canonical_name"]
        return detail

    def confirm_person_entity(self, entity_id, name, family_role=None):
        """Resolve a native person entity to its active face cluster."""
        entity = self.get_entity(entity_id)
        if not entity or entity.get("entity_type") != "person":
            return None
        scope_id = entity.get("scope_id") or "home-default"
        existing = self.find_confirmed_person_by_name(scope_id, name)
        if existing and existing["entity_id"] != entity_id:
            cluster = self._row(
                """SELECT id FROM face_clusters WHERE entity_id = ? AND status IN ('pending', 'confirmed') AND member_count > 0
                ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1""",
                (entity_id,),
            )
            if cluster:
                return self._merge_cluster_into_person(cluster["id"], existing["entity_id"])
        cluster = self._row(
            """SELECT id FROM face_clusters
            WHERE entity_id = ? AND status IN ('pending', 'confirmed') AND member_count > 0
            ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1""",
            (entity_id,),
        )
        if cluster:
            return self.confirm_face_cluster(cluster["id"], name, family_role)
        timestamp = now_iso()
        self.connection.execute(
            """UPDATE entities SET canonical_name = ?, status = 'confirmed', family_role = ?,
            confidence = MAX(confidence, 1), updated_at = ? WHERE id = ?""",
            (name, family_role, timestamp, entity_id),
        )
        self.connection.commit()
        memory = self.rebuild_person_memory(entity_id)
        detail = {**self.get_entity_detail(entity_id), "semantic_profile": memory["profile"] if memory else None, "semantic_claims": memory["claims"] if memory else []}
        if memory:
            detail["event_memory"] = memory["event_memory"]
            detail["patterns"] = memory["patterns"]
            detail["refresh_counts"] = {
                "observations": len(memory["observation_ids"]),
                "events": len(memory["event_ids"]),
                "patterns": len(memory["patterns"]),
                "claims": len(memory["claims"]),
                "appearance": len(self.list_person_appearance_evidence(entity_id, include_empty=True)),
            }
        return detail

    def delete_person_candidate(self, entity_id):
        """Remove a non-person candidate permanently.

        Deletes the entity and all of its derived associations (face clusters,
        entity mentions, relationships, semantic claims/profiles, event memory,
        appearance evidence, properties and revisions). Face instances and the
        original assets are preserved as evidence. Confirmed people cannot be
        deleted; use rename/merge instead.
        """
        entity = self.get_entity(entity_id)
        if not entity or entity.get("entity_type") != "person":
            return None
        if entity.get("status") == "confirmed":
            raise ValueError("confirmed person entities cannot be deleted")
        cluster_ids = [row["id"] for row in self._rows("SELECT id FROM face_clusters WHERE entity_id = ?", (entity_id,))]
        observation_ids = list(dict.fromkeys(
            row["observation_id"] for row in self._rows(
                "SELECT DISTINCT observation_id FROM face_instances WHERE cluster_id IN (%s)" % ",".join("?" * len(cluster_ids)) if cluster_ids else "SELECT NULL",
                tuple(cluster_ids) if cluster_ids else (),
            ) if row["observation_id"]
        ))
        # Detach face instances (keep them as evidence, unbound from the deleted cluster).
        for cluster_id in cluster_ids:
            self.connection.execute("UPDATE face_instances SET cluster_id = NULL WHERE cluster_id = ?", (cluster_id,))
            self.connection.execute("DELETE FROM face_prototypes WHERE cluster_id = ?", (cluster_id,))
            self.connection.execute("DELETE FROM face_clusters WHERE id = ?", (cluster_id,))
        for table in (
            "entity_mentions", "entity_properties", "entity_revisions",
        ):
            self.connection.execute(f"DELETE FROM {table} WHERE entity_id = ?", (entity_id,))
        for table in (
            "semantic_claims", "semantic_profiles", "person_event_memory",
            "person_appearance_evidence",
        ):
            self.connection.execute(f"DELETE FROM {table} WHERE person_id = ?", (entity_id,))
        self.connection.execute(
            "DELETE FROM entity_merge_candidates WHERE entity_ids_json LIKE ? OR target_entity_id = ?",
            (f"%{entity_id}%", entity_id),
        )
        self.connection.execute(
            "DELETE FROM relationships WHERE subject_entity_id = ? OR object_entity_id = ?",
            (entity_id, entity_id),
        )
        self.connection.execute(
            "DELETE FROM facts WHERE subject = ? OR object = ?",
            (entity["canonical_name"], entity["canonical_name"]),
        )
        self.connection.execute("DELETE FROM persons WHERE id = ?", (entity_id,))
        self.connection.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        self.connection.commit()
        self._refresh_event_participants(observation_ids)
        return {"deleted": True, "entity_id": entity_id}

    def reject_person_entity(self, entity_id):
        """Delete a candidate that is not a person. Confirmed people are refused."""
        return self.delete_person_candidate(entity_id)

    def get_person_evidence(self, person_id):
        detail = self.get_entity_detail(person_id)
        if not detail:
            return None
        scope_id = detail["entity"].get("scope_id") or "home-default"
        event_ids = self.entity_event_ids(person_id)
        detail["events"] = [self.get_event(event_id) for event_id in event_ids if self.get_event(event_id)]
        detail["event_memory"] = self.list_person_event_memory(person_id)
        samples = []
        assets = {}
        for cluster in detail.get("clusters", []):
            if (cluster.get("scope_id") or "home-default") != scope_id:
                continue
            for sample in cluster.get("samples", []):
                asset = self.get_asset(sample["asset_id"])
                if asset:
                    assets[asset["id"]] = asset
                samples.append({
                    **sample,
                    "cluster_id": cluster["id"],
                    "scope_id": scope_id,
                    "asset": asset,
                    "crop_url": f"/api/face-instances/{sample['id']}/crop",
                    "asset_url": f"/api/assets/{sample['asset_id']}/file",
                })
        detail["scope_id"] = scope_id
        detail["face_samples"] = samples
        detail["assets"] = list(assets.values())
        detail["event_details"] = [
            self.get_event_detail(event_id) for event_id in event_ids if self.get_event_detail(event_id)
        ]
        return detail

    def _add_entity_to_observation(self, observation_id, entity):
        observation = self.get_observation(observation_id)
        if not observation:
            return
        people = observation.get("people") or []
        values = []
        found = False
        for person in people:
            if isinstance(person, dict) and person.get("entity_id") == entity["id"]:
                found = True
            values.append(person)
        if not found:
            values.append({"entity_id": entity["id"], "name": entity["canonical_name"], "status": entity["status"]})
        self.connection.execute("UPDATE observations SET people_json = ? WHERE id = ?", (json_value(values, []), observation_id))

    def resegment_events_for_confirmed_entity(self, entity_id):
        """Merge only events bridged by a newly confirmed person and compatible scores."""
        event_ids = self.entity_event_ids(entity_id)
        if len(event_ids) < 2:
            return []
        merged = []
        for event_id in sorted(event_ids):
            event = self.get_event(event_id)
            if not event or event_id not in {item["id"] for item in self.list_events(1000)}:
                continue
            for other_id in sorted(event_ids):
                if other_id == event_id:
                    continue
                other = self.get_event(other_id)
                if not other:
                    continue
                observation_id = next((item for item in other["observation_ids"] if entity_id in self._confirmed_entity_ids_for_observation(item)), None)
                if not observation_id:
                    continue
                event_time = parse_time(event.get("time_start"))
                other_time = parse_time(other.get("time_start"))
                if event_time and other_time and abs((event_time - other_time).total_seconds()) > 6 * 3600:
                    continue
                score = self._event_candidate_score(self.get_observation(observation_id), event, [
                    self._event_anchor(self.get_observation(item)) for item in event["observation_ids"]
                ])
                if score["total"] >= 0.50:
                    self._merge_events(event_id, other_id)
                    merged.append((event_id, other_id))
                    break
        return merged

    def _merge_events(self, target_event_id, source_event_id, reason=None):
        if target_event_id == source_event_id:
            return self.get_event(target_event_id)
        target = self.get_event(target_event_id)
        source = self.get_event(source_event_id)
        if not target or not source:
            return target
        affected_people = self._confirmed_entity_ids_for_event(target_event_id) | self._confirmed_entity_ids_for_event(source_event_id)
        target_observation_ids = list(target["observation_ids"])
        source_observation_ids = list(source["observation_ids"])
        target_people = target.get("participants") or []
        source_people = source.get("participants") or []
        merged_people = dedupe_json_values(target_people + source_people)
        start_values = [value for value in (target.get("time_start"), source.get("time_start")) if value]
        end_values = [value for value in (target.get("time_end"), source.get("time_end")) if value]
        self.connection.execute(
            "INSERT OR IGNORE INTO event_observations(event_id, observation_id) SELECT ?, observation_id FROM event_observations WHERE event_id = ?",
            (target_event_id, source_event_id),
        )
        for participant in self.list_event_participants(source_event_id):
            self.upsert_event_participant(target_event_id, participant["person_id"], participant["role"], participant["evidence_ids_json"], participant["confidence"])
        self.connection.execute(
            """UPDATE events SET time_start = ?, time_end = ?, place = COALESCE(place, ?),
            activity = COALESCE(activity, ?), participants_json = ?, confidence = MAX(confidence, ?),
            revision = revision + 1, updated_at = ? WHERE id = ?""",
            (
                min(start_values) if start_values else None, max(end_values) if end_values else None,
                source.get("place"), source.get("activity"), json_value(merged_people, []),
                float(source.get("confidence", 0) or 0), now_iso(), target_event_id,
            ),
        )
        for observation_id in source_observation_ids:
            observation = self.get_observation(observation_id) or {}
            asset = self.get_asset(observation.get("asset_id"))
            if asset:
                metadata = asset.get("metadata_json") or {}
                metadata["event_id"] = target_event_id
                self.connection.execute("UPDATE assets SET metadata_json = ?, updated_at = ? WHERE id = ?", (json_value(metadata, {}), now_iso(), asset["id"]))
            vector = self._row("SELECT metadata_json FROM memory_vectors WHERE source_type = 'observation' AND source_id = ? ORDER BY updated_at DESC LIMIT 1", (observation_id,))
            vector_metadata = json.loads(vector["metadata_json"] or "{}") if vector else {}
            vector_metadata["event_id"] = target_event_id
            self.connection.execute("UPDATE memory_vectors SET metadata_json = ?, updated_at = ? WHERE source_type = 'observation' AND source_id = ?", (json_value(vector_metadata, {}), now_iso(), observation_id))
        # Re-home every event-backed projection before deleting the source.
        # SQLite correctly rejects deleting an event that still has audit,
        # entity, participant-memory, or feedback rows pointing at it.
        for entity_row in self._rows("SELECT entity_id, relation, evidence_ids_json, confidence FROM event_entities WHERE event_id = ?", (source_event_id,)):
            self.upsert_event_entity(
                target_event_id,
                entity_row["entity_id"],
                entity_row["relation"],
                json.loads(entity_row["evidence_ids_json"] or "[]"),
                entity_row["confidence"],
            )
        self.connection.execute("DELETE FROM event_observations WHERE event_id = ?", (source_event_id,))
        self.connection.execute("DELETE FROM event_participants WHERE event_id = ?", (source_event_id,))
        self.connection.execute("DELETE FROM event_entities WHERE event_id = ?", (source_event_id,))
        self.connection.execute("UPDATE event_revisions SET event_id = ? WHERE event_id = ?", (target_event_id, source_event_id))
        self.connection.execute("DELETE FROM person_event_memory WHERE event_id = ?", (source_event_id,))
        self.connection.execute("UPDATE memory_feedback SET target_event_id = ? WHERE target_event_id = ?", (target_event_id, source_event_id))
        self.connection.execute("DELETE FROM memory_vectors WHERE source_type = 'event' AND source_id = ?", (source_event_id,))
        self.connection.execute("DELETE FROM events WHERE id = ?", (source_event_id,))
        self.connection.commit()
        self._refresh_event_participants(target_observation_ids + source_observation_ids)
        for person_id in affected_people:
            self.rebuild_person_memory(person_id)
        return self.get_event(target_event_id)

    def _refresh_event_participants(self, observation_ids):
        for observation_id in set(observation_ids):
            rows = self._rows("SELECT event_id FROM event_observations WHERE observation_id = ?", (observation_id,))
            for row in rows:
                event = self.get_event(row["event_id"])
                participants = event.get("participants") or []
                observation = self.get_observation(observation_id) or {}
                merged = participants[:]
                for key in self._confirmed_entity_ids_for_observation(observation_id):
                    entity = self.get_entity(key)
                    person = {"entity_id": key, "name": entity["canonical_name"], "status": "confirmed"}
                    if not any((item.get("entity_id") if isinstance(item, dict) else item) == key for item in merged):
                        merged.append(person)
                    self.upsert_event_participant(row["event_id"], key, "visible_subject", [observation_id], 0.75)
                asset = self.get_asset(observation.get("asset_id")) or {}
                source_owner_id = asset.get("source_owner_id")
                source_owner = self.get_entity(source_owner_id) if source_owner_id else None
                if source_owner and source_owner.get("status") == "confirmed":
                    self.upsert_event_participant(row["event_id"], source_owner_id, "captured_by", [observation_id], float(asset.get("source_confidence", 0.5) or 0.5))
                self.connection.execute("UPDATE events SET participants_json = ?, revision = revision + 1, updated_at = ? WHERE id = ?", (json_value(merged, []), now_iso(), event["id"]))
                self.refresh_event_summary(row["event_id"])

    def reject_face_cluster(self, cluster_id):
        """Delete a face cluster that is not a person. Confirmed clusters refuse."""
        cluster = self._row("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,))
        if not cluster:
            return None
        if cluster.get("entity_id"):
            entity = self.get_entity(cluster["entity_id"])
            if entity and entity.get("status") == "confirmed":
                raise ValueError("confirmed person clusters cannot be rejected")
        return self.delete_person_candidate(cluster.get("entity_id")) if cluster.get("entity_id") else None

    def list_relationships(self, entity_id=None, scope_id=None):
        scope_clause = " AND r.scope_id = ?" if scope_id else ""
        if entity_id:
            rows = self._rows("""SELECT r.*, s.canonical_name AS subject_name, o.canonical_name AS object_name
                FROM relationships r JOIN entities s ON s.id = r.subject_entity_id JOIN entities o ON o.id = r.object_entity_id
                WHERE (r.subject_entity_id = ? OR r.object_entity_id = ?)""" + scope_clause + " ORDER BY r.updated_at DESC", (entity_id, entity_id, scope_id) if scope_id else (entity_id, entity_id))
        else:
            rows = self._rows("""SELECT r.*, s.canonical_name AS subject_name, o.canonical_name AS object_name
                FROM relationships r JOIN entities s ON s.id = r.subject_entity_id JOIN entities o ON o.id = r.object_entity_id""" + (" WHERE r.scope_id = ?" if scope_id else "") + " ORDER BY r.updated_at DESC", (scope_id,) if scope_id else ())
        return [self._decode(row, ["evidence_ids_json"]) for row in rows]

    def list_person_relationships(self, scope_id=None):
        """Family graph edges: only relationships where both ends are person entities."""
        scope_clause = " AND r.scope_id = ?" if scope_id else ""
        rows = self._rows("""SELECT r.*, s.canonical_name AS subject_name, o.canonical_name AS object_name
            FROM relationships r JOIN entities s ON s.id = r.subject_entity_id JOIN entities o ON o.id = r.object_entity_id
            WHERE s.entity_type = 'person' AND o.entity_type = 'person' AND r.status != 'retracted'""" + scope_clause + " ORDER BY r.updated_at DESC", (scope_id,) if scope_id else ())
        return [self._decode(row, ["evidence_ids_json"]) for row in rows]

    def retract_relationship(self, relationship_id):
        relationship = self._row("SELECT * FROM relationships WHERE id = ?", (relationship_id,))
        if not relationship:
            return None
        self.connection.execute("UPDATE relationships SET status = 'retracted', updated_at = ?, revision = revision + 1 WHERE id = ?", (now_iso(), relationship_id))
        self.connection.commit()
        return self._row("SELECT * FROM relationships WHERE id = ?", (relationship_id,))

    def maintain_relationship_claim(self, relationship):
        """Write a user-confirmed relationship into the subject's semantic claims so
        person profiles, knowledge and Agent recall can reference it."""
        subject = self.get_entity(relationship.get("subject_entity_id")) or {}
        object_entity = self.get_entity(relationship.get("object_entity_id")) or {}
        predicate = str(relationship.get("predicate") or "").strip()
        if not predicate or not subject or not object_entity:
            return None
        claim = self.maintain_semantic_claim(
            person_id=subject["id"],
            dimension="relationship",
            predicate=predicate,
            value_text=object_entity.get("canonical_name") or "家人",
            evidence_ids=relationship.get("evidence_ids_json", []) or [],
            confidence=float(relationship.get("confidence") or 0.75),
            confidence_source="user-confirmed",
        )
        self.connection.commit()
        return claim

    def create_relationship(self, subject_entity_id, predicate, object_entity_id, evidence_ids=None, confidence=0.5, status="pending"):
        evidence_ids = list(dict.fromkeys(evidence_ids or []))
        subject = self.get_entity(subject_entity_id)
        object_entity = self.get_entity(object_entity_id)
        if not subject or not object_entity or (subject.get("scope_id") or "home-default") != (object_entity.get("scope_id") or "home-default"):
            raise ValueError("relationship entities must belong to the same memory space")
        scope_id = subject.get("scope_id") or "home-default"
        existing = self._row("SELECT * FROM relationships WHERE scope_id = ? AND subject_entity_id = ? AND predicate = ? AND object_entity_id = ? AND status IN ('active', 'pending') ORDER BY revision DESC LIMIT 1", (scope_id, subject_entity_id, predicate, object_entity_id))
        if existing:
            old_evidence = json.loads(existing["evidence_ids_json"] or "[]")
            merged = list(dict.fromkeys(old_evidence + evidence_ids))
            next_status = "active" if status == "active" else existing["status"]
            self.connection.execute("UPDATE relationships SET status = ?, evidence_ids_json = ?, confidence = MAX(confidence, ?), updated_at = ?, revision = revision + 1 WHERE id = ?", (next_status, json_value(merged, []), float(confidence or 0), now_iso(), existing["id"]))
            self.connection.commit()
            return next((item for item in self.list_relationships() if item["id"] == existing["id"]), None)
        relationship_id = make_id("rel")
        revision = 1
        self.connection.execute("""INSERT INTO relationships(id, scope_id, subject_entity_id, predicate, object_entity_id, status, confidence, evidence_ids_json, supersedes_relationship_id, revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)""", (relationship_id, scope_id, subject_entity_id, predicate, object_entity_id, status, float(confidence or 0), json_value(evidence_ids, []), revision, now_iso(), now_iso()))
        self.connection.commit()
        return self.list_relationships()[0]

    def confirm_relationship(self, relationship_id):
        relationship = self._row("SELECT * FROM relationships WHERE id = ?", (relationship_id,))
        if not relationship:
            return None
        self.connection.execute("UPDATE relationships SET status = 'superseded', updated_at = ? WHERE subject_entity_id = ? AND predicate = ? AND object_entity_id = ? AND status = 'active' AND id != ?", (now_iso(), relationship["subject_entity_id"], relationship["predicate"], relationship["object_entity_id"], relationship_id))
        self.connection.execute("UPDATE relationships SET status = 'active', updated_at = ? WHERE id = ?", (now_iso(), relationship_id))
        self.connection.commit()
        return next((item for item in self.list_relationships() if item["id"] == relationship_id), None)

    def create_story(self, data):
        story_id = data.get("id") or make_id("story")
        timestamp = now_iso()
        self.connection.execute(
            """INSERT INTO stories(id, title, status, outline_json, event_ids_json, tags_json, content, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (story_id, data.get("title") or "未命名故事", data.get("status", "draft"), json_value(data.get("outline"), []), json_value(data.get("event_ids"), []), json_value(data.get("tags"), []), data.get("content", ""), timestamp, timestamp),
        )
        self.connection.commit()
        return self.get_story(story_id)

    def get_story(self, story_id):
        story = self._decode(self._row("SELECT * FROM stories WHERE id = ?", (story_id,)), ["outline_json", "event_ids_json", "tags_json"])
        if story:
            story["outline"] = story.pop("outline_json")
            story["event_ids"] = story.pop("event_ids_json")
            story["tags"] = story.pop("tags_json", [])
        return story

    def list_stories(self):
        rows = self._rows("SELECT * FROM stories ORDER BY updated_at DESC")
        return [self.get_story(row["id"]) for row in rows]

    def update_story(self, story_id, fields):
        story = self.get_story(story_id)
        if not story:
            return None
        values = {}
        for key in ("title", "status", "content"):
            if key in fields:
                values[key] = fields[key]
        if "outline" in fields:
            values["outline_json"] = json_value(fields["outline"], [])
        if "event_ids" in fields:
            values["event_ids_json"] = json_value(fields["event_ids"], [])
        if "tags" in fields:
            values["tags_json"] = json_value(fields["tags"], [])
        if not values:
            return story
        assignments = ", ".join(f"{key} = ?" for key in values)
        self.connection.execute(f"UPDATE stories SET {assignments}, updated_at = ? WHERE id = ?", (*values.values(), now_iso(), story_id))
        self.connection.commit()
        return self.get_story(story_id)

    def delete_story(self, story_id):
        story = self.get_story(story_id)
        if not story:
            return None
        self.connection.execute("DELETE FROM stories WHERE id = ?", (story_id,))
        self.connection.commit()
        return {"id": story_id, "deleted": True}

    def create_invite(self, label):
        invite_id = make_id("invite")
        token = uuid.uuid4().hex
        self.connection.execute("INSERT INTO invites(id, label, token, created_at) VALUES (?, ?, ?, ?)", (invite_id, label or "家庭成员", token, now_iso()))
        self.connection.commit()
        return self._row("SELECT * FROM invites WHERE id = ?", (invite_id,))
