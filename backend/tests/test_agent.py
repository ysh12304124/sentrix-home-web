import tempfile
import unittest

from backend.agent import MemoryAgent, contains
from backend.db import MemoryStore


class FakeGamma:
    model = "test-gamma"

    def answer(self, query, context):
        return {"answer": "证据中有冰箱。", "confidence": 0.9, "evidence": [], "insufficient_evidence": False}

    def embed_text(self, text):
        return []


class RefusingGamma(FakeGamma):
    def answer(self, query, context):
        return {"answer": "证据不足", "confidence": 0.1, "evidence": [], "insufficient_evidence": True}


class RecordingGamma(FakeGamma):
    def __init__(self):
        self.focus_calls = 0
        self.answer_calls = 0
        self.contexts = []

    def answer(self, query, context):
        self.answer_calls += 1
        self.contexts.append((query, context))
        return super().answer(query, context)

    def analyze_image_focus(self, path, dimension, metadata=None):
        self.focus_calls += 1
        return {"objects": ["补全物体"], "confidence": 0.8}


class FailingClip:
    def embed_text(self, text):
        raise AssertionError("pending identity review must not use vector recall")


class ControlledClip:
    model_name = "controlled-clip"

    def __init__(self, embedding):
        self.embedding = embedding

    def embed_text(self, text):
        return self.embedding


class UntrustedClip(ControlledClip):
    evidence_ready = False


class AgentEvidenceTests(unittest.TestCase):
    def test_search_terms_do_not_match_every_filename_by_one_common_token(self):
        self.assertTrue(contains("SR_AWS_N_0016.jpg", "SR_AWS_N_0016.jpg"))
        self.assertFalse(contains("SR_AWS_N_0054.jpg", "SR_AWS_N_0016.jpg"))
        self.assertTrue(contains("SR_AWS_N_0016.jpg", "请查看 SR_AWS_N_0016.jpg 中的人做了什么"))
        self.assertFalse(contains("SR_AWS_N_0054.jpg", "请查看 SR_AWS_N_0016.jpg 中的人做了什么"))

    def test_multi_concept_chinese_query_rejects_incidental_short_clues(self):
        self.assertTrue(contains("餐桌旁的家庭照片", "餐桌旁发生了什么？"))
        self.assertFalse(contains("阴天下的繁忙集装箱港口", "火星海边生日派对"))

    def test_answer_returns_asset_observation_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "打开冰箱", "place": "厨房", "raw": {"objects": ["冰箱"]}})
            store.merge_observation_into_event(observation)
            result = MemoryAgent(store, gamma=FakeGamma()).answer("冰箱")
            kinds = {item["kind"] for item in result["evidence"]}
            observation_evidence = next(item for item in result["evidence"] if item["kind"] == "observation")
            self.assertEqual(kinds, {"event", "observation"})
            self.assertEqual(observation_evidence["asset_id"], "asset_1")
            self.assertEqual(observation_evidence["raw"]["objects"], ["冰箱"])

    def test_local_evidence_fallback_answers_when_model_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "家人在客厅聚会", "place": "客厅"})
            store.merge_observation_into_event(observation)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("family.jpg")

            self.assertFalse(result["insufficient_evidence"])
            self.assertIn("家人在客厅聚会", result["answer"])
            self.assertEqual(result["evidence"][1]["asset_id"], "asset_1")

    def test_answer_returns_structured_trace_and_evidence_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "家人在客厅聚会", "place": "客厅"})
            store.merge_observation_into_event(observation)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("客厅")

            self.assertTrue(result["retrieval_trace"])
            self.assertEqual(result["retrieval_trace"][0]["stage"], "lexical")
            self.assertIn("observations", result["evidence_layers"])
            self.assertIn("assets", result["evidence_layers"])

    def test_steward_returns_read_only_tool_trace_for_person_introduction(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "妈妈在客厅"})
            event = store.create_event({"id": "event_1", "title": "家庭时光", "summary": "妈妈在客厅"})
            store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], observation["id"]))
            store.connection.commit()
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
            store.rebuild_person_memory(person["id"])

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("介绍一下妈妈")

            tools = [item["tool"] for item in result["tool_trace"]]
            self.assertEqual(tools[:2], ["resolve_constraints", "describe_entity"])
            self.assertTrue(all(item["permission"] == "read" for item in result["tool_trace"]))
            self.assertIn("event_1", [item["event_id"] for item in result["evidence"] if item["kind"] == "event"])

    def test_steward_uses_event_timeline_tools_for_timeline_request(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "客厅聚会", "place": "客厅"})
            store.merge_observation_into_event(observation)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("客厅的时间线")

            tools = [item["tool"] for item in result["tool_trace"]]
            self.assertEqual(tools[:3], ["resolve_constraints", "find_events", "trace_timeline"])

    def test_steward_recommends_only_explicitly_requested_anchored_memories(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("妈妈", "person", "confirmed", confidence=1.0)
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "妈妈在客厅看书"})
            event = store.merge_observation_into_event(observation)
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("推荐一些妈妈的回忆")

            self.assertFalse(result["insufficient_evidence"])
            self.assertIn("推荐", result["answer"])
            self.assertIn(event["id"], [item["event_id"] for item in result["evidence"] if item["kind"] == "event"])
            self.assertIn("suggest_recall", [item["tool"] for item in result["tool_trace"]])

    def test_steward_refuses_unanchored_memory_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("推荐一些回忆")

            self.assertTrue(result["insufficient_evidence"])
            self.assertIn("人物、地点或日期", result["answer"])
            self.assertIn("request_clarification", [item["tool"] for item in result["tool_trace"]])

    def test_steward_clarifies_multiple_entity_candidates_before_declaring_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            east = store.create_entity("东湖边", "place", confidence=0.8)
            west = store.create_entity("西湖边", "place", confidence=0.8)
            for index, entity in enumerate((east, west), 1):
                asset = store.create_asset(f"asset_{index}", f"{index}.jpg", "image", f"/tmp/{index}.jpg")
                observation = store.add_observation(asset["id"], {"caption": entity["canonical_name"]})
                store.connection.execute("INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, 0.8, 'test', datetime('now'))", (entity["id"], observation["id"]))
            store.connection.commit()

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("湖边在哪里")

            self.assertTrue(result["insufficient_evidence"])
            self.assertIn("东湖边", result["answer"])
            self.assertIn("西湖边", result["answer"])
            self.assertEqual(result["tool_trace"][-1]["tool"], "request_clarification")
            self.assertEqual(len(result["clarification_candidates"]), 2)

    def test_steward_does_not_clarify_candidates_from_different_entity_types(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            store.create_entity("东湖边", "place", confidence=0.8)
            store.create_entity("西湖边", "object", confidence=0.8)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("湖边在哪里")

            self.assertTrue(result["insufficient_evidence"])
            self.assertIn("没有找到", result["answer"])
            self.assertEqual(result["clarification_candidates"], [])

    def test_steward_prefers_anchored_evidence_over_clarification(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            for name in ("东湖边", "西湖边"):
                store.create_entity(name, "place", confidence=0.8)
            asset = store.create_asset("asset_1", "lake.jpg", "image", "/tmp/lake.jpg")
            observation = store.add_observation(asset["id"], {"caption": "东湖边散步", "place": "东湖边"})
            store.merge_observation_into_event(observation)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("东湖边在哪里")

            self.assertFalse(result["insufficient_evidence"])
            self.assertEqual(result.get("clarification_candidates"), None)
            self.assertTrue(result["evidence"])

    def test_steward_never_clarifies_with_candidates_from_another_memory_space(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            store.create_entity("东湖边", "place", confidence=0.8, scope_id="album_a")
            store.create_entity("西湖边", "place", confidence=0.8, scope_id="album_a")

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("湖边在哪里", scope_id="album_b")

            self.assertTrue(result["insufficient_evidence"])
            self.assertIn("没有找到", result["answer"])
            self.assertEqual(result["clarification_candidates"], [])

    def test_steward_routes_two_confirmed_people_to_evidence_backed_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            mother = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
            father = store.create_entity("爸爸", "person", "confirmed", "父亲", 1.0)
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "父母在客厅"})
            event = store.create_event({"id": "event_1", "title": "家庭时光", "summary": "父母在客厅"})
            store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], observation["id"]))
            store.connection.commit()
            for person in (mother, father):
                store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
                store.rebuild_person_memory(person["id"])

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("比较妈妈和爸爸的回忆")

            tools = [item["tool"] for item in result["tool_trace"]]
            self.assertEqual(tools[:2], ["resolve_constraints", "compare_memories"])
            self.assertIn("共同事件", result["answer"])
            self.assertIn("event_1", [item["event_id"] for item in result["evidence"] if item["kind"] == "event"])

    def test_retrieve_ranks_exact_lexical_match_before_unrelated_vector_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            first = store.create_asset("asset_1", "kitchen.jpg", "image", "/tmp/kitchen.jpg")
            second = store.create_asset("asset_2", "garden.jpg", "image", "/tmp/garden.jpg")
            first_observation = store.add_observation(first["id"], {"caption": "厨房里的冰箱", "place": "厨房"})
            second_observation = store.add_observation(second["id"], {"caption": "花园散步", "place": "花园"})
            store.merge_observation_into_event(first_observation)
            store.merge_observation_into_event(second_observation)

            result = MemoryAgent(store, gamma=FakeGamma()).retrieve("冰箱")

            self.assertEqual(result["observations"][0]["id"], first_observation["id"])

    def test_low_similarity_vector_hit_degrades_to_query_gap_instead_of_answering(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "harbor.jpg", "image", "/tmp/harbor.jpg")
            observation = store.add_observation(asset["id"], {"caption": "港口集装箱", "place": "港口"})
            event = store.merge_observation_into_event(observation)
            store.upsert_vector("episodic", "event", event["id"], [0.1, 0.995], "controlled-clip")

            result = MemoryAgent(store, gamma=FakeGamma(), clip=ControlledClip([1.0, 0.0])).answer("火星生日派对在哪里")

            self.assertTrue(result["insufficient_evidence"])
            self.assertEqual(result["confidence"], 0.0)
            self.assertEqual(result["evidence"], [])
            self.assertIn("没有找到", result["answer"])
            self.assertEqual(store.get_query_gap(result["query_gap_id"])["missing_dimension"], "spatial_relation")
            vector = next(item for item in result["retrieval_trace"] if item["stage"] == "vector")
            self.assertEqual(vector["counts"]["accepted"], 0)

    def test_high_similarity_vector_without_query_clues_cannot_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "harbor.jpg", "image", "/tmp/harbor.jpg")
            observation = store.add_observation(asset["id"], {"caption": "港口集装箱", "place": "港口"})
            event = store.merge_observation_into_event(observation)
            store.upsert_vector("episodic", "event", event["id"], [1.0, 0.0], "controlled-clip")

            result = MemoryAgent(store, gamma=FakeGamma(), clip=ControlledClip([1.0, 0.0])).answer("火星海边生日派对")

            self.assertTrue(result["insufficient_evidence"])
            self.assertEqual(result["evidence"], [])
            vector = next(item for item in result["retrieval_trace"] if item["stage"] == "vector")
            self.assertEqual(vector["counts"]["accepted"], 0)

    def test_high_similarity_vector_hit_can_recover_anchored_event_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "harbor.jpg", "image", "/tmp/harbor.jpg")
            observation = store.add_observation(asset["id"], {"caption": "港口集装箱", "place": "港口"})
            event = store.merge_observation_into_event(observation)
            store.upsert_vector("episodic", "event", event["id"], [1.0, 0.0], "controlled-clip")

            result = MemoryAgent(store, gamma=RefusingGamma(), clip=ControlledClip([1.0, 0.0])).answer("港口的活动")

            self.assertFalse(result["insufficient_evidence"])
            self.assertIn(event["id"], [item["event_id"] for item in result["evidence"] if item["kind"] == "event"])

    def test_untrusted_vector_model_cannot_supply_agent_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "harbor.jpg", "image", "/tmp/harbor.jpg")
            observation = store.add_observation(asset["id"], {"caption": "港口集装箱", "place": "港口"})
            event = store.merge_observation_into_event(observation)
            store.upsert_vector("episodic", "event", event["id"], [1.0, 0.0], "untrusted-clip")

            result = MemoryAgent(store, gamma=FakeGamma(), clip=UntrustedClip([1.0, 0.0])).answer("火星生日派对")

            self.assertTrue(result["insufficient_evidence"])
            self.assertEqual(result["evidence"], [])
            vector = next(item for item in result["retrieval_trace"] if item["stage"] == "vector")
            self.assertEqual(vector["status"], "unavailable")

    def test_person_activity_query_uses_event_level_semantic_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            entity = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "家人在餐桌旁", "activity": "家庭聚餐", "place": "餐厅"})
            event = store.create_event({
                "id": "event_1", "title": "家庭聚餐", "event_type": "聚餐", "activity": "家庭聚餐",
                "place": "餐厅", "summary": "妈妈在餐厅参与家庭聚餐", "time_start": "2025-01-01T12:00:00+00:00",
                "time_end": "2025-01-01T13:00:00+00:00", "participants": [{"entity_id": entity["id"], "name": "妈妈"}],
            })
            store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], observation["id"]))
            store.connection.commit()
            store.upsert_event_participant(event["id"], entity["id"], "visible_subject", [observation["id"]], 0.9)
            store.maintain_semantic_claim(entity["id"], "activity", "参与", "家庭聚餐", [observation["id"]], [event["id"]], 0.9)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("妈妈参与过哪些活动？")

            self.assertIn("家庭聚餐", result["answer"])
            claim_evidence = [item for item in result["evidence"] if item["kind"] == "semantic_claim"]
            self.assertTrue(claim_evidence)
            self.assertIn(event["id"], claim_evidence[0]["supporting_event_ids"])

    def test_person_activity_query_keeps_all_activity_claims_ahead_of_clothing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            entity = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "家庭记录"})
            event = store.create_event({"id": "event_1", "title": "家庭聚餐", "activity": "家庭聚餐"})
            store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], observation["id"]))
            store.connection.commit()
            store.maintain_semantic_claim(entity["id"], "clothing", "穿着", "红色外套", [observation["id"]], [event["id"]], 0.8)
            store.maintain_semantic_claim(entity["id"], "activity", "参与", "家庭聚餐", [observation["id"]], [event["id"]], 0.9)
            store.maintain_semantic_claim(entity["id"], "activity", "参与", "公园散步", [observation["id"]], ["event_2"], 0.9)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("妈妈参与过哪些活动？")

            self.assertIn("家庭聚餐", result["answer"])
            self.assertIn("公园散步", result["answer"])
            activity_evidence = [item for item in result["evidence"] if item["kind"] == "semantic_claim" and item["dimension"] == "activity"]
            self.assertEqual({item["value_text"] for item in activity_evidence}, {"家庭聚餐", "公园散步"})

    def test_place_and_date_queries_filter_to_matching_events(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            restaurant_asset = store.create_asset("asset_1", "restaurant.jpg", "image", "/tmp/restaurant.jpg")
            park_asset = store.create_asset("asset_2", "park.jpg", "image", "/tmp/park.jpg")
            restaurant_observation = store.add_observation(restaurant_asset["id"], {"caption": "餐厅聚餐"})
            park_observation = store.add_observation(park_asset["id"], {"caption": "公园散步"})
            restaurant_event = store.create_event({
                "id": "event_restaurant", "title": "餐厅聚餐", "place": "家中餐厅",
                "time_start": "2025-05-10T18:00:00+00:00", "summary": "家中餐厅聚餐",
            })
            park_event = store.create_event({
                "id": "event_park", "title": "公园散步", "place": "城市公园",
                "time_start": "2025-05-11T10:00:00+00:00", "summary": "城市公园散步",
            })
            store.connection.executemany(
                "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
                [(restaurant_event["id"], restaurant_observation["id"]), (park_event["id"], park_observation["id"])],
            )
            store.connection.commit()

            place = MemoryAgent(store, gamma=FakeGamma()).answer("家中餐厅发生了什么？")
            date = MemoryAgent(store, gamma=FakeGamma()).answer("2025-05-10发生了什么？")

            for result in (place, date):
                event_ids = [item["event_id"] for item in result["evidence"] if item["kind"] == "event"]
                self.assertEqual(event_ids, [restaurant_event["id"]])
                self.assertIn("家中餐厅聚餐", result["answer"])
                vector_stage = next(item for item in result["retrieval_trace"] if item["stage"] == "vector")
                self.assertEqual(vector_stage["status"], "skipped")

    def test_private_place_uses_user_alias_in_agent_context_and_response(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "dinner.jpg", "image", "/tmp/dinner.jpg")
            observation = store.add_observation(asset["id"], {"caption": "家中餐厅聚餐", "place": "家中餐厅"})
            event = store.create_event({"id": "event_1", "title": "晚餐", "place": "家中餐厅", "summary": "家中餐厅聚餐"})
            store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], observation["id"]))
            store.connection.commit()
            place = store.create_entity("家中餐厅", "place", confidence=1.0)
            store.set_entity_property(place["id"], "alias", "我们的饭桌", [observation["id"]])
            store.set_entity_property(place["id"], "private_flag", True, [observation["id"]])
            gamma = RecordingGamma()

            agent = MemoryAgent(store, gamma=gamma)
            result = agent.answer("晚餐")

            self.assertIn("我们的饭桌", result["answer"])
            self.assertNotIn("家中餐厅", result["answer"])
            self.assertEqual(result["evidence"][0]["place"], "我们的饭桌")
            self.assertNotIn("家中餐厅", gamma.contexts[0][1])

    def test_person_clothing_and_object_queries_use_specific_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            entity = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
            microphone_asset = store.create_asset("asset_1", "microphone.jpg", "image", "/tmp/microphone.jpg")
            book_asset = store.create_asset("asset_2", "book.jpg", "image", "/tmp/book.jpg")
            microphone_observation = store.add_observation(microphone_asset["id"], {"caption": "妈妈在麦克风前讲话", "objects": ["麦克风"]})
            book_observation = store.add_observation(book_asset["id"], {"caption": "桌上的书", "objects": ["书"]})
            event = store.create_event({"id": "event_1", "title": "演讲", "summary": "妈妈讲话"})
            store.connection.executemany(
                "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
                [(event["id"], microphone_observation["id"]), (event["id"], book_observation["id"])],
            )
            store.connection.commit()
            store.upsert_event_participant(event["id"], entity["id"], "visible_subject", [microphone_observation["id"]], 0.9)
            store.maintain_semantic_claim(entity["id"], "clothing", "穿着", "红色外套", [microphone_observation["id"]], [event["id"]], 0.9, confidence_source="user")

            clothing = MemoryAgent(store, gamma=FakeGamma()).answer("妈妈穿过什么衣服？")
            object_result = MemoryAgent(store, gamma=FakeGamma()).answer("有哪些麦克风相关证据？")

            self.assertIn("红色外套", clothing["answer"])
            self.assertIn("麦克风", object_result["answer"])
            observation_ids = [item["observation_id"] for item in object_result["evidence"] if item["kind"] == "observation"]
            self.assertEqual(observation_ids, [microphone_observation["id"]])

    def test_person_clothing_query_does_not_turn_scene_clothing_into_a_person_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            entity = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
            asset = store.create_asset("asset_1", "scene.jpg", "image", "/tmp/scene.jpg")
            observation = store.add_observation(asset["id"], {"caption": "两人站在一起", "clothing": ["红色外套", "蓝色制服"]})
            event = store.create_event({"id": "event_1", "title": "合影"})
            store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], observation["id"]))
            store.connection.commit()
            store.upsert_event_participant(event["id"], entity["id"], "visible_subject", [observation["id"]], 0.9)

            result = MemoryAgent(store, gamma=FakeGamma()).answer("妈妈穿过什么衣服？")

            self.assertIn("没有可归属到该人物", result["answer"])
            self.assertNotIn("红色外套", result["answer"])

    def test_person_clothing_query_returns_face_scoped_evidence_and_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "portrait.jpg", "image", "/tmp/portrait.jpg")
            observation = store.add_observation(asset["id"], {"caption": "人物肖像", "clothing": ["场景级蓝色外套"]})
            event = store.merge_observation_into_event(observation)
            face = store.add_face_instance(
                asset["id"], observation["id"],
                {"bbox": [10, 20, 40, 60], "confidence": 0.95, "embedding": [1, 0, 0]},
            )
            person = store.confirm_face_cluster(face["cluster_id"], "妈妈", "母亲")["entity"]
            appearance = store.record_person_appearance_evidence(
                person["id"], face["id"], [0, 0, 120, 200], ["红色针织衫"], 0.9, "test-vision",
            )
            store.rebuild_person_memory(person["id"])

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("妈妈穿过什么衣服？")

            self.assertIn("红色针织衫", result["answer"])
            appearance_evidence = next(item for item in result["evidence"] if item["kind"] == "person_appearance")
            self.assertEqual(appearance_evidence["id"], appearance["id"])
            self.assertEqual(appearance_evidence["asset_id"], asset["id"])
            claim = next(item for item in result["evidence"] if item["kind"] == "semantic_claim" and item["dimension"] == "clothing")
            self.assertEqual(claim["supporting_event_ids"], [event["id"]])

    def test_object_query_with_existing_observation_evidence_does_not_trigger_visual_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "microphone.jpg", "image", "/tmp/microphone.jpg")
            observation = store.add_observation(asset["id"], {"caption": "演讲", "objects": ["麦克风"]})
            store.merge_observation_into_event(observation)
            gamma = RecordingGamma()

            result = MemoryAgent(store, gamma=gamma).answer("有哪些麦克风相关证据？")

            self.assertIn("麦克风", result["answer"])
            self.assertEqual(gamma.focus_calls, 0)

    def test_answer_turn_routes_query_and_keeps_bounded_conversation_context(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "餐桌旁的家庭照片"})
            store.merge_observation_into_event(observation)
            gamma = RecordingGamma()
            agent = MemoryAgent(store, gamma=gamma)

            first = agent.answer_turn("餐桌旁发生了什么？", "conversation-1")
            second = agent.answer_turn("继续说说这张照片。", "conversation-1")

            self.assertEqual(first["intent"], "query")
            self.assertEqual(second["conversation_id"], "conversation-1")
            self.assertGreaterEqual(gamma.answer_calls, 2)
            self.assertIn("餐桌旁发生了什么？", gamma.contexts[-1][1])

    def test_answer_turn_feedback_persists_without_normal_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            gap = store.create_query_gap("哪天穿了红衣服", "clothing", ["asset_1"])
            gamma = RecordingGamma()
            agent = MemoryAgent(store, gamma=gamma)

            result = agent.answer_turn(
                "实际是红色针织衫",
                "conversation-2",
                {"query_gap_id": gap["id"], "correction": "红色针织衫"},
            )

            self.assertEqual(result["intent"], "feedback")
            self.assertEqual(result["conversation_id"], "conversation-2")
            self.assertEqual(store.get_query_gap(gap["id"])["status"], "resolved")
            self.assertEqual(gamma.answer_calls, 0)

    def test_pending_identity_query_uses_review_fallback_and_keeps_candidate_name_private(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "家庭照片"})
            cluster = store.create_face_cluster([0.1, 0.2, 0.3], 0.71)
            gamma = RecordingGamma()

            result = MemoryAgent(store, gamma=gamma, clip=FailingClip()).answer("还有哪些待命名人物？")

            self.assertEqual(result["model"], "sentrix-identity-review")
            self.assertTrue(result["insufficient_evidence"])
            self.assertEqual(gamma.answer_calls, 0)
            self.assertIn("1 位待命名成员", result["answer"])
            self.assertNotIn(cluster["id"], result["answer"])
            self.assertEqual(result["evidence"][0]["name"], "待命名成员 1")
            self.assertEqual(store.get_query_gap(result["query_gap_id"])["missing_dimension"], "identity")

    def test_image_query_returns_structured_asset_result(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "客厅里的家庭照片"})
            store.merge_observation_into_event(observation)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn("客厅里的图片", "conversation-3")

            self.assertTrue(result["image_results"])
            self.assertEqual(result["image_results"][0]["asset_id"], "asset_1")
            self.assertIn("/api/assets/asset_1/file", result["image_results"][0]["media_url"])

    def test_intent_router_distinguishes_clarification(self):
        agent = MemoryAgent(MemoryStore(":memory:"), gamma=FakeGamma())
        self.assertEqual(agent.classify_intent("我说的是妈妈，不是爸爸"), "clarification")


if __name__ == "__main__":
    unittest.main()
