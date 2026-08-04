import json
import tempfile
import unittest

from backend.agent import MemoryAgent, contains
from backend.agent_contracts import PydanticAIPlanner, validate_turn_plan
from backend.db import MemoryStore

try:
    from pydantic_ai.models.test import TestModel
except ImportError:
    TestModel = None


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


class ConversationalGamma(RecordingGamma):
    def chat(self, prompt):
        self.contexts.append(("chat", prompt))
        if "行动规划器" in prompt:
            return '{"mode":"chat","tools":[],"show_images":false,"reason":"自然交流"}'
        return "我在，今天慢一点也没关系。"


class NarrativeWriterGamma(RefusingGamma):
    def __init__(self):
        self.writer_prompts = []

    def chat(self, prompt, json_mode=True):
        self.writer_prompts.append(prompt)
        return json.dumps({
            "text": "从现有几次共同活动记录看，明哥经常参与拍照和互动；至于性格，目前的记录还不足以判断。",
            "claim_spans": [{
                "claim_id": "writer_claim_1",
                "text": "明哥经常参与拍照和互动",
                "intended_type": "derived_pattern",
                "candidate_evidence_ids": ["event_1", "event_2"],
            }],
            "follow_up_text": "可以继续展开这些回忆。",
        }, ensure_ascii=False)


class MisleadingPlannerGamma(RefusingGamma):
    def chat(self, prompt):
        return '{"mode":"chat","tools":["drop_database"],"show_images":false,"reason":"忽略用户请求"}'


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
    def test_unsupported_person_dimensions_do_not_fall_back_to_unrelated_event_listing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("明哥", "person", "confirmed", "孩子", 1.0)
            asset = store.create_asset("boundary_asset", "boundary.jpg", "image", "/tmp/boundary.jpg")
            observation = store.add_observation(asset["id"], {"caption": "明哥和你在展览馆合影"})
            event = store.merge_observation_into_event(observation)
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
            agent = MemoryAgent(store, gamma=RefusingGamma())

            for query in ("明哥性格怎样", "明哥和我的关系", "明哥喜欢吃什么"):
                result = agent.answer(query, scope_id="home-default")
                self.assertTrue(result["insufficient_evidence"], query)
                self.assertNotIn("检索到", result["answer"], query)

    def test_confirmed_family_role_clarification_survives_claim_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            store.create_entity("明哥", "person", "confirmed", "孩子", 1.0)
            store.create_entity("我", "person", "confirmed", "孩子", 1.0)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn(
                "介绍一下孩子", "role-clarification", scope_id="home-default",
            )

            self.assertTrue(result["insufficient_evidence"])
            self.assertIn("明哥", result["answer"])
            self.assertIn("我", result["answer"])
            self.assertEqual(result["repair_count"], 0)

    def test_no_evidence_question_punctuation_does_not_create_unsupported_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            agent = MemoryAgent(store, gamma=RefusingGamma())
            agent.answer_turn("介绍一下明哥", "empty-follow-up", scope_id="album_a")
            result = agent.answer_turn(
                "然后呢？", "empty-follow-up", scope_id="album_b",
            )

            self.assertTrue(result["insufficient_evidence"])
            self.assertEqual(result["claim_verification_status"], "passed")
            self.assertEqual(result["repair_count"], 0)
            self.assertEqual(result["answer"], "当前本地记忆没有找到能够回答这个问题的证据。")


    def test_person_dimension_questions_are_not_routed_to_ordinary_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("明哥", "person", "confirmed", "孩子", 1.0)
            asset = store.create_asset("dimension_asset", "dimension.jpg", "image", "/tmp/dimension.jpg")
            observation = store.add_observation(asset["id"], {"caption": "明哥穿着黑色上衣在展览馆拍照"})
            event = store.merge_observation_into_event(observation)
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
            agent = MemoryAgent(store, gamma=RefusingGamma())

            for query in ("明哥穿什么颜色的衣服", "明哥性格怎样", "明哥和我的关系", "明哥喜欢吃什么"):
                result = agent.answer_turn(query, f"dimension-{query}", scope_id="home-default")
                self.assertTrue(result["memory_used"], query)
                self.assertNotEqual(result["answer"], "我在听。", query)

    def test_timeline_answer_exposes_claim_verification_and_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("明哥", "person", "confirmed", "孩子", 1.0)
            asset = store.create_asset("timeline_asset", "timeline.jpg", "image", "/tmp/timeline.jpg")
            observation = store.add_observation(asset["id"], {"caption": "明哥在展览馆拍照"})
            event = store.merge_observation_into_event(observation)
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn(
                "明哥的时间线", "claim-surface", scope_id="home-default",
            )

            self.assertTrue(result["claims"])
            self.assertEqual(len(result["claims"]), len(result["claim_verifications"]))
            self.assertTrue(result["evidence_bundles"])
            self.assertTrue(result["segments"])
            self.assertTrue(result["claim_evidence_index"])
            self.assertIn(result["claim_verification_status"], {"passed", "passed_after_repair", "blocked"})

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

    def test_person_introduction_builds_profile_summary_instead_of_event_listing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("明哥", "person", "confirmed", "哥哥", 1.0)
            myself = store.create_entity("我", "person", "confirmed", "本人", 1.0)
            evidence = []
            for index, (activity, place) in enumerate((("拍照", "室内剧场"), ("参观展示", "展览馆")), 1):
                asset = store.create_asset(f"asset_{index}", f"memory_{index}.jpg", "image", f"/tmp/memory_{index}.jpg")
                observation = store.add_observation(asset["id"], {"caption": f"明哥在{place}{activity}"})
                event = store.create_event({
                    "id": f"event_{index}", "title": activity, "activity": activity,
                    "place": place, "summary": f"明哥在{place}{activity}",
                    "time_start": f"2025-0{index}-12T12:00:00+00:00",
                })
                store.connection.execute(
                    "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
                    (event["id"], observation["id"]),
                )
                store.connection.commit()
                store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
                store.upsert_event_participant(event["id"], myself["id"], "visible_subject", [observation["id"]], 0.9)
                evidence.append((observation, event))
            store.maintain_semantic_claim(
                person["id"], "clothing", "穿着", "深色外套",
                [evidence[0][0]["id"]], [evidence[0][1]["id"]], 0.85,
            )
            store.rebuild_person_memory(person["id"])

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("介绍一下明哥")

            profile = result["person_profile"]
            self.assertEqual(profile["entity_id"], person["id"])
            self.assertIn("哥哥", result["answer"])
            self.assertIn("拍照", result["answer"])
            self.assertIn("参观展示", result["answer"])
            self.assertNotIn("2025-01-12 12:00 · 室内剧场", result["answer"])
            self.assertNotIn("；2025-", result["answer"])
            self.assertGreaterEqual(len(profile["sections"]), 2)
            for section in profile["sections"]:
                self.assertTrue(section["evidence_ids"])
            self.assertTrue(result["evidence_layers"]["claims"])

    def test_person_profile_writer_receives_context_packet_and_returns_natural_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("明哥", "person", "confirmed", "孩子", 1.0)
            for index, activity in enumerate(("拍照", "互动"), 1):
                asset = store.create_asset(f"asset_writer_{index}", f"writer_{index}.jpg", "image", f"/tmp/writer_{index}.jpg")
                observation = store.add_observation(asset["id"], {"caption": f"明哥{activity}"})
                event = store.create_event({
                    "id": f"event_{index}", "activity": activity, "summary": f"明哥{activity}",
                    "time_start": f"2025-0{index}-12T12:00:00+00:00",
                })
                store.connection.execute(
                    "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
                    (event["id"], observation["id"]),
                )
                store.connection.commit()
                store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
            store.rebuild_person_memory(person["id"])
            gamma = NarrativeWriterGamma()

            result = MemoryAgent(store, gamma=gamma).answer("介绍一下明哥")

            self.assertIn("从现有几次共同活动记录看", result["answer"])
            self.assertIn("narrative_context_packet", result["person_profile"])
            packet = result["person_profile"]["narrative_context_packet"]
            self.assertEqual(packet["dialogue_goal"], "person_introduction")
            self.assertEqual(packet["focus"]["people"], [person["id"]])
            self.assertTrue(packet["relevant_scenes"])
            scene = packet["relevant_scenes"][0]
            self.assertTrue({
                "scene_id", "event_id", "time_start", "time_end", "assets",
                "observations", "participants", "source_revision", "confidence",
            }.issubset(scene))
            self.assertLessEqual(len(scene["observations"]), 12)
            self.assertLessEqual(len(scene["assets"]), 6)
            self.assertTrue(packet["evidence_map"])
            self.assertTrue(result["claims"])
            self.assertEqual(len(result["claims"]), len(result["evidence_bundles"]))
            self.assertIn("canonical", gamma.writer_prompts[0])
            self.assertNotIn("cluster_", result["answer"])

    def test_dialogue_state_uses_bounded_focus_stack_with_decay(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("明哥", "person", "confirmed", "孩子", 1.0)
            asset = store.create_asset("focus_asset", "focus.jpg", "image", "/tmp/focus.jpg")
            observation = store.add_observation(asset["id"], {"caption": "明哥在展览馆拍照", "place": "展览馆"})
            event = store.merge_observation_into_event(observation)
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
            agent = MemoryAgent(store, gamma=RefusingGamma())

            first = agent.answer_turn("介绍一下明哥", "focus-conversation", scope_id="home-default")
            second = agent.answer_turn("然后呢？", "focus-conversation", scope_id="home-default")

            focus = second["dialogue_state"]["focus_stack"]
            self.assertLessEqual(len([item for item in focus if item["type"] == "person"]), 3)
            self.assertLessEqual(len([item for item in focus if item["type"] == "event"]), 3)
            self.assertTrue(any(item["id"] == person["id"] for item in focus))
            self.assertTrue(any(item["id"] == event["id"] for item in focus))
            self.assertTrue(all(0 < item["salience"] <= 1.3 for item in focus))
            self.assertGreaterEqual(second["dialogue_state"]["turn_index"], first["dialogue_state"]["turn_index"])

    def test_dialogue_focus_stack_is_cleared_when_scope_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("scope_asset", "scope.jpg", "image", "/tmp/scope.jpg", scope_id="album_a")
            observation = store.add_observation(asset["id"], {"caption": "album a 的家庭活动"}, scope_id="album_a")
            event = store.merge_observation_into_event(observation)
            agent = MemoryAgent(store, gamma=RefusingGamma())
            agent.answer_turn("album a 的家庭活动", "scope-focus", scope_id="album_a")

            result = agent.answer_turn("然后呢？", "scope-focus", scope_id="album_b")

            self.assertEqual(result["dialogue_state"]["scope_id"], "album_b")
            self.assertFalse(any(item["id"] == event["id"] for item in result["dialogue_state"].get("focus_stack", [])))

    def test_profile_claims_bind_chinese_clothing_phrases_to_appearance_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            agent = MemoryAgent(store, gamma=RefusingGamma())

            evidence_ids = agent._lexical_claim_evidence_ids(
                {"claim_kind": "family_fact", "text": "从外观上看，他常穿着黑色连帽衫或针织上衣，并佩戴银色项链。"},
                [{"kind": "person_appearance", "id": "appearance-1", "clothing": ["黑色连帽衫"], "scope_id": "home-default"}],
            )

            self.assertEqual(evidence_ids, ["appearance-1"])

    def test_confirmed_person_introduction_does_not_offer_pending_clusters(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
            pending = store.create_entity("待确认人物簇 · cluster_test", "person", "pending", confidence=0.8)
            asset = store.create_asset("asset_1", "mother.jpg", "image", "/tmp/mother.jpg")
            observation = store.add_observation(asset["id"], {"caption": "妈妈在客厅阅读"})
            event = store.create_event({"id": "event_1", "activity": "阅读", "place": "客厅", "summary": "妈妈在客厅阅读"})
            store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], observation["id"]))
            store.connection.commit()
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
            store.rebuild_person_memory(person["id"])

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("介绍一下妈妈")

            self.assertEqual(result["person_profile"]["entity_id"], person["id"])
            self.assertEqual(result["clarification_candidates"], [])
            self.assertNotIn(pending["id"], result["answer"])
            self.assertNotIn("cluster_test", result["answer"])

    def test_family_role_resolves_confirmed_person_when_canonical_name_is_internal(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("person_confirmed_1", "person", "confirmed", "母亲", 1.0)
            pending = store.create_entity("待确认人物簇 · cluster_role", "person", "pending", confidence=0.9)
            asset = store.create_asset("asset_role", "role.jpg", "image", "/tmp/role.jpg")
            observation = store.add_observation(asset["id"], {"caption": "已确认的母亲在客厅"})
            event = store.create_event({"id": "event_role", "activity": "阅读", "place": "客厅", "summary": "已确认的母亲在客厅阅读"})
            store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], observation["id"]))
            store.connection.commit()
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
            store.rebuild_person_memory(person["id"])

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("介绍一下妈妈")

            self.assertEqual(result["identity_resolution"]["status"], "resolved")
            self.assertEqual(result["person_profile"]["entity_id"], person["id"])
            self.assertEqual(result["clarification_candidates"], [])
            self.assertNotIn("cluster_role", result["answer"])

    def test_ambiguous_confirmed_people_have_human_readable_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            first = store.create_entity("妈妈（北方）", "person", "confirmed", "母亲", 0.9)
            second = store.create_entity("妈妈（南方）", "person", "confirmed", "母亲", 0.9)
            store.create_entity("待确认人物簇 · cluster_ignored", "person", "pending", confidence=0.99)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("介绍一下妈妈")

            self.assertEqual(result["identity_resolution"]["status"], "ambiguous")
            names = {item["name"] for item in result["clarification_candidates"]}
            self.assertEqual(names, {first["canonical_name"], second["canonical_name"]})
            self.assertTrue(all(item["family_role"] == "母亲" for item in result["clarification_candidates"]))
            self.assertNotIn("cluster_ignored", result["answer"])

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
            self.assertEqual(gamma.answer_calls, 1)
            self.assertEqual(second["dialogue_plan"]["mode"], "contextual_follow_up")

    def test_dialogue_follow_up_reuses_verified_events_within_the_same_memory_space(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg", scope_id="album_a")
            observation = store.add_observation(asset["id"], {"caption": "妈妈在餐桌旁切蛋糕", "captured_at": "2025-05-01T10:00:00+00:00"})
            event = store.merge_observation_into_event(observation)
            agent = MemoryAgent(store, gamma=RefusingGamma())

            first = agent.answer_turn("餐桌旁发生了什么？", "dialogue-1", scope_id="album_a")
            second = agent.answer_turn("然后呢？", "dialogue-1", scope_id="album_a")

            self.assertIn(event["id"], first["dialogue_state"]["active_event_ids"])
            self.assertEqual(second["dialogue_plan"]["mode"], "contextual_follow_up")
            self.assertIn(event["id"], [item["event_id"] for item in second["evidence"] if item["kind"] == "event"])
            self.assertEqual(second["dialogue_state"]["scope_id"], "album_a")
            self.assertEqual(store.count("query_gaps"), 0)

    def test_dialogue_evidence_exposes_source_level_time_and_confidence_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "客厅聚会", "confidence": 0.8, "captured_at": "2025-05-01T10:00:00+00:00"})
            store.merge_observation_into_event(observation)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn("客厅发生了什么？")

            self.assertTrue(result["evidence_order"])
            self.assertEqual(result["evidence_order"][0]["source_level"], "derived_event")
            self.assertTrue(all("confidence" in item and "time" in item for item in result["evidence_order"]))

    def test_dialogue_resolves_pronoun_to_the_verified_person_event_context(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("妈妈", "person", "confirmed", confidence=1.0, scope_id="album_a")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg", scope_id="album_a")
            observation = store.add_observation(asset["id"], {"caption": "妈妈在客厅看书"})
            event = store.merge_observation_into_event(observation)
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
            agent = MemoryAgent(store, gamma=RefusingGamma())

            first = agent.answer_turn("介绍一下妈妈", "dialogue-pronoun", scope_id="album_a")
            second = agent.answer_turn("她后来呢？", "dialogue-pronoun", scope_id="album_a")

            self.assertIn(event["id"], first["dialogue_state"]["active_event_ids"])
            self.assertEqual(second["dialogue_plan"]["mode"], "contextual_follow_up")
            self.assertIn(event["id"], [item["event_id"] for item in second["evidence"] if item["kind"] == "event"])

    def test_dialogue_explicit_new_subject_does_not_reuse_previous_event_context(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            father = store.create_entity("爸爸", "person", "confirmed", confidence=1.0, scope_id="album_a")
            mother = store.create_entity("妈妈", "person", "confirmed", confidence=1.0, scope_id="album_a")
            father_asset = store.create_asset("father_asset", "father.jpg", "image", "/tmp/father.jpg", scope_id="album_a")
            father_observation = store.add_observation(father_asset["id"], {"caption": "爸爸在书房看书"})
            father_event = store.merge_observation_into_event(father_observation)
            store.upsert_event_participant(father_event["id"], father["id"], "visible_subject", [father_observation["id"]], 0.9)
            mother_asset = store.create_asset("mother_asset", "mother.jpg", "image", "/tmp/mother.jpg", scope_id="album_a")
            mother_observation = store.add_observation(mother_asset["id"], {"caption": "妈妈在花园浇花"})
            mother_event = store.merge_observation_into_event(mother_observation)
            store.upsert_event_participant(mother_event["id"], mother["id"], "visible_subject", [mother_observation["id"]], 0.9)
            agent = MemoryAgent(store, gamma=RefusingGamma())

            agent.answer_turn("介绍一下爸爸", "dialogue-topic-switch", scope_id="album_a")
            second = agent.answer_turn("然后介绍一下妈妈", "dialogue-topic-switch", scope_id="album_a")

            self.assertEqual(second["dialogue_plan"]["mode"], "planned_query")
            self.assertIn(mother_event["id"], [item["event_id"] for item in second["evidence"] if item["kind"] == "event"])
            self.assertNotIn(father_event["id"], [item["event_id"] for item in second["evidence"] if item["kind"] == "event"])

    def test_dialogue_selected_clarification_candidate_becomes_active_entity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            east = store.create_entity("东湖边", "place", confidence=0.9, scope_id="album_a")
            west = store.create_entity("西湖边", "place", confidence=0.9, scope_id="album_a")
            agent = MemoryAgent(store, gamma=RefusingGamma())

            first = agent.answer_turn("湖边在哪里", "dialogue-selection", scope_id="album_a")
            second = agent.answer_turn("东湖边", "dialogue-selection", scope_id="album_a", selected_entity_id=east["id"])

            self.assertTrue(first["clarification_candidates"])
            self.assertEqual(second["dialogue_plan"]["mode"], "clarification_selection")
            self.assertIn(east["id"], second["dialogue_state"]["active_entity_ids"])
            self.assertNotIn(west["id"], second["dialogue_state"]["active_entity_ids"])
            self.assertEqual(second["tool_trace"][0]["constraints"]["selected_entity_id"], east["id"])

    def test_dialogue_does_not_reuse_context_when_memory_space_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg", scope_id="album_a")
            observation = store.add_observation(asset["id"], {"caption": "餐桌旁聚会"})
            store.merge_observation_into_event(observation)
            agent = MemoryAgent(store, gamma=RefusingGamma())

            agent.answer_turn("餐桌旁发生了什么？", "dialogue-scope", scope_id="album_a")
            second = agent.answer_turn("然后呢？", "dialogue-scope", scope_id="album_b")

            self.assertNotEqual(second["dialogue_plan"]["mode"], "contextual_follow_up")

    def test_dialogue_keeps_only_verified_state_across_agent_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg", scope_id="album_a")
            observation = store.add_observation(asset["id"], {"caption": "餐桌旁聚会"})
            event = store.merge_observation_into_event(observation)
            first_agent = MemoryAgent(store, gamma=RefusingGamma())

            first_agent.answer_turn("餐桌旁发生了什么？", "dialogue-persisted", scope_id="album_a")
            restarted_agent = MemoryAgent(MemoryStore(f"{directory}/memory.db"), gamma=RefusingGamma())
            second = restarted_agent.answer_turn("然后呢？", "dialogue-persisted", scope_id="album_a")

            self.assertEqual(second["dialogue_plan"]["mode"], "contextual_follow_up")
            self.assertIn(event["id"], [item["event_id"] for item in second["evidence"] if item["kind"] == "event"])
            self.assertTrue({"scope_id", "active_event_ids", "active_entity_ids", "semantic_group_ids", "evidence_ids", "unresolved_ambiguity"}.issubset(second["dialogue_state"]))
            self.assertIn("focus_stack", second["dialogue_state"])
            self.assertIn("recent_evidence_ids", second["dialogue_state"])

    def test_dialogue_uses_narrative_style_for_an_entity_introduction(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            person = store.create_entity("妈妈", "person", "confirmed", confidence=1.0)
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "妈妈在客厅看书", "captured_at": "2025-05-01T10:00:00+00:00"})
            event = store.merge_observation_into_event(observation)
            store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn("介绍一下妈妈")

            self.assertEqual(result["dialogue_plan"]["style"], "narrative")
            self.assertIn("根据目前可回溯的记忆", result["answer"])

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

    def test_feedback_can_bind_to_an_explicit_entity_property_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            entity = store.create_entity("湖边", "place", confidence=0.8)
            agent = MemoryAgent(store, gamma=RecordingGamma())

            result = agent.answer_turn(
                "这里应叫作西湖边",
                feedback={"correction": "这里应叫作西湖边", "target_entity_id": entity["id"], "target_property_key": "alias"},
            )

            self.assertEqual(result["intent"], "feedback")
            self.assertFalse(result["insufficient_evidence"])
            self.assertEqual(result["feedback"]["target_entity_id"], entity["id"])
            self.assertEqual(result["feedback"]["target_property_key"], "alias")
            self.assertEqual(store.list_entity_properties(entity["id"]), [])

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

    def test_autonomous_turn_keeps_ordinary_chat_outside_memory_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            agent = MemoryAgent(store, gamma=RecordingGamma())

            result = agent.answer_turn("今天有点累，想聊聊天", "chat-conversation")

            self.assertEqual(result["intent"], "chat")
            self.assertEqual(result["agent_plan"]["mode"], "chat")
            self.assertEqual(result["agent_plan"]["tools"], [])
            self.assertEqual(result["evidence"], [])
            self.assertEqual(result["image_results"], [])
            self.assertEqual(agent.gamma.answer_calls, 0)

    def test_natural_chat_does_not_inject_household_memory_without_a_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
            agent = MemoryAgent(store, gamma=ConversationalGamma())

            result = agent.answer_turn("今天有点累", "chat-with-memory")

            self.assertEqual(result["intent"], "chat")
            self.assertEqual(result["answer"], "我在，今天慢一点也没关系。")
            self.assertEqual(result["evidence"], [])
            self.assertEqual(result["memory_intensity"], "none")
            self.assertFalse(result["memory_actually_referenced"])
            self.assertFalse(any("家庭长期记忆" in prompt and "妈妈" in prompt for _, prompt in agent.gamma.contexts))

    def test_memory_turn_hides_images_without_explicit_evidence_request(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "lake.jpg", "image", "/tmp/lake.jpg")
            observation = store.add_observation(asset["id"], {"caption": "湖边散步", "place": "湖边"})
            store.merge_observation_into_event(observation)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn("湖边发生了什么", "memory-conversation")

            self.assertEqual(result["agent_plan"]["mode"], "memory")
            self.assertFalse(result["agent_plan"]["show_images"])
            self.assertEqual(result["image_results"], [])
            self.assertTrue(all("relevance" in item for item in result["evidence"]))

    def test_explicit_evidence_request_limits_images_and_ranks_relevance(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            for index, caption in enumerate(("湖边散步", "湖边野餐", "湖边日落", "客厅聚会"), 1):
                asset = store.create_asset(f"asset_{index}", f"{index}.jpg", "image", f"/tmp/{index}.jpg")
                observation = store.add_observation(asset["id"], {"caption": caption, "place": "湖边" if "湖边" in caption else "客厅"})
                store.merge_observation_into_event(observation)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn("给我看湖边的照片证据", "evidence-conversation")

            self.assertTrue(result["agent_plan"]["show_images"])
            self.assertLessEqual(len(result["image_results"]), 3)
            self.assertEqual(result["evidence_presentation"]["image_limit"], 3)
            scores = [item["relevance"] for item in result["evidence"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_intent_router_distinguishes_clarification(self):
        agent = MemoryAgent(MemoryStore(":memory:"), gamma=FakeGamma())
        self.assertEqual(agent.classify_intent("我说的是妈妈，不是爸爸"), "clarification")


class AgentEvidenceContractTests(unittest.TestCase):
    def _agent_with_observation(self, caption="湖边散步", place="湖边"):
        store = MemoryStore(":memory:")
        asset = store.create_asset("asset_1", "lake.jpg", "image", "/tmp/lake.jpg")
        observation = store.add_observation(asset["id"], {"caption": caption, "place": place})
        store.merge_observation_into_event(observation)
        return MemoryAgent(store, gamma=RefusingGamma())

    def test_memory_turn_declares_anchored_evidence_contract(self):
        result = self._agent_with_observation().answer_turn("湖边发生了什么", "evidence-contract")

        self.assertTrue(result["memory_used"])
        self.assertTrue(result["evidence_required"])
        self.assertEqual(result["evidence_status"], "anchored")
        self.assertTrue(result["evidence"])
        self.assertTrue(result["evidence_layers"]["events"])
        self.assertTrue(result["evidence_presentation"]["required"])

    def test_memory_gap_exposes_query_gap_instead_of_empty_source(self):
        store = MemoryStore(":memory:")
        agent = MemoryAgent(store, gamma=RefusingGamma())

        result = agent.answer_turn("火星生日在哪里", "evidence-gap")

        self.assertTrue(result["memory_used"])
        self.assertTrue(result["evidence_required"])
        self.assertEqual(result["evidence_status"], "gap")
        self.assertFalse(result["evidence"])
        self.assertTrue(result["evidence_layers"]["gaps"])
        self.assertTrue(result["evidence_presentation"]["required"])

    def test_original_evidence_request_is_marked_for_direct_media_output(self):
        result = self._agent_with_observation().answer_turn("请直接给我湖边的原始照片", "original-evidence")

        self.assertTrue(result["original_evidence_requested"])
        self.assertTrue(result["image_results"])
        self.assertTrue(result["evidence_presentation"]["direct_original_evidence"])

    def test_model_plan_cannot_downgrade_memory_or_add_unregistered_tools(self):
        store = MemoryStore(":memory:")
        asset = store.create_asset("asset_1", "lake.jpg", "image", "/tmp/lake.jpg")
        observation = store.add_observation(asset["id"], {"caption": "湖边散步", "place": "湖边"})
        store.merge_observation_into_event(observation)

        result = MemoryAgent(store, gamma=MisleadingPlannerGamma()).answer_turn("请直接给我湖边的原始照片", "plan-contract")

        self.assertEqual(result["agent_plan"]["mode"], "memory")
        self.assertNotIn("drop_database", result["agent_plan"]["tools"])
        self.assertIn("resolve_constraints", result["agent_plan"]["tools"])
        self.assertTrue(result["agent_plan"]["show_images"])

    def test_feedback_response_keeps_target_evidence_visible(self):
        store = MemoryStore(":memory:")
        entity = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
        asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
        observation = store.add_observation(asset["id"], {"caption": "妈妈在湖边", "place": "湖边"})
        store.merge_observation_into_event(observation)
        store.connection.execute(
            "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
            (entity["id"], observation["id"], 1.0, "test"),
        )
        store.connection.commit()

        result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn(
            "纠正这段记忆",
            "feedback-evidence",
            feedback={"target_entity_id": entity["id"], "correction": "这里不是湖边"},
        )

        self.assertTrue(result["feedback"])
        self.assertEqual(result["evidence_status"], "anchored")
        self.assertTrue(result["evidence"])
        self.assertTrue(result["evidence_layers"]["observations"])

    def test_feedback_cannot_target_an_entity_from_another_memory_space(self):
        store = MemoryStore(":memory:")
        entity = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0, scope_id="album_a")

        result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn(
            "纠正这段记忆",
            "feedback-scope",
            feedback={"target_entity_id": entity["id"], "correction": "这不是同一个空间的记忆"},
            scope_id="album_b",
        )

        self.assertFalse(result["feedback"])
        self.assertEqual(result["tool_trace"][-1]["status"], "requires_target")
        self.assertEqual(result["evidence_status"], "gap")

    def test_feedback_cannot_use_a_query_gap_from_another_memory_space(self):
        store = MemoryStore(":memory:")
        gap = store.create_query_gap("相册 A 的哪次活动？", "event", scope_id="album_a")

        result = MemoryAgent(store, gamma=RefusingGamma()).answer_turn(
            "实际是春游",
            "feedback-gap-scope",
            feedback={"query_gap_id": gap["id"], "correction": "春游"},
            scope_id="album_b",
        )

        self.assertFalse(result["feedback"])
        self.assertEqual(store.get_query_gap(gap["id"])["status"], "open")
        self.assertEqual(result["evidence_status"], "gap")

    def test_semantic_entity_group_expands_recall_without_merging_members(self):
        store = MemoryStore(":memory:")
        event_ids = []
        member_ids = []
        for index, label in enumerate(("湖边", "水边"), 1):
            entity = store.create_entity(label, "place", "confirmed", confidence=0.8)
            member_ids.append(entity["id"])
            asset = store.create_asset(f"asset_{index}", f"{index}.jpg", "image", f"/tmp/{index}.jpg")
            observation = store.add_observation(asset["id"], {"caption": f"{label}散步", "place": label})
            event = store.create_event({"id": f"event_{index}", "title": f"{label}散步", "summary": f"在{label}散步", "place": label})
            store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], observation["id"]))
            store.connection.execute(
                "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
                (entity["id"], observation["id"], 0.8, "test"),
            )
            store.connection.commit()
            event_ids.append(event["id"])

        result = MemoryAgent(store, gamma=RefusingGamma()).answer("水边有哪些回忆")

        returned_events = {item["event_id"] for item in result["evidence"] if item["kind"] == "event"}
        self.assertEqual(returned_events, set(event_ids))
        self.assertEqual(result["semantic_groups"][0]["canonical_name"], "滨水区域")
        self.assertEqual(len(store.list_entities()), 2)
        self.assertEqual(set(store.list_semantic_entity_groups()[0]["member_entity_ids"]), set(member_ids))

    def test_person_memory_flow_keeps_evidence_through_follow_up_and_original_media(self):
        store = MemoryStore(":memory:")
        person = store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)
        asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
        observation = store.add_observation(asset["id"], {"caption": "妈妈在湖边散步", "place": "湖边"})
        event = store.merge_observation_into_event(observation)
        store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)
        agent = MemoryAgent(store, gamma=RefusingGamma())

        introduction = agent.answer_turn("介绍一下妈妈", "person-flow")
        follow_up = agent.answer_turn("她后来在哪里？", "person-flow")
        original = agent.answer_turn("请直接给我那次的原始照片", "person-flow")

        for result in (introduction, follow_up, original):
            self.assertTrue(result["memory_used"])
            self.assertTrue(result["evidence_required"])
            self.assertEqual(result["evidence_status"], "anchored")
            self.assertTrue(result["evidence"])
        self.assertEqual(follow_up["dialogue_plan"]["mode"], "contextual_follow_up")
        self.assertTrue(original["original_evidence_requested"])
        self.assertTrue(original["image_results"])


class AgentOrchestrationTests(unittest.TestCase):
    def test_typed_plan_validator_cannot_downgrade_memory_or_execute_unknown_tools(self):
        validated = validate_turn_plan(
            {"mode": "chat", "tools": ["drop_database", "find_events"], "show_images": False},
            {"mode": "memory", "tools": ["resolve_constraints"], "show_images": True, "reason": "memory request"},
        )

        self.assertEqual(validated.mode, "memory")
        self.assertEqual(validated.tools, ("resolve_constraints", "find_events"))
        self.assertTrue(validated.show_images)
        self.assertNotIn("drop_database", validated.tools)

    def test_pydantic_ai_planner_degrades_to_disabled_without_framework_model(self):
        planner = PydanticAIPlanner(model=None)

        self.assertFalse(planner.available)
        self.assertIsNone(planner.plan("只返回行动计划"))

    @unittest.skipIf(TestModel is None, "pydantic-ai-slim is optional in the local test environment")
    def test_pydantic_ai_planner_returns_structured_plan_when_enabled(self):
        planner = PydanticAIPlanner(model=TestModel())

        self.assertTrue(planner.available)
        self.assertEqual(planner.plan("只返回行动计划")["mode"], "chat")


if __name__ == "__main__":
    unittest.main()
