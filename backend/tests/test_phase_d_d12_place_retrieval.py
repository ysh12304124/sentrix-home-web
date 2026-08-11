"""Phase D — D12 地点检索修复测试（geocode 行政区匹配 + 别名 + 无检索否认 guard）。"""
import os
import tempfile
import unittest

from backend.db import MemoryStore
from backend.geocoding import place_text_matches
from backend.agent_runtime import tools as runtime_tools
from backend.agent_runtime.final_guard import FinalGuard

GEO_QHD = {"source": "tianditu", "label": "河北省秦皇岛市昌黎县",
           "province": "河北省", "city": "秦皇岛市", "district": "昌黎县",
           "country": "CN", "confidence": 0.9}
GEO_SJZ = {"source": "tianditu", "label": "河北省石家庄市桥西区",
           "province": "河北省", "city": "石家庄市", "district": "桥西区",
           "country": "CN", "confidence": 0.9}
GEO_CM = {"source": "geonames", "label": "Chiang Mai, Chiang Mai, TH",
          "name": "Chiang Mai", "city": "Chiang Mai", "country": "TH"}


class PlaceTextMatchTest(unittest.TestCase):
    def test_admin_part_inside_constraint(self):
        self.assertTrue(place_text_matches("秦皇岛如是海度假村", GEO_QHD))
        self.assertTrue(place_text_matches("上海普陀区江宁路",
                                           {"city": "上海市", "district": "普陀区"}))
        self.assertTrue(place_text_matches("河北省邯郸市永年区中华大街附近公园",
                                           {"city": "邯郸市", "district": "永年区"}))
        self.assertTrue(place_text_matches("秦皇岛", GEO_QHD))

    def test_label_substring(self):
        self.assertTrue(place_text_matches("昌黎", GEO_QHD))

    def test_alias_international(self):
        self.assertTrue(place_text_matches("清迈", GEO_CM))
        self.assertTrue(place_text_matches("泰国清迈夜间动物园", GEO_CM))

    def test_mismatch(self):
        self.assertFalse(place_text_matches("三亚", GEO_QHD))
        self.assertFalse(place_text_matches("宜昌", GEO_CM))


class PlaceRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.dir, "test.db"))
        runtime_tools._RUNTIME.clear()
        runtime_tools.bind_runtime(self.store)
        runtime_tools.register_tools()
        self.store.create_asset(
            "asset_a1", "IMG_8653.JPG", "image", "/p/8653.jpg",
            metadata={"captured_at": "2019-07-22 15:38:08",
                      "captured_location": "39.620133,119.296806",
                      "reverse_geocode": GEO_QHD}, scope_id="album3-v2")
        self.store.create_asset(
            "asset_a2", "IMG_8654.JPG", "image", "/p/8654.jpg",
            metadata={"captured_at": "2019-07-22 15:39:18",
                      "captured_location": "39.620139,119.296753",
                      "reverse_geocode": GEO_QHD}, scope_id="album3-v2")
        self.store.create_asset(
            "asset_a3", "IMG_0182.JPG", "image", "/p/0182.jpg",
            metadata={"captured_at": "2020-03-17 18:12:35",
                      "captured_location": "36.762436,114.499169",
                      "reverse_geocode": GEO_SJZ}, scope_id="album3-v2")
        self.store.add_observation(
            "asset_a1", {"caption": "三个孩子在巨大的沙雕前合影",
                         "activity": "拍照", "place": "户外沙滩或沙雕公园"},
            scope_id="album3-v2")
        self.store.add_observation(
            "asset_a2", {"caption": "两个孩子在沙雕前合影",
                         "activity": "拍照", "place": "户外沙滩或沙雕公园"},
            scope_id="album3-v2")
        self.store.add_observation(
            "asset_a3", {"caption": "厨房制作兔子造型面点",
                         "activity": "制作", "place": "厨房操作台"},
            scope_id="album3-v2")

    def test_search_memories_place_filter_no_zero_hit(self):
        ctx = {"scope_id": "album3-v2", "viewer_id": "owner", "task_state": {}}
        out = runtime_tools._search_memories(
            {"query": "沙雕合影", "filters": {"time": "2019年7月", "place": "秦皇岛如是海度假村"}},
            context=ctx)
        self.assertEqual(out["total"], 2, out)
        self.assertNotEqual(out["query_satisfaction"], "no_match")

    def test_search_memories_international_alias(self):
        ctx = {"scope_id": "album3-v2", "viewer_id": "owner", "task_state": {}}
        self.store.create_asset(
            "asset_a4", "2018-04-01 210440.jpg", "image", "/p/210440.jpg",
            metadata={"captured_at": "2018-04-01 20:04:41",
                      "captured_location": "18.742511,98.916611",
                      "reverse_geocode": GEO_CM}, scope_id="album3-v2")
        self.store.add_observation(
            "asset_a4", {"caption": "夜晚灯光下的户外广场表演",
                         "activity": "街头表演", "place": "户外广场"},
            scope_id="album3-v2")
        out = runtime_tools._search_memories(
            {"query": "夜间部落表演", "filters": {"time": "2018年4月", "place": "清迈"}},
            context=ctx)
        self.assertEqual(out["total"], 1, out)

    def test_query_memory_facts_place_count(self):
        ctx = {"scope_id": "album3-v2", "viewer_id": "owner", "task_state": {}}
        out = runtime_tools._query_memory_facts(
            {"operation": "count", "filters": {"time": "2019年7月", "place": "秦皇岛如是海度假村"}},
            context=ctx)
        self.assertEqual(out["total"], 2, out)
    def test_search_preview_includes_place(self):
        ctx = {"scope_id": "album3-v2", "viewer_id": "owner", "task_state": {}}
        out = runtime_tools._search_memories(
            {"query": "沙雕合影", "filters": {"time": "2019年7月", "place": "秦皇岛"}},
            context=ctx)
        self.assertEqual(out["total"], 2, out)
        places = {p.get("place") for p in (out.get("preview") or [])}
        self.assertTrue(places, "preview 应包含 place 字段")
        self.assertTrue(any("秦皇岛" in str(pl) for pl in places), places)


class DenialWithoutSearchTest(unittest.TestCase):
    def test_denial_without_any_tool_is_flagged(self):
        result = FinalGuard().check("抱歉，我没有找到相关照片记录。", task_state={"tool_results": []})
        self.assertIn("denial_without_search", [i.code for i in result.issues])

    def test_denial_after_search_is_allowed(self):
        result = FinalGuard().check(
            "抱歉，我没有找到相关照片记录。",
            task_state={"tool_results": [{"tool": "search_memories", "total": 0}]})
        self.assertNotIn("denial_without_search", [i.code for i in result.issues])

    def test_travel_negation_not_flagged(self):
        result = FinalGuard().check("我没有去过北京。", task_state={"tool_results": []})
        self.assertNotIn("denial_without_search", [i.code for i in result.issues])


if __name__ == "__main__":
    unittest.main()


class RecoveryResilienceTest(unittest.TestCase):
    """D12：恢复阶段模型调用失败时，必须保留已产出的 final 回答。"""

    def test_model_error_after_final_keeps_answer(self):
        from backend.agent_runtime.runtime import AgentRuntime
        calls = {"n": 0}

        def chat_fn(messages):
            calls["n"] += 1
            if calls["n"] == 1:
                return ('{"action":"final","answer":"活动是在秦皇岛如是海度假村进行的。",'
                        '"evidence_refs":["tool_call_1"]}')
            raise RuntimeError("simulated vLLM 400 on recovery")

        rt = AgentRuntime(chat_fn=chat_fn, profile_name="tool_loop")
        turn = rt.run("2019年7月22日明明和乐乐在主题沙雕前合影的活动是在哪里进行的？")
        self.assertIn("活动是在秦皇岛如是海度假村进行的", turn.final_answer)
        self.assertIn(turn.status, {"complete", "partial"})
        self.assertNotEqual(turn.status, "error")


if __name__ == "__main__":
    unittest.main()
