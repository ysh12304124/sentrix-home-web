#!/usr/bin/env python3
"""Create a provenance-first HTML report from the isolated two-video run."""
from __future__ import annotations

import html
import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
SOURCE = Path("/home/sscy/evaluations/sentrix-worldmm-hippovlog-20260928/services/worldmm_hippovlog/results/20260928-163006/svd_esvd")
DATASET = Path("/ssd/sscy/datasets/HippoVlog-svd-worldmm-20260928")
VIDEOS = ("6Z_qEtbmK34", "Ei7hTKr8Ins")
REVIEWED_RETRIEVAL_MISSES = {
    "6Z_qEtbmK34-684": "原帧约 06:45 可见化妆瓶与刷子；06:40-06:50 记忆明确写 setting spray bottle，实际 4 次视觉 Top-K 均未返回该时间窗。",
    "6Z_qEtbmK34-935": "原帧及事件主帧约 09:35 可见红色唇膏；09:30-09:40 记忆明确写 red lipstick，实际情节检索返回 06:40 附近片段而非目标记忆。",
    "Ei7hTKr8Ins-607": "原帧约 20:25 可见巧克力蛋糕、奶油及 Sachertorte 字幕；20:20-20:30 记忆写 Sachertorte，实际语义 Top-K 未返回该证据。事件主帧位于更早时刻，但 WorldMM 记忆通过原帧回填仍保有证据。",
}
STOP = set("a an the of in on at to for from with and or is are was were be being been what which when where who why how video scene character main person people they their there this that can seen shown shown background during does did do you it as by into behind about specific visual element prominently detail noticeable".split())


def load(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def words(value: str):
    return {word for word in re.findall(r"[a-z0-9]+", str(value).lower()) if len(word) > 2 and word not in STOP}


def seconds(clock: str):
    digits = str(clock).zfill(8)
    return int(digits[:2]) * 3600 + int(digits[2:4]) * 60 + int(digits[4:6]) + int(digits[6:]) / 100


def clock(value):
    if value is None:
        return "—"
    value = float(value)
    return f"{int(value // 60):02d}:{value % 60:05.2f}"


def esc(value):
    return html.escape(str(value if value is not None else "—"))


def fmt(value, digits=1):
    return "未记录" if value is None else f"{value:,.{digits}f}"


def span_from_visual_key(key: str):
    matches = re.findall(r"(\d{2}):(\d{2}):(\d{2})", key)
    if len(matches) < 2:
        return None
    return tuple(sum(int(part) * mult for part, mult in zip(item, (3600, 60, 1))) for item in matches[:2])


def captions_for(video_id):
    return load(SOURCE / "captions" / video_id / "10sec.json", [])


def trace_at_candidate(video_cache, video_id, caption):
    if caption is None:
        return None
    start, end = seconds(caption["start_time"]), seconds(caption["end_time"])
    locator = (start + end) / 2
    data = video_cache[video_id]
    near = min(data["candidates"], key=lambda item: abs(item["timestamp_sec"] - locator))
    sample = near["sample_index"]
    fold = data["analysis"]["second_pass_fold_assignments"][sample]
    event = next((item for item in data["analysis"]["metrics"]["event_generation"]["events"]
                  if item["start_index"] <= sample < item["end_index"]), None)
    observations = data["observations"].get(sample, {})
    frames = data["manifest"]["keyframes"]
    event_frames = [item for item in frames if item["raw"].get("yolo_event_id") == (event or {}).get("event_id")]
    selected = event_frames[0] if event_frames else min(frames, key=lambda item: abs(item["timestamp_sec"] - locator))
    return {
        "candidate_timestamp_sec": locator,
        "vsd_yolo_frame_index": near["frame_index"],
        "vsd_yolo_timestamp_sec": near["timestamp_sec"],
        "yolo_labels": sorted({item["label"] for item in observations.get("detections") or []}),
        "evsd_status": fold["status"], "evsd_group_id": fold.get("group_id"),
        "event_id": (event or {}).get("event_id"),
        "event_start_sec": (event or {}).get("start_sec"),
        "event_end_sec": (event or {}).get("end_sec"),
        "selected_frame_index": selected["frame_index"],
        "selected_timestamp_sec": selected["timestamp_sec"],
    }


def locate(question, captions):
    answer_text = question["options"][question["correct_answer"]]
    answer_tokens = words(answer_text)
    anchor_tokens = words(question["question_text"])
    scored = []
    for index, row in enumerate(captions):
        caption_tokens = words(row.get("text", ""))
        a = len(answer_tokens & caption_tokens)
        q = len(anchor_tokens & caption_tokens)
        score = 4 * a / max(1, len(answer_tokens)) + q / max(1, len(anchor_tokens))
        exact = bool(answer_tokens) and answer_tokens <= caption_tokens and q >= min(2, len(anchor_tokens))
        scored.append((score, exact, index, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    certified = next((item for item in scored if item[1]), None)
    chosen = certified or (scored[0] if scored else None)
    return chosen, certified is not None, scored[:3]


def retrieval_support(retrieval_rows, answer_tokens, time_window):
    text_hit = False
    visual_time_hit = False
    for item in retrieval_rows:
        if item["memory_type"] in {"episodic", "semantic"} and answer_tokens and answer_tokens <= words(item.get("result", "")):
            text_hit = True
        if item["memory_type"] == "visual" and time_window:
            for clip in item.get("result") or []:
                span = span_from_visual_key(clip.get("clip_key", ""))
                if span and max(span[0], time_window[0]) < min(span[1], time_window[1]):
                    visual_time_hit = True
    return text_hit, visual_time_hit


def retrieval_count(row):
    result = row.get("result") or ""
    if row["memory_type"] == "visual":
        return len(result)
    if row["memory_type"] == "semantic":
        return len([line for line in result.splitlines() if line.strip()])
    return len(re.findall(r"\[DAY\d+ \d\d:\d\d:\d\d - DAY\d+ \d\d:\d\d:\d\d\]", result))


def export_frame(video_id, timestamp, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(max(0, timestamp)),
        "-i", str(DATASET / "videos" / f"{video_id}.mp4"), "-frames:v", "1",
        "-q:v", "5", "-y", str(target),
    ], check=True)


def case_story(case, question, video_cache, retrieval_rows):
    video_id = case["video_id"]
    caption = case["caption"]
    if caption:
        start, end = seconds(caption["start_time"]), seconds(caption["end_time"])
        locator = (start + end) / 2
    else:
        locator, start, end = 0.0, 0.0, 10.0
    video_cache = video_cache[video_id]
    candidates = video_cache["candidates"]
    near_candidate = min(candidates, key=lambda item: abs(item["timestamp_sec"] - locator))
    obs = video_cache["observations"][near_candidate["sample_index"]]
    analysis = video_cache["analysis"]
    sample = near_candidate["sample_index"]
    fold = analysis["second_pass_fold_assignments"][sample]
    event = next((item for item in analysis["metrics"]["event_generation"]["events"]
                  if item["start_index"] <= sample < item["end_index"]), None)
    keyframes = video_cache["manifest"]["keyframes"]
    event_frames = [item for item in keyframes if item["raw"].get("yolo_event_id") == (event or {}).get("event_id")]
    final_frame = event_frames[0] if event_frames else min(keyframes, key=lambda item: abs(item["timestamp_sec"] - locator))
    evidence = OUT / "cases" / case["ID"]
    raw = evidence / "raw.jpg"
    vsd = evidence / "vsd_candidate.jpg"
    selected = evidence / "evsd_event_keyframe.jpg"
    export_frame(video_id, locator, raw)
    export_frame(video_id, near_candidate["timestamp_sec"], vsd)
    shutil.copyfile(final_frame["path"], selected)
    detections = obs.get("detections") or []
    det_html = "".join(f"<li>{esc(d['label'])} · conf {d['confidence']:.3f} · bbox {esc(d['bbox'])}</li>" for d in detections) or "<li>无检测框</li>"
    retrieved_summary = []
    for row in retrieval_rows[:5]:
        content = row.get("result")
        if isinstance(content, list):
            content = ", ".join(x.get("clip_key", "") for x in content)
        retrieved_summary.append(f"{row['memory_type']}: {str(content)[:180]}")
    return f"""
    <article class="case"><h3>{esc(case['ID'])} · {esc(case['root_cause'])}</h3>
      <p><b>问题：</b>{esc(question['question_text'])}<br><b>正确选项：</b>{esc(question['correct_answer'])} {esc(question['options'][question['correct_answer']])}<br><b>输出：</b>{esc(case['response'])}</p>
      <p class="muted">定位时间 {clock(locator)}：{('答案词及问题锚点同时出现在 caption' if case['gold_text_match'] else '仅为自动相似文本候选，非人工标注的金标准时间戳')}。</p>
      <div class="frames"><figure><img src="cases/{esc(case['ID'])}/raw.jpg"><figcaption>RAW · {clock(locator)} · 原视频解码帧</figcaption></figure>
      <figure><img src="cases/{esc(case['ID'])}/vsd_candidate.jpg"><figcaption>VSD → YOLO 输入 · 帧 {near_candidate['frame_index']} · {clock(near_candidate['timestamp_sec'])}</figcaption></figure>
      <figure><img src="cases/{esc(case['ID'])}/evsd_event_keyframe.jpg"><figcaption>EVSD / Event 输出 · 帧 {final_frame['frame_index']} · {clock(final_frame['timestamp_sec'])}</figcaption></figure></div>
      <div class="flow"><div><b>YOLO</b><ul>{det_html}</ul></div>
      <div><b>EVSD</b><p>{esc(fold['status'])} · {esc(fold['group_id'])}<br>残差 {fmt(fold.get('reconstruction_residual'),3)}</p></div>
      <div><b>Event</b><p>{esc((event or {}).get('event_id'))}<br>{clock((event or {}).get('start_sec'))}—{clock((event or {}).get('end_sec'))}<br>{esc((event or {}).get('boundary_reason'))}</p></div>
      <div><b>Memory</b><p>{esc((caption or {}).get('text','未定位'))[:380]}</p></div></div>
      <p><b>Retrieval Top-K 轨迹：</b>{esc(' | '.join(retrieved_summary) or '未发起检索')[:650]}</p>
      <p><b>最终归因：</b>{esc(case['root_cause'])} · {esc(case['reason'])}</p>
    </article>"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage = load(OUT / "stage_summary.json", {})
    questions = load(OUT / "questions.json", [])
    answers = load(OUT / "answers.json", None) or jsonl(OUT / "answers.partial.jsonl")
    resources = jsonl(OUT / "resources.jsonl")
    metrics = jsonl(OUT / "stage_metrics.jsonl")
    retrieval = defaultdict(list)
    retrieval_counts = defaultdict(lambda: {"calls": 0, "items": 0, "seconds": 0.0})
    normalized_retrieval = []
    for row in jsonl(OUT / "retrieval_trace.jsonl"):
        retrieval[row["question_id"]].append(row)
        item_count = retrieval_count(row)
        bucket = retrieval_counts[(row["video_id"], row["memory_type"])]
        bucket["calls"] += 1
        bucket["items"] += item_count
        bucket["seconds"] += row.get("seconds", 0)
        normalized_retrieval.append({**row, "input_count": 1, "output_count": item_count,
                                     "decision": "KEEP" if item_count else "DROP",
                                     "count_method": "parsed_result_items"})
    with (OUT / "retrieval_counted.jsonl").open("w", encoding="utf-8") as handle:
        for row in normalized_retrieval:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    conflicts = jsonl(OUT / "evsd_semantic_conflicts.jsonl")
    by_id = {item["ID"]: item for item in questions}
    video_cache = {}
    for video_id in VIDEOS:
        root = SOURCE / "keyframes" / video_id
        video_cache[video_id] = {
            "analysis": load(root / "analysis.json"),
            "candidates": load(root / "yolo_scan_candidates.json", []),
            "observations": {item["sample_index"]: item for item in load(root / "yolo_scan.json", {}).get("samples", [])},
            "manifest": load(root / "manifest.json"),
        }
    cases = []
    for answer in answers:
        q = by_id[answer["ID"]]
        selected, exact, top = locate(q, captions_for(q["video_id"]))
        caption = selected[3] if selected else None
        interval = (seconds(caption["start_time"]), seconds(caption["end_time"])) if caption and exact else None
        answer_tokens = words(q["options"][q["correct_answer"]])
        text_hit, visual_time_hit = retrieval_support(retrieval[answer["ID"]], answer_tokens, interval)
        failed = not bool(answer["evaluate"])
        if not failed:
            cause, reason = "PASS", "官方四选一评分正确"
        elif q["category"] == "audio":
            cause, reason = "UNDETERMINED_AUDIO", "仅有视觉链路轨迹，无法用它确认声音证据首次丢失位置"
        elif exact and text_hit:
            cause = "QA_GENERATION_FAILURE" if answer["response"] == "Unable to generate answer" else "REASONING_ERROR"
            reason = "正确选项的文字线索出现在 10 秒记忆和实际检索文本中，最终输出仍失败；词面证据需人工复核"
        elif exact and not visual_time_hit:
            cause, reason = "RETRIEVAL_MISS", "正确选项与问题锚点同现于记忆文本，但未进入任何实际检索文本或同时间窗视觉 Top-K；词面证据需人工复核"
        elif exact:
            cause, reason = "UNDETERMINED_RETRIEVED_VISUAL", "相关时间窗进入视觉 Top-K，但未验证返回图像是否直接包含答案"
        else:
            cause, reason = "UNDETERMINED_UPSTREAM", "QA 未提供金标准证据时间戳；现有自动文本匹配不足以证明 VSD/YOLO/EVSD/Event/Memory 哪一级首先丢失"
        manual_review = REVIEWED_RETRIEVAL_MISSES.get(answer["ID"]) if cause == "RETRIEVAL_MISS" else None
        if manual_review:
            reason = manual_review
        cases.append({
            "ID": answer["ID"], "video_id": q["video_id"], "category": q["category"],
            "question": q["question_text"], "answer": q["correct_answer"],
            "response": answer["response"], "correct": bool(answer["evaluate"]),
            "root_cause": cause, "reason": reason, "gold_text_match": exact,
            "caption": caption, "candidate_caption_index": selected[2] if selected else None,
            "candidate_score": round(selected[0], 3) if selected else None,
            "retrieved_gold_text": text_hit, "retrieved_visual_time": visual_time_hit,
            "retrieval_rounds": len(retrieval[answer["ID"]]), "seconds": answer.get("seconds"),
            "confidence": "manual_raw_memory_retrieval_review" if manual_review else
                          ("lexical_candidate_requires_review" if exact else "unresolved"),
            "manual_review": manual_review,
            "trace_candidate": trace_at_candidate(video_cache, q["video_id"], caption),
        })
    (OUT / "case_analysis.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    resource_by_question = defaultdict(list)
    for sample in resources:
        if sample.get("question_id") and sample.get("stage") == "retrieval_qa":
            resource_by_question[sample["question_id"]].append(sample)
    with (OUT / "qa_stage_trace.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            samples = resource_by_question[case["ID"]]
            row = {
                "stage": "QA", "video_id": case["video_id"], "question_id": case["ID"],
                "frame_id": (f"{case['video_id']}:{case['trace_candidate']['vsd_yolo_frame_index']}"
                             if case["trace_candidate"] else None),
                "timestamp_sec": (case["trace_candidate"] or {}).get("candidate_timestamp_sec"),
                "candidate_is_gold": False, "input_count": 1, "output_count": 1,
                "decision": "PASS" if case["correct"] else "FAIL",
                "reason": "official_multiple_choice_eval", "seconds": case["seconds"],
                "gpu_allocated_peak_mib": max((r.get("gpu_allocated_mib") or 0 for r in samples), default=None),
                "rss_peak_mib": max((r.get("rss_mib") or 0 for r in samples), default=None),
                "cpu_peak_percent": max((r.get("cpu_percent") or 0 for r in samples), default=None),
                "provenance": {"retrieval_calls": len(retrieval[case["ID"]]),
                               "candidate_memory_index": case["candidate_caption_index"]},
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    anomalies = []
    for conflict in conflicts:
        anomalies.append({
            "kind": "EVSD_SEMANTIC_CONFLICT_CANDIDATE", "video_id": conflict["video_id"],
            "fold_group_id": conflict["fold_group_id"],
            "frames": [{"frame_index": obs["frame_index"], "timestamp_sec": obs["timestamp_sec"],
                        "labels": obs["labels"], "fold_status": obs["fold_status"]}
                       for obs in conflict["observations"]],
            "review_status": conflict["review_status"],
        })
    for case in cases:
        if not case["correct"]:
            anomalies.append({
                "kind": "QA_FAILURE", "video_id": case["video_id"], "qa_id": case["ID"],
                "candidate_frame": case["trace_candidate"], "candidate_is_gold": False,
                "root_cause_candidate": case["root_cause"], "reason": case["reason"],
            })
    for error in jsonl(OUT / "qa_errors.jsonl"):
        anomalies.append({"kind": "QA_RUNTIME_ERROR", **error})
    (OUT / "anomalies.json").write_text(json.dumps(anomalies, ensure_ascii=False, indent=2), encoding="utf-8")
    causes = Counter(item["root_cause"] for item in cases if not item["correct"])
    per_video_qa = {
        video_id: {
            "answered": sum(item["video_id"] == video_id for item in cases),
            "correct": sum(item["video_id"] == video_id and item["correct"] for item in cases),
            "failed": sum(item["video_id"] == video_id and not item["correct"] for item in cases),
        }
        for video_id in VIDEOS
    }
    conflicts_by_video = Counter(item["video_id"] for item in conflicts)
    resource_peak = defaultdict(lambda: {"gpu_allocated_mib": 0, "gpu_reserved_mib": 0, "rss_mib": 0, "cpu_percent": 0})
    for sample in resources:
        key = (sample.get("video_id") or "global", sample.get("stage") or "unknown")
        for field in resource_peak[key]:
            resource_peak[key][field] = max(resource_peak[key][field], float(sample.get(field) or 0))
    report_data = {
        "status": load(OUT / "status.json", {}), "answers": len(answers), "correct": sum(c["correct"] for c in cases),
        "root_causes": dict(causes), "per_video_qa": per_video_qa, "stage_summary": stage,
        "retrieval_summary": {f"{video}/{kind}": value for (video, kind), value in retrieval_counts.items()},
        "resource_peaks": {f"{video}/{phase}": value for (video, phase), value in resource_peak.items()},
        "semantic_conflict_candidates": dict(conflicts_by_video),
    }
    (OUT / "report_data.json").write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Prefer cases whose downstream failure is supported by a concrete
    # caption/retrieval trace. Include unresolved examples to expose limits.
    priority = {"RETRIEVAL_MISS": 0, "REASONING_ERROR": 1, "QA_GENERATION_FAILURE": 2,
                "UNDETERMINED_RETRIEVED_VISUAL": 3, "UNDETERMINED_UPSTREAM": 4, "UNDETERMINED_AUDIO": 5}
    failures = [case for case in cases if not case["correct"]]
    ranked = sorted(failures, key=lambda case: (priority.get(case["root_cause"], 9),
                                                -float(case.get("candidate_score") or 0)))
    typical = []
    for case in ranked:
        if len(typical) >= 8:
            break
        if case["category"] == "audio" and len(typical) < 6:
            continue
        if sum(item["video_id"] == case["video_id"] for item in typical) >= 5:
            continue
        typical.append(case)
    story_html = ""
    for case in typical:
        story_html += case_story(case, by_id[case["ID"]], video_cache, retrieval[case["ID"]])

    rows = []
    for case in cases:
        if case["correct"]:
            state = "PASS"
        else:
            state = "FAIL"
        memory_cell = "词面证据" if case["gold_text_match"] else "未核验"
        retrieval_cell = "文本命中" if case["retrieved_gold_text"] else ("视觉时间命中" if case["retrieved_visual_time"] else "无已核验命中")
        trace = case["trace_candidate"] or {}
        vsd_cell = f"候选帧 {trace.get('vsd_yolo_frame_index', '—')}<br>{clock(trace.get('vsd_yolo_timestamp_sec'))}"
        yolo_cell = ", ".join(trace.get("yolo_labels") or []) or "无检测框"
        evsd_cell = f"{trace.get('evsd_status', '—')}<br>{trace.get('evsd_group_id', '—')}"
        event_cell = f"{trace.get('event_id', '—')}<br>主帧 {trace.get('selected_frame_index', '—')}"
        rows.append(f"<tr><td>{esc(case['video_id'])}</td><td>{esc(case['ID'])}<br><small>{esc(case['category'])}</small></td>"
                    f"<td>{vsd_cell}</td><td>{esc(yolo_cell)}</td><td>{esc(evsd_cell)}</td>"
                    f"<td>{esc(event_cell)}</td><td>{memory_cell}</td><td>{retrieval_cell}</td>"
                    f"<td class='{state.lower()}'>{state}</td><td>{esc(case['root_cause'])}</td></tr>")
    table_rows = "\n".join(rows)
    stage_rows = ""
    preservation_rows = ""
    for video_id, data in stage.items():
        timing = data["timing_seconds"]
        index = sum(item.get("seconds", 0) for item in metrics if item.get("video_id") == video_id and item["stage"] == "memory_index")
        qa_seconds = sum(case.get("seconds") or 0 for case in cases if case["video_id"] == video_id)
        peak = resource_peak.get((video_id, "retrieval_qa"), {})
        stage_rows += (f"<tr><td>{esc(video_id)}</td><td>{data['raw_frame_count']:,}</td><td>{data['sampled_frame_count']:,}</td>"
                       f"<td>{data['vsd_candidate_count']:,}</td><td>{data['yolo_detection_count']:,}</td>"
                       f"<td>{data['evsd_fold_status'].get('folded',0):,}</td><td>{data['event_count']}</td>"
                       f"<td>{data['final_keyframe_count']}</td><td>{data['episodic_10sec_count']}</td>"
                       f"<td>{data['episodic_midpoint_fallback_count']}</td><td>{data['semantic_triple_count']}</td>"
                       f"<td>{fmt(timing['vsd_yolo_evsd_event_total'])} s</td>"
                       f"<td>{fmt(timing['first_pass_fold'])} s</td><td>{fmt(timing['second_pass_evsd_fold'])} s</td>"
                       f"<td>{fmt(timing['caption_and_multiscale'])} s</td>"
                       f"<td>{fmt(index)} s</td><td>{fmt(qa_seconds)} s</td><td>{fmt(peak.get('gpu_allocated_mib'))} MiB</td></tr>")
        sampled = data["sampled_frame_count"]
        vsd = data["vsd_candidate_count"]
        memory = data["episodic_10sec_count"]
        preservation_rows += (
            f"<tr><td>{esc(video_id)}</td><td>{vsd:,}/{sampled:,} ({vsd/sampled:.1%})</td>"
            f"<td>{data['yolo_observation_count']:,}/{vsd:,} ({data['yolo_observation_count']/vsd:.1%})</td>"
            f"<td>{data['evsd_fold_status'].get('folded',0):,} folded · {data['evsd_fold_status'].get('outlier',0):,} outlier</td>"
            f"<td>{data['event_count']} event → {data['final_keyframe_count']} 主帧；合并 {data['event_merged_count']}</td>"
            f"<td>{memory} 片段；{data['episodic_midpoint_fallback_count']} 原帧中点回填 "
            f"({data['episodic_midpoint_fallback_count']/memory:.1%})</td></tr>"
        )
    resource_rows = "".join(
        f"<tr><td>{esc(video)}</td><td>{esc(phase)}</td><td>{fmt(peaks['gpu_allocated_mib'])}</td>"
        f"<td>{fmt(peaks['gpu_reserved_mib'])}</td><td>{fmt(peaks['rss_mib'])}</td>"
        f"<td>{fmt(peaks['cpu_percent'])}</td></tr>"
        for (video, phase), peaks in sorted(resource_peak.items())
    )
    retrieval_rows = "".join(
        f"<tr><td>{esc(video)}</td><td>{esc(kind)}</td><td>{values['calls']}</td>"
        f"<td>{values['items']}</td><td>{fmt(values['seconds'])} s</td></tr>"
        for (video, kind), values in sorted(retrieval_counts.items())
    )
    cause_rows = "".join(f"<li>{esc(cause)}：{count}</li>" for cause, count in causes.most_common())
    qa_rows = "".join(
        f"<tr><td>{esc(video_id)}</td><td>{result['answered']}/40</td><td>{result['correct']}</td>"
        f"<td>{result['failed']}</td><td>{result['correct']/result['answered']:.1%}</td></tr>"
        for video_id, result in per_video_qa.items() if result["answered"]
    )
    status = report_data["status"].get("status", "unknown")
    content = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>HippoVlog HINT-SD 式故障归因</title>
<style>body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Noto Sans CJK SC',sans-serif;color:#172333;background:#f5f7fa;margin:0}}main{{max-width:1320px;margin:auto;padding:30px}}h1,h2,h3{{line-height:1.25}}h1{{font-size:28px}}section,.case{{background:white;border:1px solid #dfe5eb;border-radius:10px;padding:18px;margin:18px 0}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #dce2e8;padding:7px;vertical-align:top}}th{{background:#213f60;color:white;position:sticky;top:0}}.scroll{{overflow:auto;max-height:720px}}.pass{{color:#147d49}}.fail{{color:#be3131;font-weight:700}}.muted,small{{color:#607080}}.frames{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}figure{{margin:0}}img{{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}}figcaption{{font-size:12px;color:#536273}}.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.flow>div{{background:#f1f5f8;padding:10px;border-radius:8px;overflow-wrap:anywhere}}p{{overflow-wrap:anywhere}}code{{background:#eff2f5;padding:2px 4px}}@media(max-width:800px){{.frames,.flow{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>HippoVlog · HINT-SD 式长视频记忆故障归因</h1><p>服务器 200 · 独立实验目录 · 状态 <b>{esc(status)}</b> · QA {len(answers)}/80，正确 {report_data['correct']}/{len(answers) or 0}</p>
<section><h2>两段视频 QA 结果</h2><table><tr><th>Video</th><th>已答／目标</th><th>正确</th><th>失败</th><th>准确率</th></tr>{qa_rows}</table></section>
<section><h2>实验边界与环境</h2><p>WorldMM 服务源：<code>/home/sscy/evaluations/sentrix-worldmm-hippovlog-20260928/services/worldmm_hippovlog</code>；隔离输出：<code>/home/sscy/evaluations/hint-sd-hippovlog-20261009/results</code>。Python 3.10.20、Torch 2.6.0+cu124、CUDA 12.4、RTX 3090 24GB；Qwen3-VL-4B、Qwen3-Embedding-4B、VLM2Vec-V2.0/Qwen2-VL-2B 与 yolo11n.pt 均复用已有权重。GPU 放置：Qwen3-VL 与视觉检索在 GPU，文字 embedding 在 CPU（同权重，同 Top-K，避免首次 24GB OOM）。启动：<code>.venv/bin/python qa_trace.py --output results</code>。</p>
<p>两条视频：6Z_qEtbmK34，1530.172 秒、1920×1080、36,686 帧；Ei7hTKr8Ins，1548.771 秒、1920×1080、46,415 帧。源视频只读。VSD→YOLO→eSVD→Event→Memory 采用原运行 20260928-163006 已完成产物；本实验新跑 WorldMM 检索与 QA。不是重新构建上游阶段。首次全 GPU QA 因 OOM 作废并保存在 <code>results/attempt1_gpu_oom</code>。</p>
<p><b>归因限制：</b>HippoVlog 原始 QA 没有金标准证据时间戳；词面匹配只能提供可复核候选，不能凭“答案词没出现在 caption”宣称 VSD/YOLO/EVSD 漏检。3 例检索漏召回由原帧、记忆文本与实际 Top-K 人工抽查支持；其他上游未核实样本标记 UNDETERMINED。YOLO 不支持的开放类别不能算 YOLO_MISS。</p></section>
<section><h2>阶段数量、耗时与资源</h2><div class="scroll"><table><tr><th>Video</th><th>RAW</th><th>10fps 采样</th><th>VSD→YOLO</th><th>YOLO boxes</th><th>eSVD 折叠帧</th><th>Event</th><th>主帧</th><th>10s Memory</th><th>原帧回填窗</th><th>语义三元组</th><th>视频管线合计</th><th>VSD 首次折叠</th><th>eSVD 二次折叠</th><th>Caption/多尺度</th><th>检索索引</th><th>QA</th><th>QA Torch 显存峰值</th></tr>{stage_rows}</table></div>
<p class="muted">视频管线为 VSD+YOLO+eSVD+Event 合计，原运行未分别持久化各子阶段耗时及 GPU/RAM 峰值；首轮全 GPU失败数据不混入有效结果。Torch 显存峰值仅指本次 QA 进程的 allocated，不是全卡使用量。Memory 构建的历史单视频耗时未记录。</p><h3>Retrieval Top-K 调用</h3><table><tr><th>Video</th><th>检索类型</th><th>调用次数</th><th>实际返回条目</th><th>检索耗时合计</th></tr>{retrieval_rows}</table><p class="muted">文本结果经片段时间头／三元组行解析计数；原检索函数将文本拼成单个字符串，原始 trace 的 result_count 对文本不代表实际条目数。逐次校正记录见 <a href="retrieval_counted.jsonl">retrieval_counted.jsonl</a>。</p></section>
<section><h2>本次索引／QA 资源峰值</h2><div class="scroll"><table><tr><th>Video</th><th>阶段</th><th>Torch GPU allocated MiB</th><th>Torch GPU reserved MiB</th><th>进程 RSS MiB</th><th>进程 CPU %（多核）</th></tr>{resource_rows}</table></div><p class="muted">3 秒采样。进程 CPU % 可超过 100%，因为按单核等于 100% 计。历史视频管线的资源峰值未记录，不能用本次 QA 峰值代替。</p></section>
<section><h2>证据保留与异常</h2><div class="scroll"><table><tr><th>Video</th><th>VSD 候选／采样帧</th><th>YOLO 扫描／VSD 候选</th><th>eSVD</th><th>Event</th><th>Memory</th></tr>{preservation_rows}</table></div><p class="muted">这些是结构数量，不是视觉语义召回率。VSD 非候选帧仍可能成为最终事件主帧；10 秒 caption 在没有关键帧的窗口会用原视频中点帧回填，因此部分 WorldMM Memory 证据不经最终事件主帧。不能把此类证据的 QA 错误直接归因于 eSVD/Event。</p><p>RAW→10fps 的逐原帧 KEEP/DROP 见 <a href="raw_sampling_6Z_qEtbmK34.jsonl">6Z 采样轨迹</a>、<a href="raw_sampling_Ei7hTKr8Ins.jsonl">Ei7 采样轨迹</a>；未采样帧时间为 frame_index/fps 估算。完整逐采样帧决策、分数和 VSD→YOLO→eSVD provenance 见 <a href="stage_trace.jsonl">stage_trace.jsonl</a>；YOLO 检测包含 bbox/class/confidence；Event→Memory 保留事件 ID、关键帧 ID、10 秒 caption 映射。<a href="evsd_semantic_conflicts.jsonl">eSVD 语义冲突候选</a>：{len(conflicts)} 组（其中 6Z {conflicts_by_video['6Z_qEtbmK34']}，Ei7 {conflicts_by_video['Ei7hTKr8Ins']}）。这是“同组 YOLO 标签变化且存在 folded 帧”的候选，未自动等同证据丢失。VSD/Yolo/Event/Memory 的语义保留率需要金标准时间戳或逐例人工核验；本报告给出结构数量与逐例候选，不伪造语义召回百分比。</p>
<p>失败根因数量：</p><ul>{cause_rows}</ul><p>3 例 RETRIEVAL_MISS 已对原帧、记忆文本和实际 Top-K 逐一人工抽查；这证明这些案例在 Memory 阶段仍有证据，但检索未命中。其余失败维持未确定，不以相似文本自动判定上游损失。<a href="anomalies.json">全部异常（含候选时间戳、帧号和复核状态）</a>。各 QA 表行的阶段帧只作自动定位候选，不能当作数据集给出的金标准时间戳。</p></section>
<section><h2>全部 QA／Case 链路表</h2><div class="scroll"><table><tr><th>Video</th><th>QA/Case</th><th>VSD</th><th>YOLO</th><th>eSVD</th><th>Event</th><th>Memory</th><th>Retrieval</th><th>Result</th><th>Root Cause</th></tr>{table_rows}</table></div><p><a href="case_analysis.json">全部案例 JSON（含时间候选、检索轮次、置信级别）</a> · <a href="retrieval_trace.jsonl">检索 Top-K 原始轨迹</a> · <a href="qa_stage_trace.jsonl">QA 逐题耗时／资源／provenance</a> · <a href="resources.jsonl">资源采样</a></p></section>
<section><h2>典型失败案例：原始帧 → VSD → YOLO → eSVD → Event → Memory → QA</h2><p class="muted">图像用于回查算法实际输入/输出；候选定位不等于已人工验证的金标准证据帧。</p>{story_html}</section></main></body></html>"""
    (OUT / "report.html").write_text(content, encoding="utf-8")
    print(json.dumps({"status": status, "answers": len(answers), "correct": report_data["correct"], "root_causes": dict(causes), "typical_cases": [c["ID"] for c in typical]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
