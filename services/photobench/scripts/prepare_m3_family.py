#!/usr/bin/env python3
"""将锁定的 M3-Bench 官方问答和已校验转码视频加入独立视频测评相册。"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def link_or_copy(src, dst):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


def make_questions(annotations, lock):
    rows = []
    seen = set()
    for video in lock['videos']:
        video_id = video['video_id']
        questions = annotations[video_id]['qa_list']
        if len(questions) != video['qa_count']:
            raise ValueError(f'QA count changed: {video_id}')
        media_id = 'm3-' + video_id
        for q in questions:
            qa_id = 'm3-robot-' + q['question_id']
            if qa_id in seen:
                raise ValueError(f'Duplicate QA: {qa_id}')
            seen.add(qa_id)
            refs = [{'media_type': 'video', 'media_id': media_id}]
            rows.append({
                'qa_id': qa_id,
                'question': f"In the video titled M3-Bench {video_id}: {q['question']}",
                'answer': q['answer'],
                'task_type': 'T2_fact_qa',
                'question_type': 'm3_official_video_qa',
                'tags': q['type'],
                'difficulty': 'not_annotated',
                'answerability': 'answerable',
                'expected_action': 'answer',
                'scope': 'video',
                'scope_anchor': media_id,
                'required_evidence_sources': ['video'],
                'retrieval_media_refs': refs,
                'answer_evidence_media_refs': copy.deepcopy(refs),
                'evaluation_protocol': 'full_video_album_adaptation',
                'evidence_granularity': 'video_only_no_temporal_ground_truth',
                'source_dataset': 'ByteDance-Seed/M3-Bench',
                'source_video_id': video_id,
                'source_question': q['question'],
                'source_annotation': copy.deepcopy(q),
                'source_annotation_commit': lock['annotation_commit'],
                'audio_requirement': 'not_individually_audited',
            })
    return rows


def prepare(annotations_path, lock_path, encoded_dir, validation_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite album: {output}')
    lock = read(lock_path)
    if digest(annotations_path) != lock['annotation_sha256']:
        raise ValueError('Annotation SHA256 mismatch')
    annotations = read(annotations_path)
    rows = make_questions(annotations, lock)
    validation = read(validation_path)
    encoded = {v['video_id']: v for v in validation['videos']}
    for v in lock['videos']:
        e = encoded[v['video_id']]
        p = Path(encoded_dir) / (v['video_id'] + '.mp4')
        if e['source_sha256'] != v['sha256'] or digest(p) != e['sha256'] or p.stat().st_size != e['bytes']:
            raise ValueError(f'Video checksum mismatch: {p.name}')
        if abs(e['duration_seconds'] - e['source_duration_seconds']) > 0.25:
            raise ValueError(f'Video duration changed: {p.name}')
        if not e['has_audio']:
            raise ValueError(f'Missing audio: {p.name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.tmp-', dir=output.parent))
    try:
        manifest = {'schema_version': '1.1', 'scope_kind': 'benchmark', 'photos': [], 'videos': [],
                    'faces': [], 'family_relationships': [], 'qa_sets': {}, 'metadata_files': {}}
        manifest['album_id'] = output.name
        manifest['album_name'] = f'M3 家庭视频（{len(rows)} 道官方题改编协议）'
        manifest['require_complete_memory'] = True
        manifest['evaluation_protocol'] = 'full_video_album_adaptation'
        manifest['sentrix_baseline_commit'] = lock['sentrix_baseline_commit']
        manifest['m3_license'] = lock['license']
        manifest['benchmark_notes'] = [
            '完整视频回顾式相册协议，不等同于官方按 before_clip 截断记忆的因果协议。',
            '官方 timestamp 是提问时刻，不是证据定位标注；GT 仅提供视频级归属。',
            '保留原始英文题及答案；仅增加中性视频名称上下文；未用改写扩充题量。',
            '保留音轨；yxw2 视觉关键帧流程不含语音转写，音频依赖题单独解释。',
            '问答、答案及 reasoning 仅供评测，不得导入模型记忆。',
        ]
        provenance_rel = manifest.setdefault('metadata_files', {}).get('video_source_provenance', 'metadata/video_source_provenance.json')
        provenance_path = staging / provenance_rel
        provenance = read(provenance_path) if provenance_path.is_file() else {'videos': []}
        for v in lock['videos']:
            vid = v['video_id']
            rel = f'videos/m3-{vid}.mp4'
            link_or_copy(Path(encoded_dir) / f'{vid}.mp4', staging / rel)
            manifest['videos'].append(rel)
            item = {'video_id': f'm3-{vid}', 'title': f'M3-Bench {vid}', 'images': []}
            provenance['videos'].append(item)
            manifest.setdefault('video_sources', []).append(copy.deepcopy(item))
        manifest['metadata_files']['video_source_provenance'] = provenance_rel
        write(provenance_path, provenance)
        qa_name = f'm3-official-{len(rows)}q'
        qa_rel = f'qa/{qa_name}.jsonl'
        write_rows(staging / qa_rel, rows)
        manifest['qa_sets'][qa_name] = qa_rel
        write(staging / 'provenance/selection.lock.json', lock)
        write(staging / 'provenance/encoding-validation.json', validation)
        write(staging / 'manifest.json', manifest)
        (staging / 'ATTRIBUTION.md').write_text(
            '# M3-Bench 家庭视频子集\n\n来源：ByteDance-Seed/M3-Bench，'
            'https://huggingface.co/datasets/ByteDance-Seed/M3-Bench\n\n'
            '视频及官方标注许可：CC-BY-NC-SA-4.0，'
            'https://creativecommons.org/licenses/by-nc-sa/4.0/\n\n'
            '变更：完整视频统一转码；原始问答保留，问题增加中性视频名称上下文。'
            '此许可说明仅针对新增 M3 数据，原家庭相册保持原有权属。\n', encoding='utf-8')
        staging.rename(output)
        return {'album': output.name, 'photos': len(manifest['photos']), 'videos': len(manifest['videos']),
                'official_m3_questions': len(rows), 'qa_sets': manifest['qa_sets']}
    except BaseException:
        shutil.rmtree(staging)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('annotations', 'lock', 'encoded-dir', 'validation', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(prepare(a.annotations, a.lock, a.encoded_dir, a.validation, a.output), ensure_ascii=False, indent=2))
