#!/usr/bin/env python3
"""统一压缩完整 M3 视频；原片 SHA256、完整时长和音轨均验证后才输出清单。"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from prepare_m3_family import digest, read, write


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]))


def run(source, output, lock_path):
    lock = read(lock_path)
    output.mkdir(parents=True, exist_ok=True)
    inputs = []
    for v in lock['videos']:
        p = source / v['path']
        if p.stat().st_size != v['bytes'] or digest(p) != v['sha256']:
            raise ValueError(f'Original checksum mismatch: {p}')
        info = probe(p)
        if not any(s['codec_type'] == 'audio' for s in info['streams']):
            raise ValueError(f'Original missing audio: {p}')
        inputs.append((v, p, float(info['format']['duration'])))
    total_duration = sum(d for _, _, d in inputs)
    # Reserve 2% container overhead; AAC 64 kbit/s; common video bitrate.
    bitrate = int(lock['target_encoded_bytes'] * 8 * .98 / total_duration - 64000)
    if bitrate < 250000:
        raise ValueError('Budget too small for this profile')
    results = []
    for v, src, duration in inputs:
        dst = output / (v['video_id'] + '.mp4')
        if dst.exists():
            raise FileExistsError(f'Refusing to overwrite {dst}')
        partial = output / (v['video_id'] + '.partial.mp4')
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostdin', '-y',
               '-i', str(src), '-map', '0:v:0', '-map', '0:a:0',
               '-vf', 'scale=960:540,fps=15', '-c:v', 'h264_nvenc', '-preset', 'slow',
               '-rc', 'cbr', '-b:v', str(bitrate), '-maxrate', str(bitrate), '-bufsize', str(bitrate * 2),
               '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '64k', '-movflags', '+faststart', str(partial)]
        started = time.time()
        subprocess.run(cmd, check=True)
        info = probe(partial)
        actual_duration = float(info['format']['duration'])
        has_audio = any(s['codec_type'] == 'audio' for s in info['streams'])
        if abs(actual_duration - duration) > .25 or not has_audio:
            raise ValueError(f'Duration/audio validation failed: {partial}')
        partial.rename(dst)
        record = {'video_id': v['video_id'], 'bytes': dst.stat().st_size, 'sha256': digest(dst),
                  'source_sha256': v['sha256'], 'source_duration_seconds': duration,
                  'duration_seconds': actual_duration, 'has_audio': has_audio,
                  'encoding_seconds': round(time.time() - started, 3)}
        results.append(record)
        print(json.dumps(record), flush=True)
    result = {'status': 'verified', 'profile': {'width': 960, 'height': 540, 'fps': 15,
              'video_codec': 'h264_nvenc', 'preset': 'slow', 'rate_control': 'cbr',
              'video_bitrate': bitrate, 'audio_codec': 'aac', 'audio_bitrate': 64000},
              'ffmpeg_version': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
              'total_bytes': sum(r['bytes'] for r in results),
              'total_duration_seconds': total_duration, 'videos': results}
    write(output / 'encoding-validation.json', result)
    if not .9 * lock['target_encoded_bytes'] <= result['total_bytes'] <= 1.1 * lock['target_encoded_bytes']:
        raise ValueError('Encoded size outside ±10% target; inspect profile and re-encode in new directory')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'lock'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.output, a.lock)
