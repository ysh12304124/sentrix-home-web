import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare_m3_family', Path(__file__).parents[1] / 'scripts/prepare_m3_family.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class M3AdapterTests(unittest.TestCase):
    def fixture(self, root):
        q = {'question_id': 'kitchen_05_Q01', 'question': 'Where is the flour?', 'answer': 'Bottom cabinet.',
             'reasoning': 'SECRET_EVALUATOR_REASONING', 'timestamp': '06:18', 'before_clip': 11,
             'type': ['Cross-Modal Reasoning']}
        annotation = root / 'robot.json'
        m.write(annotation, {'kitchen_05': {'qa_list': [q]}})
        video = root / 'kitchen_05.mp4'
        video.write_bytes(b'validated-video-fixture')
        lock = {'annotation_sha256': m.digest(annotation), 'annotation_commit': 'source-commit',
                'sentrix_baseline_commit': 'baseline', 'license': 'CC-BY-NC-SA-4.0',
                'videos': [{'video_id': 'kitchen_05', 'qa_count': 1, 'sha256': 'source-sha'}]}
        m.write(root / 'lock.json', lock)
        validation = {'videos': [{'video_id': 'kitchen_05', 'source_sha256': 'source-sha',
                    'sha256': m.digest(video), 'bytes': video.stat().st_size, 'has_audio': True,
                    'duration_seconds': 100, 'source_duration_seconds': 100}]}
        m.write(root / 'validation.json', validation)
        return q, lock

    def test_official_preservation_and_no_import_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q, _ = self.fixture(root)
            out = root / 'album'
            m.prepare(root/'robot.json', root/'lock.json', root, root/'validation.json', out)
            self.assertTrue(m.read(out/'manifest.json')['require_complete_memory'])
            row = json.loads((out/'qa/m3-official-1q.jsonl').read_text())
            self.assertEqual(row['source_annotation'], q)
            self.assertEqual(row['answer'], q['answer'])
            self.assertEqual(row['retrieval_media_refs'], [{'media_type': 'video', 'media_id': 'm3-kitchen_05'}])
            metadata = (out/'metadata/video_source_provenance.json').read_text()
            self.assertNotIn(q['reasoning'], metadata)
            self.assertNotIn(q['answer'], metadata)
            self.assertNotIn('timestamp', metadata)
            self.assertEqual(row['evidence_granularity'], 'video_only_no_temporal_ground_truth')
            with self.assertRaises(FileExistsError):
                m.prepare(root/'robot.json', root/'lock.json', root, root/'validation.json', out)

    def test_corruption_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root/'kitchen_05.mp4').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                m.prepare(root/'robot.json', root/'lock.json', root, root/'validation.json', root/'album')
            self.assertFalse((root/'album').exists())

    def test_audio_and_duration_required(self):
        for changed in ({'has_audio': False}, {'duration_seconds': 90}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                validation = m.read(root/'validation.json')
                validation['videos'][0].update(changed)
                m.write(root/'validation.json', validation)
                with self.assertRaises(ValueError):
                    m.prepare(root/'robot.json', root/'lock.json', root, root/'validation.json', root/'album')
                self.assertFalse((root/'album').exists())


if __name__ == '__main__':
    unittest.main()
