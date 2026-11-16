import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts'))


class PaperExampleTests(unittest.TestCase):
    def test_paper_preset_selects_fp_test_event_with_csv_label_policy(self):
        from fp_audit import read_fp_split
        nb = json.loads((REPO/'notebooks/ts_satfire_paper_fp_example.ipynb').read_text())
        config = next(c for c in nb['cells'] if 'parameters' in c['metadata'].get('tags', []))
        namespace = dict(Path=Path, REPO=REPO)
        exec(''.join(config['source']), namespace)
        records, _ = read_fp_split(REPO/'legacy/roi', REPO/'legacy/scripts/dataset_gen_pred.py')
        event = records[namespace['EVENT_ID']]
        self.assertEqual(event['event_id'], 'US_2021_NM3323810847220210520')
        self.assertEqual(event['split'], 'test')
        self.assertEqual(event['legacy_label_sel'], '0')
        self.assertIsNone(namespace['ANCHOR_DATE'])
        self.assertFalse(namespace['DEMO_MODE'])
        self.assertTrue(all(not c.get('outputs') for c in nb['cells']))


if __name__ == '__main__':
    unittest.main()
