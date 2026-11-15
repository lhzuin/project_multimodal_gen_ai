from pathlib import Path
import unittest

from analysis.ply_comparison.audit_corpus import audit
from dataset.offsets import ensure_offsets
from tests.test_twic_category_builder import workspace_temp_dir


class CorpusAuditTests(unittest.TestCase):
    def test_audits_dates_and_explicit_fast_controls(self):
        with workspace_temp_dir() as raw_tmp:
            root = Path(raw_tmp)
            pgn = root / "games.pgn"
            records = []
            for issue in range(1662, 1668):
                control = "300" if issue % 2 else "5400+30"
                records.append(
                    f'[Event "Test"]\n[TWICIssue "{issue}"]\n'
                    f'[Date "2026.10.{issue - 1650:02d}"]\n'
                    f'[TimeControl "{control}"]\n\n1. e4 e5 *\n\n'
                )
            pgn.write_text("".join(records), encoding="utf-8")
            index = root / "index.json"
            ensure_offsets(str(pgn), str(index))
            result = audit(pgn, index)
            self.assertEqual(result["total_games"], 6)
            self.assertEqual(result["explicit_180_or_300_second_games"], 3)
            self.assertTrue(all(issue["dates_valid"] == 1 for issue in result["issues"].values()))


if __name__ == "__main__":
    unittest.main()
