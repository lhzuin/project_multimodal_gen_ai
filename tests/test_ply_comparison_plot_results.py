import csv
from contextlib import contextmanager
from pathlib import Path
import shutil
import unittest
import uuid
import xml.etree.ElementTree as ET

from analysis.ply_comparison.plot_results import BUCKETS, FAMILIES, MODELS, load_results, main


SUMMARY_FIELDS = [
    "model_id", "bucket", "n_games", "n_plies", "accuracy_top1",
    "accuracy_top5", "nll_legal",
]
DELTA_FIELDS = [
    "family", "bucket", "n_games", "n_plies", "delta_accuracy_top1",
    "ci_low_accuracy_top1", "ci_high_accuracy_top1", "delta_nll_legal",
    "ci_low_nll_legal", "ci_high_nll_legal",
]


def write_csv(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def workspace_dir():
    directory = Path(__file__).parent / f"_plot_test_{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def fixture(directory: Path):
    summary = directory / "bucket_summary.csv"
    delta = directory / "paired_deltas.csv"
    rows = []
    for bucket in BUCKETS:
        for model in MODELS:
            balanced = model.endswith("balanced")
            rows.append({
                "model_id": model, "bucket": bucket, "n_games": 10,
                "n_plies": 100, "accuracy_top1": 0.52 if balanced else 0.50,
                "accuracy_top5": 0.82 if balanced else 0.80,
                "nll_legal": 1.45 if balanced else 1.50,
            })
    write_csv(summary, SUMMARY_FIELDS, rows)
    delta_rows = []
    for family in FAMILIES:
        for bucket in BUCKETS:
            delta_rows.append({
                "family": family, "bucket": bucket, "n_games": 10, "n_plies": 100,
                "delta_accuracy_top1": 0.02,
                "ci_low_accuracy_top1": 0.01, "ci_high_accuracy_top1": 0.03,
                "delta_nll_legal": -0.05,
                "ci_low_nll_legal": -0.07, "ci_high_nll_legal": -0.03,
            })
    write_csv(delta, DELTA_FIELDS, delta_rows)
    return summary, delta


class PlotResultsTest(unittest.TestCase):
    def test_outputs_are_well_formed_and_document_metrics(self):
        with workspace_dir() as directory:
            summary, delta = fixture(directory)
            output = directory / "figures"
            main(["--summary", str(summary), "--deltas", str(delta),
                  "--output-dir", str(output)])
            for name in (
                "accuracy_by_ply.svg", "nll_by_ply.svg",
                "delta_accuracy_by_ply.svg", "delta_nll_by_ply.svg",
            ):
                self.assertEqual(ET.parse(output / name).getroot().tag,
                                 "{http://www.w3.org/2000/svg}svg")
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("paired resampling", readme)
            self.assertIn("+2.00", readme)
            self.assertIn("-0.0500", readme)

    def test_rejects_mismatched_samples(self):
        with workspace_dir() as directory:
            summary, delta = fixture(directory)
            text = summary.read_text(encoding="utf-8")
            summary.write_text(text.replace("encoder_balanced,0_20,10,100,",
                                            "encoder_balanced,0_20,10,99,"),
                               encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unequal samples"):
                load_results(summary, delta)


if __name__ == "__main__":
    unittest.main()
