import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chess
import numpy as np
import torch

from analysis.state_probe.core import BloomIndex, MODEL_IDS, StateProbe
from analysis.state_probe.pipeline import METRIC_FIELDS, _evaluate_selfplay
from models.chess_tokenizer import ChessTokenizer
from tests.test_twic_category_builder import workspace_temp_dir


class SelfplayInferenceTests(unittest.TestCase):
    def test_selfplay_scoring_disables_gradients_before_numpy_metrics(self):
        tokenizer = ChessTokenizer(str(Path(__file__).resolve().parents[1] / 'tokenizers/chess_uci_vocab.json'))
        models = {name: SimpleNamespace(decoder=SimpleNamespace(tokenizer=tokenizer)) for name in MODEL_IDS}
        prefix = ['e2e4', 'e7e5']
        generated = ['g1f3', 'b8c6']
        board = chess.Board()
        for move in prefix:
            board.push_uci(move)
        fens = []
        for move in generated:
            fens.append(board.fen(en_passant='legal'))
            board.push_uci(move)
        record = {'opening_id': 0, 'twic_issue': 1662, 'trajectory_source': MODEL_IDS[0],
                  'initial_fen': chess.Board().fen(), 'prefix_moves': prefix,
                  'generated_moves': generated, 'fens_before_generated_moves': fens}
        # These readouts intentionally have grad-enabled parameters: before the
        # fix, calling NumPy metrics on their outputs raises RuntimeError.
        probe = StateProbe().eval()
        self.assertTrue(next(probe.parameters()).requires_grad)
        ensemble = ([probe], np.zeros(256, dtype=np.float32), np.ones(256, dtype=np.float32))
        with workspace_temp_dir() as raw:
            directory = Path(raw)
            (directory / 'selfplay_trajectories.jsonl').write_text(json.dumps(record) + '\n')
            config = {'paths': {'output_dir': str(directory)}, 'extraction': {'batch_size': 2},
                      'probe': {'batch_size': 2}}
            with patch('analysis.state_probe.pipeline._load_probe_ensemble', return_value=ensemble), \
                 patch('analysis.state_probe.pipeline.extract_condition_batch',
                       return_value={'final_w256': np.zeros((2, 256), dtype=np.float16)}), \
                 torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                rows = _evaluate_selfplay(config, models, BloomIndex(10), torch.device('cpu'))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(math.isfinite(row[key]) for row in rows for key in METRIC_FIELDS))
        self.assertEqual({row['fen'] for row in rows}, set(fens))
