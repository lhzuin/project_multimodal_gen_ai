import csv
import io
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import chess
import chess.pgn

from analysis.ply_comparison.aggregate import ComparisonAccumulator, relative_bucket_for_ply
from analysis.ply_comparison.evaluate import PER_MOVE_FIELDS, main
from analysis.ply_comparison.plot_results import MODELS
from analysis.ply_comparison.resume import recover, text_hashes


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows, offsets, raw = [], [], b''
        for gid, sans in enumerate((('e4','e5','Nf3','Nc6'),('d4','d5','c4','e6'))):
            game = chess.pgn.Game()
            game.headers.update(TWICIssue='1667', Date='2026.10.20', WhiteElo='2000', BlackElo='2100')
            node, board, positions = game, game.board(), []
            for ply, san in enumerate(sans):
                move = board.parse_san(san)
                positions.append((board.fen(), move.uci(), board.legal_moves.count()))
                node = node.add_variation(move)
                board.push(move)
            data = (game.accept(chess.pgn.StringExporter(headers=True,variations=False,comments=False))+'\n\n').encode()
            offsets.append({'idx':gid,'start':len(raw),'end':len(raw)+len(data)})
            raw += data
            for model in MODELS:
                for ply, (fen, target, legal) in enumerate(positions):
                    self.rows.append(dict(zip(PER_MOVE_FIELDS, (
                        gid,ply,4,'0_20',relative_bucket_for_ply(ply,4),model,1667,'2026.10.20',
                        2000,2100,fen,target,target,1,1,.1,math.exp(-.1),legal,
                        ply+1 if model.startswith('decoder') else '',
                        0 if model.startswith('decoder') else '',
                        1 if model.startswith('encoder') else ''))))
        self.pgn = self.root/'games.pgn'
        self.pgn.write_bytes(raw)
        self.index = {'games':offsets,'num_games':2}

    def run_recovery(self, rows, **kwargs):
        source = self.root/'source.csv'
        with source.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=PER_MOVE_FIELDS)
            w.writeheader(); w.writerows(rows)
        out=io.StringIO()
        writer=csv.DictWriter(out,fieldnames=PER_MOVE_FIELDS)
        acc, relative = ComparisonAccumulator(), ComparisonAccumulator()
        result = recover(source,pgn_path=self.pgn,index=self.index,fields=PER_MOVE_FIELDS,
                         writer=writer,accumulator=acc,relative_accumulator=relative,**kwargs)
        return result, acc, out.getvalue()

    def test_complete_games_rebuild_all_model_counts(self):
        result, acc, output=self.run_recovery(self.rows)
        self.assertEqual(result['game_ids'],{0,1})
        self.assertEqual(result['early_occurrences'],8)
        self.assertEqual(len(output.splitlines()),32)
        for game in acc.games.values():
            for model in MODELS:
                self.assertEqual(game[model,'0_20'].n,4)
                self.assertAlmostEqual(game[model,'0_20'].nll,.4)

    def test_incomplete_final_game_is_not_credited(self):
        result,acc,_=self.run_recovery(self.rows[:-1])
        self.assertEqual(result['game_ids'],{0})
        self.assertEqual(result['discarded_final_games'],[1])
        self.assertNotIn(1,acc.games)

    def test_incomplete_interior_game_fails(self):
        with self.assertRaisesRegex(ValueError,'incomplete interior'):
            self.run_recovery(self.rows[:15]+self.rows[16:])

    def test_wrong_fen_fails(self):
        rows=[dict(r) for r in self.rows]
        rows[0]['fen_before']='wrong'
        with self.assertRaisesRegex(ValueError,'decision mismatch'):
            self.run_recovery(rows)

    def test_duplicate_model_decision_fails(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.run_recovery(self.rows[:1]+self.rows)

    def test_resume_cannot_overwrite_source(self):
        with self.assertRaisesRegex(ValueError,'fresh output'):
            main(['--resume-csv','source.csv','--resume-source','proof.json','--overwrite'])

    def test_earlier_source_owns_entire_overlapping_game(self):
        result,acc,output=self.run_recovery(self.rows,skip_game_ids={0})
        self.assertEqual(result['game_ids'],{1})
        self.assertEqual(result['validated_game_ids'],{0,1})
        self.assertNotIn(0,acc.games)
        self.assertEqual(len(output.splitlines()),16)
        self.assertEqual(result['early_occurrences'],4)

    def test_mismatched_provenance_list_fails_before_inference(self):
        with self.assertRaisesRegex(ValueError,'matching resume-source'):
            main(['--resume-csv','a.csv','--resume-csv','b.csv','--resume-source','a.json'])

    def test_text_compatibility_accepts_newlines_but_not_code_changes(self):
        accepted=text_hashes(b'x = 1\ny = 2\n')
        self.assertIn(hashlib.sha256(b'x = 1\r\ny = 2\r\n').hexdigest(),accepted)
        self.assertNotIn(hashlib.sha256(b'x = 2\ny = 2\n').hexdigest(),accepted)
