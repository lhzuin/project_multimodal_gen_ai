"""Scientific protocol, atomic recovery, and end-to-end reporting regressions."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np
import torch
import yaml

from analysis.time_control_comparison.aggregate import METRICS, bootstrap, game_arrays, summarize
from analysis.time_control_comparison.common import CATEGORIES, Config, load_config
from analysis.time_control_comparison.corpus import game_identity, prepare, validate_game
from analysis.time_control_comparison.inference import evaluate_game, game_positions, score_moves
from analysis.time_control_comparison.report import render
from analysis.time_control_comparison.storage import ResultStore, run_lock
from analysis.time_control_comparison.__main__ import evaluate, main, summarize_saved
from dataset.offsets import build_pgn_offsets
from models.chess_encoder_player import ChessEncoderPlayerV2
from tests.test_twic_category_builder import workspace_temp_dir
from utils.dataset_by_category.categories import get_event_category


LINES = (
    ("e4", "e5", "Nf3", "Nc6"), ("d4", "d5", "c4", "e6"), ("c4", "e5", "Nc3", "Nf6"),
    ("Nf3", "d5", "g3", "c5"), ("g3", "d5", "Bg2", "e5"), ("b3", "e5", "Bb2", "Nc6"),
    ("f4", "d5", "Nf3", "Nf6"), ("Nc3", "d5", "e4", "d4"), ("e3", "e5", "d4", "exd4"),
)


def make_game(sans=LINES[0], *, category="classic", fen=None):
    board = chess.Board(fen) if fen else chess.Board()
    game = chess.pgn.Game.from_board(board)
    event = {"classic": "Open", "rapid": "Super RAPID", "blitz": "Super Blitz Rapid"}[category]
    game.headers.update(Event=event, WhiteElo="2000", BlackElo="2100", TWICIssue="1667", Date="2026.10.20")
    node = game
    for san in sans:
        move = board.parse_san(san)
        node = node.add_variation(move)
        board.push(move)
    return game


def make_fixture(root, *, quota=2):
    corpus = root / "corpus"
    corpus.mkdir()
    counts = {}
    for i, category in enumerate(CATEGORIES):
        path = corpus / f"test_{category}.pgn"
        games = [make_game(line, category=category) for line in LINES[i*3:(i+1)*3]]
        path.write_text("\n\n".join(g.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)) for g in games) + "\n\n")
        build_pgn_offsets(str(path), str(corpus / f"test_{category}_index.json"))
        counts[category] = {"games": 3}
    (corpus / "test_report.json").write_text(json.dumps({
        "range_complete": True, "config": {"all_games_in_range": True, "start_week": 1667, "lower_week": 1662,
        "categories": list(CATEGORIES), "min_elo": 1600, "include_variant": False}, "categories": counts}))
    checkpoints = {}
    for category in CATEGORIES:
        path = root / f"{category}.pt"
        path.write_bytes(category.encode())
        checkpoints[category] = path
    return Config(root, corpus, "test", root / "output", checkpoints, games_per_category=quota, bootstrap=50, batch_size=2)


class FakePolicy(torch.nn.Module):
    def __init__(self, category):
        super().__init__()
        self.category = category
        self.seen_inputs = []

    def forward(self, **inputs):
        assert not torch.is_grad_enabled()
        self.seen_inputs.append({k: v.detach().cpu().clone() for k, v in inputs.items()})
        n = len(inputs["labels64"])
        policy = torch.zeros(n, 64, 64)
        origin, destination = {"classic": (chess.E2, chess.E4), "rapid": (chess.D2, chess.D4), "blitz": (chess.G1, chess.F3)}[self.category]
        policy[:, origin, destination] = 2
        return {"policy_logits": policy, "promo_logits": torch.zeros(n, 5)}


def fake_models(*args):
    return {category: FakePolicy(category) for category in CATEGORIES}


class InferenceTests(unittest.TestCase):
    def test_legal_distribution_is_normalized_including_underpromotions(self):
        board = chess.Board("k7/4P3/8/8/8/8/8/K7 w - - 0 1")
        legal = list(board.legal_moves)
        policy = np.zeros((64, 64)); policy[chess.E7, chess.E8] = 3
        promo = np.asarray([100., -2., 0., 1., 4.])
        scores = [score_moves(legal, target, policy, promo) for target in legal]
        self.assertAlmostEqual(sum(math.exp(s.target_logp) for s in scores), 1)
        q = score_moves(legal, chess.Move.from_uci("e7e8q"), policy, promo)
        knight = score_moves(legal, chess.Move.from_uci("e7e8n"), policy, promo)
        self.assertAlmostEqual(q.target_logp - knight.target_logp, 6.)

    def test_pair_first_greedy_matches_deployment_even_when_joint_argmax_differs(self):
        board = chess.Board("k7/4P3/8/8/8/8/8/K7 w - - 0 1")
        legal = list(board.legal_moves)
        policy = torch.zeros((64, 64)); policy[chess.E7, chess.E8] = 1
        policy[chess.A1, chess.B1] = .9
        promo = torch.zeros(5)
        score = score_moves(legal, legal[0], policy, promo)
        fake = SimpleNamespace(
            forward=lambda **kwargs: {"policy_logits": policy[None], "promo_logits": promo[None]},
            legal_mask_from_fen=ChessEncoderPlayerV2.legal_mask_from_fen,
            apply_legal_mask=ChessEncoderPlayerV2.apply_legal_mask,
        )
        deployed = ChessEncoderPlayerV2.sample_moves(fake, fen=[board.fen()], greedy=True)[0].uci()
        self.assertEqual(score.predicted, deployed)
        self.assertEqual(score.predicted, "e7e8q")  # class-zero fallback follows python-chess legal order
        probability = {move.uci(): score_moves(legal, move, policy, promo).target_logp for move in legal}
        self.assertEqual(max(probability, key=probability.get), "a1b1")

    def test_flat_argmax_ties_and_extreme_logits_are_stable(self):
        board = chess.Board()
        legal = list(board.legal_moves)
        policy = np.full((64, 64), -10000.)
        score = score_moves(legal, chess.Move.from_uci("e2e4"), policy, np.zeros(5))
        expected = min(legal, key=lambda m: m.from_square * 64 + m.to_square)
        self.assertEqual(score.predicted, expected.uci())
        self.assertAlmostEqual(-score.target_logp, math.log(len(legal)))
        policy[chess.E2, chess.E4] = 10000
        score = score_moves(legal, chess.Move.from_uci("d2d4"), policy, np.zeros(5))
        self.assertTrue(math.isfinite(score.target_logp))
        self.assertLess(score.target_logp, -19000)

    def test_illegal_target_or_nan_fails(self):
        legal = list(chess.Board().legal_moves)
        with self.assertRaisesRegex(ValueError, "target"):
            score_moves(legal, chess.Move.from_uci("e2e5"), np.zeros((64,64)), np.zeros(5))
        policy = np.zeros((64,64)); policy[chess.E2, chess.E4] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            score_moves(legal, legal[0], policy, np.zeros(5))

    def test_forced_move_has_probability_one(self):
        move = chess.Move.from_uci("a1a2")
        score = score_moves([move], move, np.zeros((64,64)), np.zeros(5))
        self.assertEqual(score.target_logp, 0)
        self.assertEqual(score.predicted, move.uci())

    def test_true_pre_move_state_orientation_and_shared_inputs(self):
        game = make_game()
        positions = list(game_positions(game))
        self.assertEqual(positions[0]["fen_before"], chess.Board().fen())
        self.assertEqual(positions[0]["labels"][0], 10)  # black rook at a8
        self.assertEqual(positions[0]["labels"][56], 4)  # white rook at a1
        self.assertEqual(positions[1]["ep_square"], chess.E3)
        self.assertEqual([p["turn_white"] for p in positions], [1,0,1,0])
        models = fake_models()
        kept, predictions = evaluate_game(game, models, torch.device("cpu"), 3)
        self.assertEqual(len(kept), 4); self.assertEqual(len(predictions), 12)
        for batch in range(2):
            for key in models['classic'].seen_inputs[batch]:
                self.assertTrue(torch.equal(models['classic'].seen_inputs[batch][key], models['rapid'].seen_inputs[batch][key]))
        # Batched predictions equal one-position inference.
        _, singly = evaluate_game(game, fake_models(), torch.device("cpu"), 1)
        key = lambda r: (r['ply'],r['model'])
        self.assertEqual(sorted(predictions,key=key), sorted(singly,key=key))

    def test_en_passant_and_castling_targets_are_complete_legal_moves(self):
        for game in (make_game(("e4","a6","e5","d5","exd6")),
                     make_game(("e4","e5","Nf3","Nc6","Bc4","Nf6","O-O"))):
            models = fake_models()
            _, predictions = evaluate_game(game, models, torch.device('cpu'), 8)
            self.assertTrue(all(math.isfinite(p['target_logp']) for p in predictions))


class CorpusTests(unittest.TestCase):
    def test_frozen_selection_is_reproducible_and_resume_checks_content(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            config = make_fixture(Path(tmp))
            first = prepare(config, resume=True)
            second = prepare(config, resume=True)
            self.assertEqual(first, second)
            self.assertEqual(len({g['game_id'] for g in first['games']}), 6)
            self.assertEqual([sum(g['category']==c for g in first['games']) for c in CATEGORIES], [2,2,2])
            other = prepare(replace(config, output_dir=Path(tmp)/'second'), resume=True)
            self.assertEqual([g['game_id'] for g in first['games']], [g['game_id'] for g in other['games']])
            with self.assertRaisesRegex(FileExistsError,'selection exists'):
                prepare(config, resume=False)
            with self.assertRaisesRegex(ValueError,'changed'):
                prepare(replace(config, seed=99), resume=True)
            (config.output_dir/'selected_games.pgn').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                prepare(config, resume=True)

    def test_edited_selection_manifest_is_rejected(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            config=make_fixture(Path(tmp));prepare(config,resume=True)
            path=config.output_dir/'selection_manifest.json';manifest=json.loads(path.read_text())
            manifest['games'].pop();path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'changed'):
                prepare(config,resume=True)

    def test_quota_shortfall_is_explicit_and_never_duplicates(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            config = make_fixture(Path(tmp),quota=4)
            with self.assertRaisesRegex(ValueError,'insufficient distinct'):
                prepare(config,resume=True)
            self.assertFalse((config.output_dir/'selection_manifest.json').exists())
            report=json.loads((config.output_dir/'corpus_audit.json').read_text())
            self.assertEqual(report['shortfalls'],dict.fromkeys(CATEGORIES,1))
            selected=prepare(replace(config,allow_shortfall=True),resume=True)
            self.assertEqual(len(selected['games']),9)

    def test_event_precedence_rating_filter_and_variant_match_training(self):
        with workspace_temp_dir() as tmp:
            config=make_fixture(Path(tmp)); game=make_game(category='blitz')
            self.assertEqual(get_event_category(game.headers['Event']),'blitz')
            self.assertIsNone(validate_game(game,config,'blitz'))
            game.headers['BlackElo']='1599'
            self.assertEqual(validate_game(game,config,'blitz'),'elo')
            game.headers['BlackElo']='?'
            self.assertEqual(validate_game(game,config,'blitz'),'missing_or_invalid_metadata')
            game.headers['BlackElo']='2000';game.headers['Variant']='Standard'
            self.assertEqual(validate_game(game,config,'blitz'),'variant')

    def test_identity_ignores_metadata_but_not_moves(self):
        game=make_game(); identity=game_identity(game)
        game.headers.update(Event='Reprinted Blitz',Date='2020.01.01')
        self.assertEqual(identity,game_identity(game))
        self.assertNotEqual(identity,game_identity(make_game(LINES[1])))

    def test_reference_overlap_is_removed_before_sampling(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);config=make_fixture(root)
            ref=root/'training.pgn';ref.write_text(str(make_game()))
            result=prepare(replace(config,exclude_pgns=(ref,)),resume=True)
            self.assertNotIn(game_identity(make_game()),{g['game_id'] for g in result['games']})
            self.assertEqual(result['audit']['filters']['classic']['excluded_reference_overlap'],1)

    def test_incomplete_builder_range_is_rejected(self):
        with workspace_temp_dir() as tmp:
            config=make_fixture(Path(tmp));path=config.corpus_dir/'test_report.json'
            report=json.loads(path.read_text());report['range_complete']=False;path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'complete requested range'):
                prepare(config,resume=True)

    def test_cross_category_reprints_are_excluded_from_every_category(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            config=make_fixture(Path(tmp));path=config.corpus_dir/'test_rapid.pgn'
            duplicate=make_game(category='rapid')
            with path.open('a') as handle:handle.write(str(duplicate)+'\n\n')
            build_pgn_offsets(str(path),str(config.corpus_dir/'test_rapid_index.json'))
            report_path=config.corpus_dir/'test_report.json';report=json.loads(report_path.read_text())
            report['categories']['rapid']['games']=4;report_path.write_text(json.dumps(report))
            selected=prepare(config,resume=True)
            self.assertEqual(selected['audit']['cross_category_duplicate_conflicts_excluded'],1)
            self.assertNotIn(game_identity(duplicate),{g['game_id'] for g in selected['games']})

    def test_nonstandard_initial_fen_is_supported_without_variant(self):
        with workspace_temp_dir() as tmp:
            config=make_fixture(Path(tmp))
            game=make_game(('e8=N',),fen='k7/4P3/8/8/8/8/8/K7 w - - 0 1')
            self.assertIsNone(validate_game(game,config,'classic'))
            position=list(game_positions(game))[0]
            self.assertEqual(position['target_uci'],'e7e8n')
            self.assertEqual(position['is_promotion'],1)


class RecoveryAndReportingTests(unittest.TestCase):
    def test_interrupted_evaluation_resumes_exactly_and_reports_all_matrices(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            config=make_fixture(Path(tmp));selection=prepare(config,resume=True)
            with patch('analysis.time_control_comparison.__main__.load_models',side_effect=fake_models):
                self.assertFalse(evaluate(config,selection,resume=True,device=torch.device('cpu'),max_new_games=2))
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    summarize_saved(config)
                with patch('analysis.time_control_comparison.__main__.evaluate_game', wraps=evaluate_game) as calls:
                    self.assertTrue(evaluate(config,selection,resume=True,device=torch.device('cpu')))
                    self.assertEqual(calls.call_count,4)
            summarize_saved(config);render(config)
            for name in ('README.md','corpus_profile.json','per_move.csv','prediction_summary.csv','paired_deltas.csv',
                         'disagreement.csv','category_macro.csv','specialization.csv','games.csv'):
                self.assertTrue((config.output_dir/name).exists(),name)
            self.assertEqual(len((config.output_dir/'per_move.csv').read_text().splitlines()),73)
            store=ResultStore(config.output_dir/'results.sqlite',json.loads((config.output_dir/'run_manifest.json').read_text())['identity'],resume=True)
            try:
                arrays=game_arrays(store.connection)
                self.assertEqual(arrays['classic','all'][:,0].sum(),8)
                self.assertTrue(all((v[:,0]>0).all() for v in arrays.values()))
                # Paired game resampling leaves identical policies' differences exactly zero.
                draws=bootstrap(arrays['classic','all'],50,42)
                self.assertTrue(np.array_equal(draws,bootstrap(arrays['classic','all'],50,42)))
            finally: store.close()
            readme=(config.output_dir/'README.md').read_text()
            self.assertIn('Paired differences',readme)
            self.assertIn('not certified clock',readme)
            self.assertIn('classic',(config.output_dir/'figures/agreement_matrix.svg').read_text())

    def test_keyboard_interrupt_does_not_credit_partial_game(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            config=make_fixture(Path(tmp));selection=prepare(config,resume=True)
            with patch('analysis.time_control_comparison.__main__.load_models',side_effect=fake_models), \
                 patch('analysis.time_control_comparison.__main__.evaluate_game',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    evaluate(config,selection,resume=True,device=torch.device('cpu'))
            run=json.loads((config.output_dir/'run_manifest.json').read_text())
            self.assertEqual(run['status'],'interrupted');self.assertEqual(run['completed_games'],0)

    def test_interrupt_after_committed_game_resumes_remaining_games(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            config=make_fixture(Path(tmp));selection=prepare(config,resume=True)
            calls=0
            def interrupt_second(*args):
                nonlocal calls
                calls+=1
                if calls==2:raise KeyboardInterrupt
                return evaluate_game(*args)
            with patch('analysis.time_control_comparison.__main__.load_models',side_effect=fake_models), \
                 patch('analysis.time_control_comparison.__main__.evaluate_game',side_effect=interrupt_second):
                with self.assertRaises(KeyboardInterrupt):
                    evaluate(config,selection,resume=True,device=torch.device('cpu'))
            run=json.loads((config.output_dir/'run_manifest.json').read_text());self.assertEqual(run['completed_games'],1)
            with patch('analysis.time_control_comparison.__main__.load_models',side_effect=fake_models), \
                 patch('analysis.time_control_comparison.__main__.evaluate_game',wraps=evaluate_game) as continued:
                evaluate(config,selection,resume=True,device=torch.device('cpu'))
                self.assertEqual(continued.call_count,5)

    def test_changed_checkpoint_rejects_resume(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            config=make_fixture(Path(tmp));selection=prepare(config,resume=True)
            with patch('analysis.time_control_comparison.__main__.load_models',side_effect=fake_models):
                evaluate(config,selection,resume=True,device=torch.device('cpu'),max_new_games=1)
            config.checkpoints['rapid'].write_bytes(b'new weights')
            with self.assertRaisesRegex(ValueError,'identity changed'):
                evaluate(config,selection,resume=True,device=torch.device('cpu'))

    def test_inference_code_change_rejects_resume_but_existing_results_survive(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            config=make_fixture(Path(tmp));selection=prepare(config,resume=True)
            with patch('analysis.time_control_comparison.__main__.load_models',side_effect=fake_models):
                evaluate(config,selection,resume=True,device=torch.device('cpu'),max_new_games=1)
            with patch('analysis.time_control_comparison.__main__.code_hashes',return_value={'inference':'new'}):
                with self.assertRaisesRegex(ValueError,'identity changed'):
                    evaluate(config,selection,resume=True,device=torch.device('cpu'))
            run=json.loads((config.output_dir/'run_manifest.json').read_text())
            with sqlite3.connect(config.output_dir/'results.sqlite') as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM games').fetchone()[0],1)
            self.assertEqual(run['completed_games'],1)

    def test_atomic_game_rollback_and_missing_policy_rejection(self):
        with workspace_temp_dir() as tmp:
            root=Path(tmp);store=ResultStore(root/'results.sqlite',{'test':1},resume=False)
            try:
                game=make_game();positions,predictions=evaluate_game(game,fake_models(),torch.device('cpu'),4)
                entry={'game_id':'one','category':'classic','game_plies':4,'headers':dict(game.headers)}
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    store.save_game(entry,positions,predictions[:-1])
                self.assertEqual(store.completed_ids(),set())
                broken=[dict(p) for p in predictions];broken[0].pop('entropy')
                with self.assertRaises(KeyError):store.save_game(entry,positions,broken)
                self.assertEqual(store.completed_ids(),set())
                self.assertEqual(store.connection.execute('SELECT COUNT(*) FROM positions').fetchone()[0],0)
                store.save_game(entry,positions,predictions);store.validate_complete([entry])
                with self.assertRaises(sqlite3.IntegrityError):store.save_game(entry,positions,predictions)
                self.assertEqual(store.completed_ids(),{'one'})
            finally:store.close()

    def test_position_weighting_and_paired_intervals(self):
        n=len(METRICS);array=np.zeros((2,n+1));array[:,0]=[1,9]
        classic=METRICS.index('classic:agreement')+1;rapid=METRICS.index('rapid:agreement')+1
        array[:,classic]=[1,0];array[:,rapid]=[1,0]
        self.assertAlmostEqual(array[:,classic].sum()/array[:,0].sum(),.1)
        self.assertAlmostEqual((array[:,classic]/array[:,0]).mean(),.5)
        draws=bootstrap(array,200,42)
        self.assertTrue(np.all(draws[:,classic-1]-draws[:,rapid-1]==0))
        self.assertGreater(np.ptp(draws[:,classic-1]),0)

    def test_category_macro_weights_categories_equally(self):
        with workspace_temp_dir() as tmp:
            root=Path(tmp);config=make_fixture(root);config.output_dir.mkdir()
            store=ResultStore(root/'tiny.sqlite',{'test':1},resume=False)
            try:
                for length,category in zip((2,4,6),CATEGORIES):
                    correct=int(category=='classic')
                    positions=[{'ply':i,'bucket':'0_20','fen_before':chess.Board().fen(),'target_uci':'e2e4',
                                'n_legal':20,'turn_white':1,'is_promotion':0} for i in range(length)]
                    predictions=[{'ply':i,'model':m,'predicted_uci':'e2e4' if correct else 'd2d4',
                        'target_logp':-1.,'predicted_logp':-1.,'top1_correct':correct,'top5_correct':1,
                        'pair_correct':correct,'entropy':1.} for i in range(length) for m in CATEGORIES]
                    store.save_game({'game_id':category,'category':category,'game_plies':length,
                                     'headers':dict(make_game().headers)},positions,predictions)
                summarize(store,config)
                import csv
                with (config.output_dir/'category_macro.csv').open() as source:rows=list(csv.DictReader(source))
                for row in rows:
                    if row['metric']=='agreement':
                        self.assertAlmostEqual(float(row['category_macro_mean']),1/3)
                        self.assertAlmostEqual(float(row['ci_low']),1/3)
            finally:store.close()

    def test_nonforced_disagreement_removes_only_forced_positions(self):
        with workspace_temp_dir() as tmp:
            root=Path(tmp);config=make_fixture(root);store=ResultStore(root/'tiny.sqlite',{'test':1},resume=False)
            try:
                for category in CATEGORIES:
                    positions=[{'ply':i,'bucket':'0_20','fen_before':chess.Board().fen(),'target_uci':'e2e4',
                                'n_legal':1 if i==0 else 20,'turn_white':1,'is_promotion':0} for i in range(2)]
                    predictions=[]
                    for p in positions:
                        for model in CATEGORIES:
                            correct=not (p['ply']==1 and model=='rapid')
                            predictions.append({'ply':p['ply'],'model':model,'predicted_uci':'e2e4' if correct else 'd2d4',
                                'target_logp':-1.,'predicted_logp':-1.,'top1_correct':int(correct),
                                'top5_correct':1,'pair_correct':int(correct),'entropy':1.})
                    store.save_game({'game_id':category,'category':category,'game_plies':2,'headers':dict(make_game().headers)},positions,predictions)
                config.output_dir.mkdir()
                summaries,contrasts,disagreements=summarize(store,config)
                selected={(r['scope'],r['model'],r['reference']):r for r in disagreements if r['test_category']=='classic'}
                self.assertEqual(selected['all','rapid','classic']['disagreement'],.5)
                self.assertEqual(selected['nonforced','rapid','classic']['disagreement'],1)
                self.assertEqual(selected['all','blitz','classic']['disagreement'],0)
                contrast=next(r for r in contrasts if r['scope']=='all' and r['test_category']=='classic' and r['model']=='rapid' and r['reference']=='classic')
                self.assertEqual(contrast['delta_agreement'],-.5)
                self.assertEqual(contrast['delta_agreement_ci_low'],-.5)
            finally:store.close()

    def test_run_lock_prevents_two_evaluators(self):
        with workspace_temp_dir() as tmp:
            with run_lock(Path(tmp)):
                with self.assertRaisesRegex(RuntimeError,'another process'):
                    with run_lock(Path(tmp)):pass

    def test_cli_resolves_artifacts_against_explicit_project_root(self):
        with workspace_temp_dir() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);fixture=make_fixture(root);path=root/'config.yaml'
            path.write_text(yaml.safe_dump({'corpus_dir':'corpus','prefix':'test','output_dir':'cli_output',
                'checkpoints':{c:f'{c}.pt' for c in CATEGORIES},'games_per_category':1,'bootstrap':10,'batch_size':2}))
            config=load_config(path,root);self.assertEqual(config.checkpoints['rapid'],root/'rapid.pt')
            args=['--config',str(path),'--project-root',str(root),'--resume','--device','cpu']
            with patch('analysis.time_control_comparison.__main__.load_models',side_effect=fake_models):main(args)
            self.assertTrue((root/'cli_output/README.md').exists())
            # Regeneration requires neither checkpoint files nor a GPU.
            for p in fixture.checkpoints.values():p.unlink()
            main(args+['--stage','summarize'])


if __name__ == '__main__':
    unittest.main()
