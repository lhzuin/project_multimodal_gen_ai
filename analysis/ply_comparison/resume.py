"""Validate and recover complete games from a frozen partial per-move CSV.

Export provenance on the source computer after stopping or snapshotting its CSV:
python -m analysis.ply_comparison.resume --csv outputs/twic_1662_1667/per_move.csv
"""
from collections import Counter
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import chess
import chess.pgn

from .aggregate import bucket_for_ply, relative_bucket_for_ply
from .plot_results import MODELS

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = ('analysis/ply_comparison/evaluate.py', 'analysis/ply_comparison/aggregate.py',
              'dataset/offsets.py', 'dataset/renderer.py', 'models/chess_decoder_player.py',
              'models/chess_encoder_player.py', 'models/chess_llm.py', 'models/chess_tokenizer.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def text_hashes(data):
    """Only universal-newline variants, with no content/whitespace removal."""
    lf = data.replace(b'\r\n', b'\n')
    return {hashlib.sha256(value).hexdigest() for value in (data, lf, lf.replace(b'\n', b'\r\n'))}


def verify_source(csv_path, proof_path, paths, *, code_review=None):
    proof = json.loads(Path(proof_path).read_text())
    if proof.get('format') != 'twic_resume_source_v1':
        raise ValueError('use the source provenance exporter, not an unrelated manifest')
    if sha(csv_path) != proof['csv']['sha256']:
        raise ValueError('imported CSV hash differs from its frozen source snapshot')
    newline_equivalent = []
    reviewed_files = []
    review = json.loads(Path(code_review).read_text()) if code_review else None
    for name, path in paths.items():
        if sha(path) != proof['files'][name]['sha256']:
            if name == 'tokenizer' and proof['files'][name]['sha256'] in text_hashes(Path(path).read_bytes()):
                newline_equivalent.append(name)
            else:
                raise ValueError(f'source/local input mismatch: {name}')
    for name in CODE_FILES:
        accepted = text_hashes((ROOT / name).read_bytes())
        original = subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT)
        accepted.update(text_hashes(original))
        if proof['code_sha256'].get(name) not in accepted:
            entry = review.get('files', {}).get(name) if review and review.get('status') == 'passed' else None
            if not (entry and entry['source_sha256'] == proof['code_sha256'][name]
                    and entry['reference_sha256'] == sha(ROOT / name)
                    and sha(entry['source_file']) == entry['source_sha256']):
                raise ValueError(f'source evaluation code differs: {name}')
            reviewed_files.append(name)
        if name not in reviewed_files and proof['code_sha256'][name] not in {sha(ROOT / name), hashlib.sha256(original).hexdigest()}:
            newline_equivalent.append(name)
    proof['verification'] = {'csv_exact_checksum_match': True,
                             'text_files_matching_only_after_line_ending_conversion': newline_equivalent,
                             'separately_reviewed_source_files': reviewed_files,
                             'code_review': review if reviewed_files else None}
    return proof


def recover(csv_path, *, pgn_path, index, fields, writer, accumulator, relative_accumulator,
            skip_game_ids=frozenset(), progress=None):
    """Copy verified complete games and rebuild all aggregate/provenance state.

    Only an incomplete final game can be omitted. Corrupt rows, duplicate keys,
    mismatched positions and incomplete interior games are hard errors.
    """
    offsets = {int(r['idx']): r for r in index['games']}
    recovered = set()
    validated = set()
    issue_counts, date_counts = Counter(), Counter()
    early_hashes, early_occurrences = set(), 0
    discarded = []

    def commit(game_id, rows, source, final=False, truncated=False):
        nonlocal early_occurrences
        if game_id not in offsets:
            raise ValueError(f'imported game ID absent from corpus: {game_id}')
        offset = offsets[game_id]
        source.seek(int(offset['start']))
        raw = source.read(int(offset['end']) - int(offset['start']))
        game = chess.pgn.read_game(io.StringIO(raw.decode('utf-8', errors='replace')))
        if game is None or game.errors:
            raise ValueError(f'cannot validate imported game {game_id}')
        moves = list(game.mainline_moves())
        keys = set()
        for r in rows:
            key = (r['model_id'], int(r['ply']))
            if key in keys or key[0] not in MODELS or not 0 <= key[1] < len(moves):
                raise ValueError(f'duplicate or invalid imported decision: {game_id}, {key}')
            keys.add(key)
        expected = {(m, p) for m in MODELS for p in range(len(moves))}
        if keys != expected or truncated:
            if final:
                discarded.append(game_id)
                return
            raise ValueError(f'incomplete interior game: {game_id}')
        boards = []
        board = game.board()
        for move in moves:
            if move not in board.legal_moves:
                raise ValueError(f'illegal corpus move in imported game {game_id}')
            boards.append((board.fen(), {m.uci() for m in board.legal_moves}))
            board.push(move)
        for r in rows:
            ply = int(r['ply'])
            fen, legal = boards[ply]
            if (int(r['game_plies']) != len(moves) or r['fen_before'] != fen
                    or r['target_uci'] != moves[ply].uci() or r['predicted_uci'] not in legal
                    or int(r['n_legal']) != len(legal)
                    or r['bucket'] != bucket_for_ply(ply)
                    or r['relative_bucket'] != relative_bucket_for_ply(ply, len(moves))
                    or int(r['twic_issue']) != int(game.headers['TWICIssue'])
                    or r['date'] != game.headers.get('Date', '')
                    or int(r['white_elo']) != int(game.headers['WhiteElo'])
                    or int(r['black_elo']) != int(game.headers['BlackElo'])):
                raise ValueError(f'CSV/corpus decision mismatch: game {game_id}, ply {ply}')
            top1, top5, nll = int(r['top1_correct']), int(r['top5_correct']), float(r['nll_legal'])
            if (top1 not in (0, 1) or top5 not in (0, 1) or top5 < top1
                    or top1 != int(r['predicted_uci'] == r['target_uci'])
                    or not math.isfinite(nll) or nll < 0
                    or not math.isclose(float(r['p_target_legal']), math.exp(-nll), rel_tol=1e-6, abs_tol=1e-12)):
                raise ValueError(f'invalid imported metrics: game {game_id}, ply {ply}')
        # Validate the whole game before copying or crediting any of its rows.
        validated.add(game_id)
        if progress and len(validated) % 500 == 0:
            progress(len(validated), game_id)
        # Earlier sources own all four models for overlapping games. Never
        # choose individual rows using their scores or mix duplicate attempts.
        if game_id in skip_game_ids:
            return
        for r in rows:
            ply, model = int(r['ply']), r['model_id']
            kw = dict(top1=bool(int(r['top1_correct'])), top5=bool(int(r['top5_correct'])), nll=float(r['nll_legal']))
            accumulator.add(game_id, ply, model, **kw)
            relative_accumulator.add_bucket(game_id, r['relative_bucket'], model, **kw)
            writer.writerow(r)
        for ply, (fen, _) in enumerate(boards[:40]):
            early_occurrences += 1
            early_hashes.add(hashlib.blake2b(' '.join(fen.split()[:4]).encode('ascii'), digest_size=16).digest())
        recovered.add(game_id)
        issue_counts[int(game.headers['TWICIssue'])] += 1
        date_counts[game.headers.get('Date', '')] += 1

    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f, Path(pgn_path).open('rb') as source:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(fields):
            raise ValueError('source CSV columns/order differ from this evaluator')
        game_id, rows, truncated = None, [], False
        for r in reader:
            if None in r or any(v is None for v in r.values()):
                if next(reader, None) is not None:
                    raise ValueError('malformed interior CSV row')
                truncated = True
                break
            incoming = int(r['game_id'])
            if game_id is not None and incoming != game_id:
                if incoming <= game_id:
                    raise ValueError('CSV game groups are duplicated or out of order')
                commit(game_id, rows, source)
                rows = []
            game_id = incoming
            rows.append(r)
        if game_id is not None:
            commit(game_id, rows, source, final=True, truncated=truncated)
    return {'game_ids': recovered, 'validated_game_ids': validated,
            'issue_counts': issue_counts, 'date_counts': date_counts,
            'early_hashes': early_hashes, 'early_occurrences': early_occurrences,
            'discarded_final_games': discarded}


def main():
    from .evaluate import CHECKPOINTS
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-device', default='unknown')
    args = parser.parse_args()
    output = args.output or args.csv.with_name('resume_source.json')
    if output.exists():
        raise FileExistsError(output)
    paths = {'pgn': ROOT / 'processed_games/eval/twic_1662_1667_classic.pgn',
             'tokenizer': ROOT / 'tokenizers/chess_uci_vocab.json', **CHECKPOINTS}
    before = args.csv.stat()
    csv_sha = sha(args.csv)
    after = args.csv.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('CSV changed during hashing; stop the run or export from a stable copy')
    result = {'format': 'twic_resume_source_v1',
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_device': args.source_device,
              'python_version': sys.version, 'torch_version': torch.__version__,
              'chess_version': chess.__version__,
              'csv': {'path': str(args.csv.resolve()), 'size': after.st_size, 'sha256': csv_sha},
              'files': {name: {'path': str(path.resolve()), 'sha256': sha(path)} for name, path in paths.items()},
              'code_sha256': {name: sha(ROOT / name) for name in CODE_FILES}}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote source provenance: {output}')


if __name__ == '__main__':
    main()
