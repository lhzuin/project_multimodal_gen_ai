"""Legal history reconstruction and selection; solution moves never become inputs."""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import chess

from train.rl.contracts import PositionPrompt
from .config import BuildConfig


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def position_key(board: chess.Board) -> str:
    """Transposition identity, including side, castling and legal en passant.

    Draw counters/history stay in the actual prompt; ignoring them here makes
    deduplication and cross-split exclusion conservative rather than permissive.
    """
    return " ".join(board.fen().split()[:4])


def san_tokens(row: Mapping[str, Any]) -> list[str]:
    value = row.get("moves")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing original SAN game history")
    return value.split()


def game_fingerprint(row: Mapping[str, Any]) -> str:
    return digest("standard|" + " ".join(san_tokens(row)))


def game_id(row: Mapping[str, Any]) -> str:
    # Dataset GameId can contain '/black#50'. Those suffixes are puzzle links,
    # not distinct games, and must never influence partitioning.
    return str(row.get("GameId", "")).split("/")[0].split("#")[0]


@dataclass
class Candidate:
    prompt: PositionPrompt
    bucket: str
    board: chess.Board
    game_hash: str
    source_game_id: str
    audit: dict[str, Any]


def _candidate(
    row: Mapping[str, Any],
    bucket: str,
    board: chess.Board,
    history: list[str],
    audit: dict[str, Any],
) -> Candidate:
    puzzle_id = str(row["PuzzleId"])
    identifier = f"lichess-{puzzle_id}-{bucket}-ply-{len(history)}"
    source_id = game_id(row)
    # Deliberately omit solutions, SAN continuation, future evaluations, game
    # outcome and puzzle themes from everything passed to the trainer.
    prompt = PositionPrompt(
        identifier,
        moves_uci=tuple(history),
        metadata={
            "source": "lichess",
            "bucket": bucket,
            "game_id": source_id,
            "puzzle_id": puzzle_id,
            "selected_ply": len(history),
        },
    )
    return Candidate(
        prompt,
        bucket,
        board.copy(stack=False),
        game_fingerprint(row),
        source_id,
        {
            "entry_id": identifier,
            "bucket": bucket,
            "source_game_id": source_id,
            "puzzle_id": puzzle_id,
            "game_fingerprint": game_fingerprint(row),
            "position_key": position_key(board),
            "selected_fen": board.fen(),
            "selected_ply": len(history),
            "puzzle_rating": row.get("Rating"),
            "themes": list(row.get("Themes") or []),
            **audit,
        },
    )


def puzzle_candidate(row: Mapping[str, Any], bucket: str) -> Candidate:
    """Return the state after Moves[0], with only the original game prefix.

    FEN is BEFORE the trigger. Moves[1:] are checked for legality but never
    appended to the history, metadata, or audit output.
    """
    before = chess.Board(str(row["FEN"]))
    if not before.is_valid():
        raise ValueError("invalid puzzle FEN")
    line = str(row["Moves"]).split()
    if len(line) < 2:
        raise ValueError("missing puzzle trigger or solution")
    check = before.copy(stack=False)
    for uci in line:
        move = chess.Move.from_uci(uci)
        if move not in check.legal_moves:
            raise ValueError("illegal puzzle solution line")
        check.push(move)

    tokens = san_tokens(row)
    trigger_ply = before.ply()
    if trigger_ply >= len(tokens):
        raise ValueError("puzzle trigger is outside the original game")
    board = chess.Board()
    history = []
    for token in tokens[:trigger_ply]:
        move = board.parse_san(token)
        history.append(move.uci())
        board.push(move)
    if board.fen() != before.fen():
        raise ValueError("original history does not match the pre-trigger puzzle FEN")
    trigger = board.parse_san(tokens[trigger_ply])
    if trigger.uci() != line[0]:
        raise ValueError("original game trigger does not match Moves[0]")
    board.push(trigger)
    history.append(trigger.uci())
    return _candidate(
        row,
        bucket,
        board,
        history,
        {
            "pre_trigger_fen": before.fen(),
            "trigger_uci": trigger.uci(),
            "solution_plies_excluding_trigger": len(line) - 1,
            "alignment": "original_SAN_prefix_through_trigger_only",
        },
    )


def _evaluation(value: Any) -> dict[str, Any]:
    result = json.loads(value) if isinstance(value, str) else value
    return result if isinstance(result, dict) else {}


def quiet_candidate(row: Mapping[str, Any], config: BuildConfig) -> Candidate:
    """One stable, non-capture middlegame state, using existing annotations.

    Evaluation at analysis[p-1] describes the board after p played plies.
    No annotation, actual continuation, or original game outcome enters inputs.
    Quiet is an operational heuristic, not a proof of absence of tactics.
    """
    tokens = san_tokens(row)
    annotations = row.get("analysis") or []
    limit = min(config.quiet_max_ply, int(len(tokens) * 0.75), len(annotations))
    trigger_ply = chess.Board(str(row["FEN"])).ply() + 1
    board = chess.Board()
    history = []
    best = None
    best_priority = None
    previous_cp = None
    for ply, token in enumerate(tokens[:limit], start=1):
        move = board.parse_san(token)
        capture = board.is_capture(move)
        board.push(move)
        history.append(move.uci())
        evaluation = _evaluation(annotations[ply - 1])
        cp = evaluation.get("eval")
        cp_ok = isinstance(cp, (int, float)) and "mate" not in evaluation
        delta = abs(cp - previous_cp) if cp_ok and previous_cp is not None else None
        previous_cp = cp if cp_ok else None
        if (
            ply < config.quiet_min_ply
            or abs(ply - trigger_ply) <= 10
            or capture
            or move.promotion
            or board.is_check()
            or len(board.piece_map()) < 14
            or board.legal_moves.count() < 12
            or not cp_ok
            or abs(cp) > config.quiet_max_abs_cp
            or delta is None
            or delta > config.quiet_max_delta_cp
            or evaluation.get("judgment")
            or board.is_game_over(claim_draw=True)
        ):
            continue
        priority = digest(f"{config.seed}|quiet|{game_id(row)}|{ply}")
        if best_priority is None or priority < best_priority:
            best_priority = priority
            best = _candidate(
                row,
                "quiet",
                board,
                history,
                {
                    "existing_annotation_cp_white": cp,
                    "previous_annotation_delta_cp": delta,
                    "alignment": "original_SAN_prefix_at_quiet_position",
                },
            )
    if best is None:
        raise ValueError("no eligible quiet position")
    return best
