"""Audit frozen category-test games against saved policy data-plan identities.

The command validates frozen plan fingerprints against actual PGN records.
The frozen plans already store BLAKE2b-128 identities of complete games. Compare
the test games with the entire reservoirs (a conservative superset), as well as
the reconstructed training and checkpoint-selection subsets. Positive matches
are confirmed by reparsing the actual reservoir PGN record.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import chess.pgn
import numpy as np
import torch

from utils.balanced_training_plan import _game_fingerprint


ROOT = Path(__file__).resolve().parents[2]
NAMES = {"classic": "chess_encoder_seq_balanced_v1",
         "rapid": "chess_encoder_seq_balanced_rapid_v1",
         "blitz": "chess_encoder_seq_balanced_blitz_v1"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def identity(game):
    data = [game.board().fen(), [m.uci() for m in game.mainline_moves()]]
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_record(source, start, end):
    source.seek(int(start))
    game = chess.pgn.read_game(io.StringIO(source.read(int(end) - int(start)).decode("utf-8")))
    if game is None or game.errors:
        raise ValueError(f"invalid indexed PGN at byte {start}")
    return game


def local_path(recorded, root):
    """Map a saved corpus path into the current checkout without its old home directory."""
    parts = Path(str(recorded).replace("\\", "/")).parts
    return root.joinpath(*parts[parts.index("processed_games"):])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=Path("outputs/twic_1662_1667_time_controls"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/category_overlap_audit"))
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    test = root / args.source
    output = root / args.output_dir
    destination = output / "overlap_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    selection = json.loads((test / "selection_manifest.json").read_text())
    run = json.loads((test / "run_manifest.json").read_text())
    snapshot = test / "selected_games.pgn"
    if not (run["complete"] and sha(snapshot) == selection["selected_pgn_sha256"]):
        raise ValueError('Evaluation must be complete and its selected PGN hash must match.')
    target_games = {}
    first_test = None
    with snapshot.open("rb") as source:
        for row in selection["games"]:
            game = read_record(source, row["start"], row["end"])
            if not (identity(game) == row["game_id"]):
                raise ValueError('Selected game identity differs from the frozen manifest.')
            key = _game_fingerprint(game)
            if not (key not in target_games):
                raise ValueError('duplicate canonical test game')
            target_games[key] = row
            first_test = first_test or game
    if first_test is None:
        raise ValueError("The selected evaluation population is empty")
    # A metadata-only reprint must match the same canonical identity.
    original = _game_fingerprint(first_test)
    first_test.headers["Event"] = "Metadata-only positive control"
    if not (_game_fingerprint(first_test) == original):
        raise ValueError('Canonical identity changed after a metadata-only edit.')
    first_test.headers["FEN"] = "8/8/8/8/8/8/4K3/6k1 w - - 0 1"
    first_test.headers["SetUp"] = "1"
    if not (_game_fingerprint(first_test) != original):
        raise ValueError('Canonical identity ignored a changed initial position.')
    keys_hi = np.array([key[0] for key in target_games], dtype=np.uint64)
    result = {"test_games": len(target_games),
              "test_category_counts": dict(Counter(r["category"] for r in target_games.values())),
              "identity": "normalized initial FEN and full mainline UCI sequence; metadata ignored",
              "plan_fingerprint": "BLAKE2b-128, person=ch-game-bal-v1; existing frozen reservoir fingerprints",
              "test_identity_verified_against_frozen_selection": True,
              "metadata_and_initial_position_controls_passed": True,
              "models": {}, "input_sha256": {str(p): sha(p) for p in (
                  test / "selection_manifest.json", test / "run_manifest.json", snapshot, Path(__file__),
                  root / "utils/balanced_training_plan.py", root / "train/train_encoder_seq.py")}}
    all_overlaps = set()
    for category, name in NAMES.items():
        checkpoint = root / "checkpoints" / (name + ".best.pt")
        latest = root / "checkpoints" / (name + ".latest.pt")
        if not (sha(checkpoint) == run["identity"]["checkpoints"][category]):
            raise ValueError('Policy checkpoint hash differs from the evaluated weights.')
        state = torch.load(latest, map_location="cpu", weights_only=False)
        config = state["config"]
        if not (config["experiment_name"] == name):
            raise ValueError('Saved training configuration belongs to a different policy.')
        train_config, val_config = config["dataset_train"], config["dataset_val"]
        if not (local_path(train_config["balanced_plan_dir"], root) == local_path(val_config["balanced_plan_dir"], root)):
            raise ValueError('Training and validation use different balancing plans.')
        if not (local_path(train_config["pgn_path"], root) == local_path(val_config["pgn_path"], root)):
            raise ValueError('Training and validation use different PGN sources.')
        plan_dir = local_path(train_config["balanced_plan_dir"], root)
        plan = json.loads((plan_dir / "plan.json").read_text())
        report = json.loads((plan_dir / "report.json").read_text())
        source_path = local_path(train_config["pgn_path"], root)
        recorded = report["sources"]["reservoir"]
        if not (source_path.name == Path(recorded["path"]).name):
            raise ValueError('Reservoir source filename differs from its frozen report.')
        if not (source_path.stat().st_size == recorded["size_bytes"]):
            raise ValueError('Reservoir source size differs from its frozen report.')
        if not (source_path.stat().st_mtime_ns == recorded["mtime_ns"]):
            raise ValueError('Reservoir source timestamp differs from its frozen report.')
        reservoir = np.load(plan_dir / "reservoir_games.npy", mmap_mode="r")
        selected = np.load(plan_dir / "selected_games.npy", mmap_mode="r")
        if not (len(reservoir) == plan["reservoir"]["valid_games"]):
            raise ValueError('Reservoir array count differs from the plan.')
        if not (len(selected) == plan["budget"]["training_items"]):
            raise ValueError('Selected item count differs from the plan budget.')
        if not (len(np.unique(selected["game_id"])) == len(selected)):
            raise ValueError('The selected plan contains duplicate game IDs.')
        for field in ("pgn_start", "pgn_end", "ply_count"):
            if not (np.array_equal(selected[field], reservoir[field][selected["game_id"]])):
                raise ValueError('Selected game offsets or ply counts differ from the reservoir.')
        indices = np.arange(len(selected))
        np.random.default_rng(int(config["seed"])).shuffle(indices)
        n_val = int(round(float(config["data"]["val_fraction"]) * len(selected)))
        role = np.zeros(len(reservoir), dtype=np.uint8)
        role[selected["game_id"][indices[n_val:]]] = 1
        role[selected["game_id"][indices[:n_val]]] = 2
        candidates = np.flatnonzero(np.isin(reservoir["fingerprint_hi"], keys_hi))
        hits = [int(i) for i in candidates if (int(reservoir[i]["fingerprint_hi"]), int(reservoir[i]["fingerprint_lo"])) in target_games]
        audit_samples = sorted(set(np.random.default_rng(20261028).choice(len(reservoir), min(64, len(reservoir)), replace=False).tolist() + hits))
        confirmed = []
        with source_path.open("rb") as source:
            for i in audit_samples:
                row = reservoir[i]
                game = read_record(source, row["pgn_start"], row["pgn_end"])
                fingerprint = _game_fingerprint(game)
                if not (fingerprint == (int(row["fingerprint_hi"]), int(row["fingerprint_lo"]))):
                    raise ValueError('Stored fingerprint differs from the actual reservoir PGN record.')
                if i in hits:
                    target = target_games[fingerprint]
                    if not (identity(game) == target["game_id"]):
                        raise ValueError('An overlapping fingerprint failed full canonical identity verification.')
                    confirmed.append({"test_game_id": target["game_id"], "test_category": target["category"],
                                      "reservoir_game_id": i, "role": {0: "unselected_reservoir", 1: "training", 2: "checkpoint_selection"}[int(role[i])]})
                    all_overlaps.add(target["game_id"])
        counts = {kind: {c: len({h["test_game_id"] for h in confirmed if h["test_category"] == c and (kind == "entire_reservoir" or h["role"] == kind)})
                         for c in NAMES} for kind in ("entire_reservoir", "training", "checkpoint_selection")}
        result["models"][category] = {
            "checkpoint": str(checkpoint), "checkpoint_hash_matches_evaluation": True,
            "saved_checkpoint_configuration": {k: config[k] for k in ("experiment_name", "seed", "data", "dataset_train", "dataset_val")},
            "reservoir_games": len(reservoir), "training_plan_games": len(selected) - n_val,
            "checkpoint_selection_games": n_val, "entire_reservoir_is_conservative_superset": True,
            "source_size_and_mtime_match_frozen_plan": True,
            "fingerprint_source_record_checks": len(audit_samples), "overlap_counts": counts,
            "confirmed_matches": confirmed,
        }
        for path in (checkpoint, latest, source_path, plan_dir / "plan.json", plan_dir / "report.json",
                     plan_dir / "reservoir_games.npy", plan_dir / "selected_games.npy"):
            result["input_sha256"][str(path)] = sha(path)
        print(category, counts, flush=True)
    result["distinct_test_games_matching_any_policy_reservoir"] = len(all_overlaps)
    direct = {h["test_game_id"] for model in result["models"].values() for h in model["confirmed_matches"]
              if h["role"] in ("training", "checkpoint_selection")}
    result["distinct_test_games_matching_training_or_checkpoint_selection"] = len(direct)
    result["conclusion"] = (
        f"{len(direct)} test games match selected training or checkpoint-selection games; "
        f"{len(all_overlaps)} match the broader candidate reservoirs. Distinguish direct model exposure from candidate-pool membership."
    )
    output.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(result["conclusion"], flush=True)


if __name__ == "__main__":
    main()
