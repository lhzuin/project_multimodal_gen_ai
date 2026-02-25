# stockfish_teacher.py
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class PVLine:
    """One multipv line from Stockfish."""
    multipv: int
    uci: str               # first move of PV
    cp: Optional[int]      # centipawns from side-to-move POV (Stockfish's usual convention)
    mate: Optional[int]    # mate in N if present
    depth: int


def _softmax_from_scores(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """Numerically stable softmax(scores / tau)."""
    tau = max(float(tau), 1e-6)
    x = scores / tau
    x = x - x.max()
    return torch.softmax(x, dim=-1)


def _score_to_scalar(cp: Optional[int], mate: Optional[int], mate_cp: int = 20000) -> float:
    """
    Convert Stockfish score to a single scalar for softmax.
    - cp: use directly.
    - mate: map mate-in-N to large +/- value (sign matters).
    """
    if mate is not None:
        # mate > 0 means side to move mates, mate < 0 means side to move gets mated
        return float(mate_cp if mate > 0 else -mate_cp)
    if cp is None:
        return 0.0
    return float(cp)


def _uci_to_from_to_promo(uci: str) -> Tuple[int, int, Optional[str]]:
    """
    Convert UCI like 'e2e4' or 'e7e8q' to (from_idx, to_idx, promo).
    Indexing: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63.
    """
    if len(uci) < 4:
        raise ValueError(f"Bad UCI: {uci}")
    f = uci[0:2]
    t = uci[2:4]
    promo = uci[4].lower() if len(uci) >= 5 else None

    def sq_to_idx(sq: str) -> int:
        file = ord(sq[0]) - ord('a')  # 0..7
        rank = int(sq[1]) - 1         # 0..7
        if not (0 <= file <= 7 and 0 <= rank <= 7):
            raise ValueError(f"Bad square: {sq}")
        return rank * 8 + file

    return sq_to_idx(f), sq_to_idx(t), promo


class StockfishUCI:
    """
    Minimal Stockfish UCI driver:
    - supports setting options (MultiPV, Threads, Hash, etc.)
    - supports analyze_fen() returning top-K PVLines with cp/mate
    """

    _re_multipv = re.compile(r"\bmultipv\s+(\d+)\b")
    _re_depth = re.compile(r"\bdepth\s+(\d+)\b")
    _re_score_cp = re.compile(r"\bscore\s+cp\s+(-?\d+)\b")
    _re_score_mate = re.compile(r"\bscore\s+mate\s+(-?\d+)\b")
    _re_pv = re.compile(r"\bpv\s+([a-h][1-8][a-h][1-8][qrbn]?)\b")

    def __init__(
        self,
        engine_path: str,
        *,
        threads: int = 1,
        hash_mb: int = 256,
        multipv: int = 5,
        uci_elo: Optional[int] = None,
        limit_strength: bool = False,
        startup_timeout_s: float = 5.0,
    ) -> None:
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Stockfish not found: {engine_path}")

        self.engine_path = engine_path
        self.proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        self._send("uci")
        self._wait_for("uciok", timeout_s=startup_timeout_s)

        # Options
        self.setoption("Threads", str(int(threads)))
        self.setoption("Hash", str(int(hash_mb)))
        self.setoption("MultiPV", str(int(multipv)))

        if limit_strength:
            self.setoption("UCI_LimitStrength", "true")
        if uci_elo is not None:
            self.setoption("UCI_Elo", str(int(uci_elo)))

        self._send("isready")
        self._wait_for("readyok", timeout_s=startup_timeout_s)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send("quit")
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _send(self, line: str) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError("Stockfish process is not running.")
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout_s: float) -> None:
        t0 = time.time()
        assert self.proc.stdout is not None
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Timed out waiting for '{token}'")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish output ended unexpectedly.")
            if token in line:
                return

    def setoption(self, name: str, value: str) -> None:
        self._send(f"setoption name {name} value {value}")

    def analyze_fen(
        self,
        fen: str,
        *,
        movetime_ms: int = 100,
        multipv: int = 5,
        min_depth: int = 1,
        read_timeout_s: float = 5.0,
    ) -> List[PVLine]:
        """
        Returns up to `multipv` PV lines (best first), taking the latest info per multipv.
        """
        self.setoption("MultiPV", str(int(multipv)))
        self._send(f"position fen {fen}")
        self._send(f"go movetime {int(movetime_ms)}")

        latest: Dict[int, PVLine] = {}
        t0 = time.time()
        assert self.proc.stdout is not None

        while True:
            if time.time() - t0 > read_timeout_s:
                # If the engine is slow / busy, we still return what we have
                break

            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()

            if line.startswith("bestmove"):
                break

            if " info " not in (" " + line + " "):
                continue

            m_mpv = self._re_multipv.search(line)
            m_depth = self._re_depth.search(line)
            m_pv = self._re_pv.search(line)

            if not (m_mpv and m_depth and m_pv):
                continue

            mpv = int(m_mpv.group(1))
            depth = int(m_depth.group(1))
            if depth < min_depth:
                continue

            cp = None
            mate = None
            m_cp = self._re_score_cp.search(line)
            m_mate = self._re_score_mate.search(line)
            if m_cp:
                cp = int(m_cp.group(1))
            if m_mate:
                mate = int(m_mate.group(1))

            uci = m_pv.group(1)

            latest[mpv] = PVLine(multipv=mpv, uci=uci, cp=cp, mate=mate, depth=depth)

        # Sort by multipv (1 is best)
        out = [latest[k] for k in sorted(latest.keys()) if 1 <= k <= multipv]
        return out


# ==========================
# Teacher: Decoder (vocab)
# ==========================

class DecoderStockfishTeacher:
    """
    Idea A (decoder):
    - Get top-5 UCI moves + cp from Stockfish
    - Map UCIs -> vocab ids (your tokenizer/vocab JSON)
    - Build a sparse teacher distribution over the full vocab
    """

    def __init__(
        self,
        engine: StockfishUCI,
        *,
        move2id: Dict[str, int],
        vocab_size: int,
        tau: float = 50.0,
        mate_cp: int = 20000,
        device: Optional[torch.device] = None,
    ) -> None:
        self.engine = engine
        self.move2id = move2id
        self.vocab_size = int(vocab_size)
        self.tau = float(tau)
        self.mate_cp = int(mate_cp)
        self.device = device

    @torch.no_grad()
    def teacher_distribution(
        self,
        fen: str,
        *,
        k: int = 5,
        movetime_ms: int = 100,
    ) -> Tuple[torch.Tensor, List[Tuple[str, Optional[int], Optional[int]]]]:
        """
        Returns:
          - p_teacher: [vocab_size] float tensor summing to 1
          - debug_moves: list of (uci, cp, mate) in Stockfish order
        """
        lines = self.engine.analyze_fen(fen, movetime_ms=movetime_ms, multipv=k)
        if len(lines) == 0:
            p = torch.full((self.vocab_size,), 1.0 / self.vocab_size, device=self.device)
            return p, []

        # Keep only moves that exist in vocab
        kept: List[PVLine] = [ln for ln in lines if ln.uci in self.move2id]
        if len(kept) == 0:
            p = torch.full((self.vocab_size,), 1.0 / self.vocab_size, device=self.device)
            dbg = [(ln.uci, ln.cp, ln.mate) for ln in lines]
            return p, dbg

        scores = torch.tensor(
            [_score_to_scalar(ln.cp, ln.mate, mate_cp=self.mate_cp) for ln in kept],
            dtype=torch.float32,
            device=self.device,
        )
        probs = _softmax_from_scores(scores, tau=self.tau)  # [k']

        p_teacher = torch.zeros((self.vocab_size,), dtype=torch.float32, device=self.device)
        for ln, pr in zip(kept, probs):
            p_teacher[self.move2id[ln.uci]] = pr

        dbg = [(ln.uci, ln.cp, ln.mate) for ln in lines]
        return p_teacher, dbg


# ==========================
# Teacher: Encoder (64x64)
# ==========================

class EncoderStockfishTeacher:
    """
    Idea A (encoder):
    - Get top-5 UCI moves + cp from Stockfish
    - Convert UCI -> (from_idx, to_idx, promo)
    - Build a sparse teacher distribution over [64,64]
    - Optionally return a promotion target distribution (for your promo head)
    """

    PROMO_TO_ID = {"n": 0, "b": 1, "r": 2, "q": 3}  # you can change to match your dataset convention

    def __init__(
        self,
        engine: StockfishUCI,
        *,
        tau: float = 50.0,
        mate_cp: int = 20000,
        device: Optional[torch.device] = None,
    ) -> None:
        self.engine = engine
        self.tau = float(tau)
        self.mate_cp = int(mate_cp)
        self.device = device

    @torch.no_grad()
    def teacher_distribution(
        self,
        fen: str,
        *,
        k: int = 5,
        movetime_ms: int = 100,
        return_promo: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[str, Optional[int], Optional[int]]]]:
        """
        Returns:
          - p_teacher_64: [64,64] float tensor summing to 1
          - p_promo (optional): [4] distribution over {n,b,r,q} if at least one top move is a promotion, else None
          - debug_moves: list of (uci, cp, mate) in Stockfish order
        """
        lines = self.engine.analyze_fen(fen, movetime_ms=movetime_ms, multipv=k)
        if len(lines) == 0:
            p64 = torch.full((64, 64), 1.0 / (64 * 64), device=self.device)
            return p64, None, []

        # Parse UCIs; skip malformed
        parsed = []
        for ln in lines:
            try:
                f, t, promo = _uci_to_from_to_promo(ln.uci)
                parsed.append((ln, f, t, promo))
            except Exception:
                continue

        if len(parsed) == 0:
            p64 = torch.full((64, 64), 1.0 / (64 * 64), device=self.device)
            dbg = [(ln.uci, ln.cp, ln.mate) for ln in lines]
            return p64, None, dbg

        scores = torch.tensor(
            [_score_to_scalar(ln.cp, ln.mate, mate_cp=self.mate_cp) for (ln, _, _, _) in parsed],
            dtype=torch.float32,
            device=self.device,
        )
        probs = _softmax_from_scores(scores, tau=self.tau)  # [k']

        p64 = torch.zeros((64, 64), dtype=torch.float32, device=self.device)
        promo_mass = torch.zeros((4,), dtype=torch.float32, device=self.device)

        for (ln, f, t, promo), pr in zip(parsed, probs):
            p64[f, t] += pr
            if return_promo and promo in self.PROMO_TO_ID:
                promo_mass[self.PROMO_TO_ID[promo]] += pr

        # Normalize promotion distribution if any promo happened
        p_promo = None
        if return_promo and float(promo_mass.sum()) > 0:
            p_promo = promo_mass / promo_mass.sum()

        dbg = [(ln.uci, ln.cp, ln.mate) for ln in lines]
        return p64, p_promo, dbg