import time
import torch
import torch.nn.functional as F
from stockfish_teacher import StockfishUCI, EncoderStockfishTeacher

engine = StockfishUCI(
    engine_path="E:\\Ariel\\stockfish\\stockfish-windows-x86-64-avx2.exe",
    threads=1,
    hash_mb=256,
    multipv=5,
)

teacher = EncoderStockfishTeacher(
    engine=engine,
    tau=50.0,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

fen = "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

t0 = time.perf_counter()
p64, p_promo, dbg = teacher.teacher_distribution(fen, k=5, movetime_ms=100)
t1 = time.perf_counter()

print("p64 shape:", p64.shape)  # [64, 64]
print("p_promo shape:", p_promo.shape if p_promo is not None else "None")  # [4] or None
print("Top-5 moves info (uci, cp, mate):")
print("dbg:", dbg)  # dbg = [(uci, cp, mate), ...]

print(f"\nStockfish teacher_distribution time: {(t1 - t0)*1000:.2f} ms")

# ===== Save p64 to txt file =====
output_path = "p64_output.txt"

t2 = time.perf_counter()
with open(output_path, "w") as f:
    f.write(f"FEN: {fen}\n")
    f.write(f"p64 shape: {tuple(p64.shape)}\n\n")

    # Write full matrix
    f.write("=== FULL 64x64 MATRIX ===\n")
    p64_cpu = p64.detach().cpu()

    for i in range(64):
        row_vals = " ".join(f"{float(x):.6f}" for x in p64_cpu[i])
        f.write(row_vals + "\n")

    nz = (p64_cpu > 0).nonzero(as_tuple=False)

    # Write only non-zero entries (much easier to inspect)
    def idx_to_sq(idx):
        file = idx % 8
        rank = idx // 8
        return chr(ord('a') + file) + str(rank + 1)

    f.write("\n=== NON-ZERO ENTRIES (square -> square) ===\n")
    for idx in nz:
        from_sq = int(idx[0])
        to_sq = int(idx[1])
        prob = float(p64_cpu[from_sq, to_sq])
        f.write(f"{idx_to_sq(from_sq)} -> {idx_to_sq(to_sq)} : {prob:.6f}\n")

t3 = time.perf_counter()

print(f"\nSaved p64 to {output_path}")
print(f"File write time: {(t3 - t2)*1000:.2f} ms")
print(f"Total time (teacher + write): {(t3 - t0)*1000:.2f} ms\n")

# Pretty print dbg
for move, cp, mate in dbg:
    cp_str = str(cp) if cp is not None else "None"
    mate_str = str(mate) if mate is not None else "continue playing"
    print(f"Move: {move}, cp: {cp_str}, Mate in: {mate_str}")

# (Optional) if you loop over many FENs later, you can also estimate ETA like:
# avg_ms = running_total_ms / n_done
# eta_s = (n_total - n_done) * (avg_ms / 1000.0)