import os
import time
import json
import torch
import torch.nn.functional as F

from stockfish_teacher import StockfishUCI, DecoderStockfishTeacher


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# 1) Load vocab -> move2id
with open("E:\\Ariel\\project_multimodal_gen_ai\\tokenizers\\chess_uci_vocab.json", "r") as f:
    vocab = json.load(f)

id2move = vocab["id2move"]
move2id = {m: i for i, m in enumerate(id2move)}
vocab_size = len(id2move)

print(f"Loaded vocab_size={vocab_size}")

# 2) Start engine
engine = StockfishUCI(
    engine_path="E:\\Ariel\\stockfish\\stockfish-windows-x86-64-avx2.exe",
    threads=1,
    hash_mb=256,
    multipv=5,
)

# 3) Teacher
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = DecoderStockfishTeacher(
    engine=engine,
    move2id=move2id,
    vocab_size=vocab_size,
    tau=50.0,
    device=device,
)

fen = "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

# ===== Run teacher with timing =====
t0 = time.perf_counter()
pT, dbg = teacher.teacher_distribution(fen, k=5, movetime_ms=100)
t1 = time.perf_counter()

print("\nTeacher output:")
print("pT shape:", tuple(pT.shape))  # [V]
print("dbg (uci, cp, mate):", dbg)
print(f"Stockfish teacher_distribution time: {(t1 - t0) * 1000:.2f} ms")

# ===== Extract top teacher moves (non-zero mass) =====
pT_cpu = pT.detach().cpu()
nonzero_ids = (pT_cpu > 0).nonzero(as_tuple=False).view(-1).tolist()

top_moves = []
for idx in nonzero_ids:
    move = id2move[idx]
    prob = float(pT_cpu[idx])
    top_moves.append((move, prob))

top_moves.sort(key=lambda x: x[1], reverse=True)

print("\nTop teacher moves (from pT non-zero mass):")
for move, prob in top_moves:
    print(f"Move: {move} | P={prob:.6f}")

# ===== Save pT to txt file =====
# Put outputs next to the script if possible; otherwise current working directory
out_dir = os.path.join(os.getcwd(), "teacher_outputs")
ensure_dir(out_dir)
output_path = os.path.join(out_dir, "decoder_pT_output.txt")

t2 = time.perf_counter()
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"FEN: {fen}\n")
    f.write(f"vocab_size: {vocab_size}\n")
    f.write(f"pT shape: {tuple(pT.shape)}\n\n")

    f.write("=== STOCKFISH RAW TOP-K (uci, cp, mate) ===\n")
    for uci, cp, mate in dbg:
        cp_str = str(cp) if cp is not None else "None"
        mate_str = str(mate) if mate is not None else "None"
        f.write(f"{uci} | cp={cp_str} | mate={mate_str}\n")

    f.write("\n=== TEACHER DISTRIBUTION: NON-ZERO ENTRIES ===\n")
    for move, prob in top_moves:
        f.write(f"{move} : {prob:.8f}\n")

    # Optional: write full vector (large!)
    f.write("\n=== FULL pT VECTOR (index : prob) ===\n")
    for i in range(vocab_size):
        f.write(f"{i} : {float(pT_cpu[i]):.10f}\n")

t3 = time.perf_counter()

print(f"\nSaved decoder pT to {output_path}")
print(f"File write time: {(t3 - t2) * 1000:.2f} ms")
print(f"Total time (teacher + write): {(t3 - t0) * 1000:.2f} ms\n")

# ===== Example loss (when you have student logits) =====
# logits = model(...).logits[:, -1, :]  # [B,V]
# pS = torch.softmax(logits, dim=-1)
# loss = F.kl_div(pS.log(), pT.unsqueeze(0).expand_as(pS), reduction="batchmean")