# train.py
import os, sys, signal, random
import numpy as np
import torch
import wandb
import hydra
import chess
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from transformers.optimization import get_scheduler
from tqdm.auto import tqdm

os.environ["HYDRA_FULL_ERROR"] = "1"

OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_scheduler_for_stage(optimizer, *, stage_dict, cfg, num_training_steps):
    scheduler_type = stage_dict["lr_scheduler"]
    warmup_fraction = stage_dict["warmup_fraction"]
    use_warmup = getattr(cfg, "use_warmup", True)
    num_warmup_steps = int(warmup_fraction * num_training_steps) if use_warmup else 0

    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def param_groups_for(model, cfg, stage_dict):
    groups = model.get_param_groups()
    stage_groups = stage_dict.get("groups", {})
    optim_cfg = cfg.optim

    def get_lr(key, fallback="lr"):
        if hasattr(optim_cfg, key):
            return getattr(optim_cfg, key)
        if hasattr(optim_cfg, fallback):
            return getattr(optim_cfg, fallback)
        raise ValueError(f"Could not find LR '{key}' or '{fallback}' in cfg.optim")

    pg = []
    for group_name, lr_key in stage_groups.items():
        if group_name not in groups:
            continue
        params = [p for p in groups[group_name] if p.requires_grad]
        if not params:
            continue
        lr = get_lr(lr_key)
        weight_decay = cfg.optim.weight_decay
        pg.append({"params": params, "lr": lr, "weight_decay": weight_decay})
    return pg


def split_indices(n: int, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(val_fraction * n))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return train_idx, val_idx


def batch_is_empty(batch: dict) -> bool:
    if not isinstance(batch, dict):
        return True
    if "piece_probs" in batch and batch["piece_probs"].numel() > 0:
        return False
    if "images" in batch and batch["images"].numel() > 0:
        return False
    return True


def build_legal_mask_from_fen(fen_list):
    """
    fen_list: list[str] length B
    returns mask [B,64,64] bool where mask[b,from,to]=True iff legal
    """
    B = len(fen_list)
    mask = torch.zeros((B, 64, 64), dtype=torch.bool)
    for i, fen in enumerate(fen_list):
        board = chess.Board(fen)
        for mv in board.legal_moves:
            mask[i, mv.from_square, mv.to_square] = True
    return mask

def apply_legal_mask(policy_logits, legal_mask):
    # policy_logits: [B,64,64]
    # legal_mask:    [B,64,64] bool
    neg_inf = torch.finfo(policy_logits.dtype).min
    return policy_logits.masked_fill(~legal_mask.to(policy_logits.device), neg_inf)


def move_metrics(from_logits, to_logits, move_from, move_to):
    # accuracy on from/to separately
    from_pred = from_logits.argmax(dim=-1)
    to_pred = to_logits.argmax(dim=-1)
    acc_from = (from_pred == move_from).float().mean().item()
    acc_to = (to_pred == move_to).float().mean().item()
    return acc_from, acc_to


class Trainer:
    def __init__(self, cfg, logger, device):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.model = None
        self.loss_fn = None
        self.scaler = GradScaler(device="cuda") if device.type == "cuda" else None

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.optimizer = None

        self.patience = cfg.early_stopping.patience
        self.min_epochs = cfg.early_stopping.min_epochs
        self.checkpoint_save_path = cfg.checkpoint_save_path
        self.acc_epsilon = cfg.acc_epsilon

        # Build opt cfg like before (keep only AdamW-safe args)
        opt_cfg_full = OmegaConf.to_container(cfg.optim, resolve=True, enum_to_str=True)
        target = opt_cfg_full.pop("_target_")
        allowed_keys = {
            "lr", "betas", "eps", "weight_decay",
            "amsgrad", "foreach", "maximize",
            "capturable", "differentiable", "fused",
        }
        opt_kwargs = {k: v for k, v in opt_cfg_full.items() if k in allowed_keys}
        opt_kwargs["_target_"] = target
        self.opt_cfg = opt_kwargs

        self.grad_accum_steps = int(getattr(cfg, "trainer", {}).get("grad_accum_steps", 1))
        self.grad_clip_norm = float(getattr(cfg, "trainer", {}).get("grad_clip_norm", 1.0))

    def init_data(self):
        # Dataset + collate from Hydra
        self.train_dataset = hydra.utils.instantiate(self.cfg.dataset_train)
        self.val_dataset   = hydra.utils.instantiate(self.cfg.dataset_val)
        collate_fn = hydra.utils.instantiate(self.cfg.collate)


        # Split into train/val
        val_fraction = float(getattr(self.cfg.data, "val_fraction", 0.05))
        train_idx, val_idx = split_indices(len(self.train_dataset), val_fraction, seed=int(self.cfg.seed))

        train_ds = Subset(self.train_dataset, train_idx)
        val_ds   = Subset(self.val_dataset, val_idx)


        dl_cfg = self.cfg.dataloader
        self.train_loader = DataLoader(
            train_ds,
            batch_size=dl_cfg.batch_size,
            shuffle=dl_cfg.shuffle,
            num_workers=dl_cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=dl_cfg.pin_memory,
            persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
            drop_last=getattr(dl_cfg, "drop_last", False),
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=dl_cfg.batch_size,
            shuffle=False,
            num_workers=dl_cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=dl_cfg.pin_memory,
            persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
            drop_last=False,
        )

        print(f"Dataset size={len(self.train_dataset)} | train={len(train_ds)} | val={len(val_ds)}")

    def init_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model.instance).to(self.device)
        self.loss_fn = hydra.utils.instantiate(self.cfg.loss_fn)

    
    def _forward_loss_and_metrics(self, batch):
        move_from = batch["move_from"].to(self.device, non_blocking=True)
        move_to = batch["move_to"].to(self.device, non_blocking=True)
        move_promo = batch["move_promo"].to(self.device, non_blocking=True)

        train_input = getattr(self.cfg, "train_input", "image")

        turn = batch["turn"].to(self.device, non_blocking=True)
        castling = batch["castling"].to(self.device, non_blocking=True)
        ep_square = batch["ep_square"].to(self.device, non_blocking=True)

        if train_input == "pieces":
            piece_probs = batch["piece_probs"].to(self.device, non_blocking=True)  # [B,64,13]
            out = self.model(
                piece_probs=piece_probs,
                turn=turn,
                castling=castling,
                ep_square=ep_square,
            )
        else:
            images = batch["images"].to(self.device, non_blocking=True)
            out = self.model(
                images=images,
                turn=turn,
                castling=castling,
                ep_square=ep_square,
            )


        policy_logits = out["policy_logits"]  # [B,64,64]
        promo_logits = out["promo_logits"]    # [B,5]
        B = policy_logits.size(0)

        # --- legal masking during training ---
        if getattr(self.cfg, "use_legal_mask", True):
            fen_list = batch["fen"]  # list[str]
            legal_mask = self.model.legal_mask_from_fen(fen_list, device=policy_logits.device)
            policy_logits = self.model.apply_legal_mask(policy_logits, legal_mask)

        flat_logits = policy_logits.view(B, 4096)
        target = move_from * 64 + move_to  # [B]

        loss_policy = F.cross_entropy(flat_logits, target)
        loss_promo = F.cross_entropy(promo_logits, move_promo)
        loss = loss_policy + loss_promo
        if not torch.isfinite(loss):
            # Skip this batch: don't backward, don't step
            return None

        pred = flat_logits.argmax(dim=-1)
        acc = (pred == target).float().mean().item()
        n = B
        return loss, acc, n

    def run_epoch(self, stage_name, scheduler):
        self.model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        accum = 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(self.train_loader, desc=f"{stage_name} | training", leave=False)
        for batch in progress:
            if batch_is_empty(batch):
                continue

            if self.scaler:
                with autocast(device_type=self.device.type):
                    res = self._forward_loss_and_metrics(batch)
                    if res is None:
                        continue
                    loss, correct, n = res
                    loss_scaled = loss / self.grad_accum_steps

                self.scaler.scale(loss_scaled).backward()
            else:
                res = self._forward_loss_and_metrics(batch)
                if res is None:
                    continue
                loss, correct, n = res
                (loss / self.grad_accum_steps).backward()

            accum += 1

            # --- logging stays per *micro-batch* exactly like before ---
            loss_val = float(loss.detach().cpu().item())
            total_loss += loss_val * n
            total_correct += correct
            total_n += n

            acc_val = correct / max(1, n)
            progress.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.3f}")

            if self.logger:
                wandb.log({
                    f"{stage_name}/loss_step": loss_val,
                    f"{stage_name}/acc_step": acc_val,
                    "aggregate/loss_step": loss_val,
                })

            # --- optimizer/scheduler step only every grad_accum_steps ---
            if accum % self.grad_accum_steps == 0:
                if self.scaler:
                    # IMPORTANT: unscale before clipping when using GradScaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if scheduler:
                    scheduler.step()

        # If last partial accumulation didn't trigger a step, you can either:
        # (A) ignore it (common), or
        # (B) step once more to not waste gradients.
        # Minimal behavior: DO NOT step here (keeps semantics stable).

        return (total_loss / max(1, total_n)), (total_correct / max(1, total_n))

    @torch.no_grad()
    def run_validation(self):
        self.model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in self.val_loader:
            if batch_is_empty(batch):
                continue

            loss, correct, n = self._forward_loss_and_metrics(batch)
            total_loss += float(loss.detach().cpu().item()) * n
            total_correct += correct
            total_n += n

        val_loss = total_loss / max(1, total_n)
        val_acc = total_correct / max(1, total_n)
        return val_loss, val_acc

    def run_stage(self, stage_dict):
        stage_name = stage_dict["name"]
        print(f"\nRunning stage: {stage_name}")
        print(">>> Optim cfg:", self.cfg.optim)

        # Freeze/unfreeze as before
        self.model.set_trainable_groups(list(stage_dict["groups"].keys()))

        self.optimizer = hydra.utils.instantiate(
            self.opt_cfg,
            params=param_groups_for(self.model, self.cfg, stage_dict),
            _convert_="all",
        )

        scheduler = None
        if self.cfg.use_warmup:
            num_training_steps = stage_dict["epochs"] * len(self.train_loader)
            scheduler = build_scheduler_for_stage(
                self.optimizer,
                stage_dict=stage_dict,
                cfg=self.cfg,
                num_training_steps=num_training_steps,
            )

        best_val_loss, best_val_acc, epochs_no_improve = float("inf"), 0.0, 0

        for epoch in range(stage_dict["epochs"]):
            train_loss, train_acc = self.run_epoch(stage_name, scheduler)
            val_loss, val_acc = self.run_validation()

            print(
                f"[{stage_name}][Epoch {epoch}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if self.logger:
                wandb.log({
                    "epoch": epoch,
                    f"{stage_name}/loss_epoch": train_loss,
                    f"{stage_name}/acc_epoch": train_acc,
                    f"{stage_name}/val_loss_epoch": val_loss,
                    f"{stage_name}/val_acc_epoch": val_acc,
                    "aggregate/val_loss_epoch": val_loss,
                    "aggregate/val_acc_epoch": val_acc,
                })

            improved = (val_acc > best_val_acc) or (
                (best_val_acc - val_acc < self.acc_epsilon) and (val_loss < best_val_loss)
            )

            if improved:
                best_val_loss, best_val_acc, epochs_no_improve = val_loss, val_acc, 0
                torch.save(self.model.state_dict(), self.checkpoint_save_path)
                print(f"  -> saved best: val_acc={best_val_acc:.4f} val_loss={best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience and epoch >= self.min_epochs:
                    print(f"Early stopping {stage_name}.")
                    break

    def run_train(self):
        for i, stage_dict in enumerate(self.cfg.train_stages):
            if i > 0:
                print("Reloading best model from previous stage.")
                state = torch.load(self.checkpoint_save_path, map_location=self.device)
                self.model.load_state_dict(state)
            self.run_stage(stage_dict)


@hydra.main(config_path="../configs", config_name="chess_encoder_player", version_base="1.1")
def main(cfg):
    set_seed(int(cfg.seed))

    logger = wandb.init(project=cfg.wandb_project, name=cfg.experiment_name) if cfg.log else None

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device={device} | PID={os.getpid()}  (kill -SIGUSR1 {os.getpid()} to checkpoint+exit)")

    trainer = Trainer(cfg=cfg, logger=logger, device=device)
    trainer.init_data()
    trainer.init_model()

    def save_and_exit(*_):
        torch.save(trainer.model.state_dict(), cfg.checkpoint_save_path)
        print(f"Saved checkpoint → {cfg.checkpoint_save_path}")
        sys.exit(0)

    signal.signal(signal.SIGUSR1, save_and_exit)

    trainer.run_train()

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()