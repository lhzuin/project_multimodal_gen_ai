
# train/distillation_encoder.py
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

# NEW: distillation DB dataset + collate
from dataset.distillation_db import DistillationEncoderDataset, collate_distill_encoder

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
    # Distillation collate returns {"valid": False} when batch had no valid samples.
    if not isinstance(batch, dict):
        print("Warning: expected batch to be a dict, got", type(batch))
        return True
    if batch.get("valid", True) is False:
        print("Warning: batch marked as invalid (empty after filtering)")
        return True
    return False


# def teacher_sparse_ce(policy_logits_flat: torch.Tensor,
#                       teacher_idx: torch.Tensor,
#                       teacher_probs: torch.Tensor,
#                       teacher_mask: torch.Tensor) -> torch.Tensor:
#     """
#     policy_logits_flat: [B, 4096]
#     teacher_idx:        [B, K] with -1 for padding
#     teacher_probs:      [B, K] (already sums to 1 per row over valid entries)
#     teacher_mask:       [B, K] bool
#     Returns: mean loss over batch (float tensor)
#     """
#     # gather logits on teacher support
#     safe_idx = teacher_idx.clamp_min(0)  # avoid negative indices in gather
#     gathered = policy_logits_flat.gather(1, safe_idx)  # [B,K]
#     # mask out padded entries
#     gathered = gathered.masked_fill(~teacher_mask, float("-inf"))
#     logp = torch.log_softmax(gathered, dim=-1)  # normalized on teacher support
#     loss = -(teacher_probs * logp).sum(dim=-1)  # [B]
#     return loss.mean()

def teacher_sparse_ce_support(policy_logits_flat, teacher_idx, teacher_probs, teacher_mask):
    safe_idx = teacher_idx.clamp_min(0)
    gathered = policy_logits_flat.gather(1, safe_idx)          # [B,K]
    gathered = gathered.masked_fill(~teacher_mask, float("-inf"))

    logp = torch.log_softmax(gathered, dim=-1)                 # [B,K]
    logp = logp.masked_fill(~teacher_mask, 0.0)                # IMPORTANT: avoid 0 * -inf

    # (optional but nice)
    teacher_probs = teacher_probs.masked_fill(~teacher_mask, 0.0)

    loss = -(teacher_probs * logp).sum(dim=-1)                 # [B]
    return loss.mean()


def teacher_sparse_ce_full(policy_logits_flat, teacher_idx, teacher_probs, teacher_mask):
    # full distribution over 4096
    logp_full = torch.log_softmax(policy_logits_flat, dim=-1)   # [B,4096]

    safe_idx = teacher_idx.clamp_min(0)
    logp_sup = logp_full.gather(1, safe_idx)                   # [B,K]
    logp_sup = logp_sup.masked_fill(~teacher_mask, 0.0)

    teacher_probs = teacher_probs.masked_fill(~teacher_mask, 0.0)
    return -(teacher_probs * logp_sup).sum(dim=-1).mean()


def promo_distill_ce(promo_logits: torch.Tensor,
                     teacher_promo_probs_5: torch.Tensor) -> torch.Tensor:
    """
    promo_logits: [B, 5]
    teacher_promo_probs_5: [B, 5] distribution over (none, n, b, r, q)
    """
    logp = torch.log_softmax(promo_logits, dim=-1)
    return -(teacher_promo_probs_5 * logp).sum(dim=-1).mean()


class Trainer:
    def __init__(self, cfg, logger, device):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.model = None
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

        # Build opt cfg like train_chessformer.py (keep only AdamW-safe args)
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
        # NEW: build dataset from distill DB meta.json
        ds = DistillationEncoderDataset(
            meta_path=self.cfg.distill.meta_path,
            tau=float(self.cfg.distill.tau),
            mate_cp=int(self.cfg.distill.mate_cp),
            return_text=bool(getattr(self.cfg.distill, "return_text", False)),
            return_state_tensors=True,
        )

        val_fraction = float(getattr(self.cfg.data, "val_fraction", 0.05))
        train_idx, val_idx = split_indices(len(ds), val_fraction, seed=int(self.cfg.seed))

        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

        dl_cfg = self.cfg.dataloader

        self.train_loader = DataLoader(
            train_ds,
            batch_size=dl_cfg.batch_size,
            shuffle=dl_cfg.shuffle,
            num_workers=dl_cfg.num_workers,
            collate_fn=collate_distill_encoder,
            pin_memory=dl_cfg.pin_memory,
            persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
            drop_last=getattr(dl_cfg, "drop_last", False),
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=dl_cfg.batch_size,
            shuffle=False,
            num_workers=dl_cfg.num_workers,
            collate_fn=collate_distill_encoder,
            pin_memory=dl_cfg.pin_memory,
            persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
            drop_last=False,
        )

        print(f"Distill DB size={len(ds)} | train={len(train_ds)} | val={len(val_ds)}")

        print("len(ds)=", len(ds))
        print("len(train_ds)=", len(train_ds), "len(val_ds)=", len(val_ds))
        print("batch_size=", dl_cfg.batch_size)
        print("len(train_loader)=", len(self.train_loader), "len(val_loader)=", len(self.val_loader))
        print("num_workers=", dl_cfg.num_workers, "persistent=", dl_cfg.persistent_workers)
    
    def init_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model.instance).to(self.device)

        # Load pretrained full-model checkpoint if provided
        pretrained = getattr(self.cfg.model, "pretrained_path", None)
        if pretrained:
            state = torch.load(pretrained, map_location=self.device)

            # handle either raw state_dict or {"state_dict": ...}
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            strict = bool(getattr(self.cfg.model, "strict_pretrained", True))
            missing, unexpected = self.model.load_state_dict(state, strict=strict)
            print(f"[pretrained] loaded from {pretrained} | missing={len(missing)} unexpected={len(unexpected)}")


    def _forward_loss_and_metrics(self, batch):
        # Teacher (sparse)
        teacher_from = batch["teacher_from"].to(self.device, non_blocking=True)   # [B,K]
        teacher_to = batch["teacher_to"].to(self.device, non_blocking=True)       # [B,K]
        teacher_probs = batch["teacher_probs"].to(self.device, non_blocking=True) # [B,K]
        teacher_mask = batch["teacher_mask"].to(self.device, non_blocking=True)   # [B,K] bool
        fen_list = batch["fen"]  # list[str]

        # Here, we reconstruct these from FEN on the fly (cheap, avoids storing extra fields in DB).

        # Model expects turn/castling/ep_square; derive from FEN.
        piece_probs = batch["piece_probs"].to(self.device, non_blocking=True)  # [B,64,13]
        turn       = batch["turn"].to(self.device, non_blocking=True)          # [B]
        castling   = batch["castling"].to(self.device, non_blocking=True)      # [B,4]
        ep_square  = batch["ep_square"].to(self.device, non_blocking=True)     # [B]

        # Forward (use pieces pathway by default for distillation)
        out = self.model(
            piece_probs=piece_probs,
            turn=turn,
            castling=castling,
            ep_square=ep_square,
        )

        policy_logits = out["policy_logits"]  # [B,64,64]
        promo_logits = out["promo_logits"]    # [B,5]
        B2 = policy_logits.size(0)

        # --- legal mask ---
        if getattr(self.cfg, "use_legal_mask", True):
            legal_mask = self.model.legal_mask_from_fen(fen_list, device=policy_logits.device)
            policy_logits = self.model.apply_legal_mask(policy_logits, legal_mask)

        flat_logits = policy_logits.view(B2, 4096)

        # Convert teacher (from,to) -> flat index
        teacher_idx = teacher_from * 64 + teacher_to  # [B,K], -64.. etc if padded; mask will remove
        teacher_idx = teacher_idx.to(dtype=torch.long)

        loss_support = teacher_sparse_ce_support(flat_logits, teacher_idx, teacher_probs, teacher_mask)
        loss_full = teacher_sparse_ce_full(flat_logits, teacher_idx, teacher_probs, teacher_mask)
        full_coeff = getattr(self.cfg.distill, "full_loss_coeff", 1.0)
        loss_policy = full_coeff * loss_full + (1.0 - full_coeff) * loss_support

        # Promo distillation (optional)
        loss_promo = torch.tensor(0.0, device=self.device)
        if "teacher_promo_probs" in batch and getattr(self.cfg.distill, "use_promo_loss", True):
            tp = batch["teacher_promo_probs"].to(self.device, non_blocking=True)  # [B,4] over n,b,r,q
            # Build 5-class dist for (none,n,b,r,q). Assumes your model uses 0=none,1=n,2=b,3=r,4=q.
            teacher_promo_5 = torch.zeros((B2, 5), dtype=torch.float32, device=self.device)
            teacher_promo_5[:, 1:] = tp
            teacher_promo_5[:, 0] = (1.0 - tp.sum(dim=-1)).clamp(min=0.0)
            loss_promo = promo_distill_ce(promo_logits, teacher_promo_5)

        loss = loss_policy + float(getattr(self.cfg.distill, "promo_loss_weight", 1.0)) * loss_promo
        if not torch.isfinite(loss):
            print(f"Non-finite loss encountered: {loss.item()} | skipping batch")
            return None

        # Metrics: match teacher top-1 (argmax over teacher support)
        # teacher best index per sample:
        best_k = teacher_probs.argmax(dim=-1)  # [B]
        teacher_best_idx = teacher_idx.gather(1, best_k.unsqueeze(1)).squeeze(1)  # [B]
        pred = flat_logits.argmax(dim=-1)  # [B]
        correct1 = (pred == teacher_best_idx).sum().item()

        # top-5 vs teacher best
        topk_idx = torch.topk(flat_logits, k=5, dim=-1).indices
        correct5 = (topk_idx == teacher_best_idx.unsqueeze(1)).any(dim=1).sum().item()


        tp = teacher_probs  # [B,K]
        tm = teacher_mask   # [B,K]
        eps = 1e-8
        tp2 = tp.masked_fill(~tm, 0.0)
        H = -(tp2 * (tp2 + eps).log()).sum(dim=-1).mean()

        K = teacher_mask.sum(dim=-1).float().clamp_min(1.0)
        logK = K.log().mean()

        # after flat_logits exists and teacher_idx/teacher_mask exist
        with torch.no_grad():
            logp_full = torch.log_softmax(flat_logits, dim=-1)  # [B,4096]
            sup = logp_full.gather(1, teacher_idx.clamp_min(0)) # [B,K]
            sup = sup.masked_fill(~teacher_mask, float("-inf"))
            support_mass = torch.exp(torch.logsumexp(sup, dim=-1)).mean()

        return loss, correct1, correct5, B2, float(loss_policy.detach().cpu().item()), float(loss_promo.detach().cpu().item()), float(H.detach().cpu().item()), float(logK.detach().cpu().item()), float(loss_support.detach().cpu().item()), float(loss_full.detach().cpu().item()), float(support_mass.detach().cpu().item())

    def run_epoch(self, stage_name, scheduler):
        self.model.train()
        total_loss, total_correct, total_correct5, total_n = 0.0, 0, 0, 0
        total_policy_loss, total_promo_loss = 0.0, 0.0
        total_H, total_logK = 0.0, 0.0
        total_loss_support = 0.0
        total_loss_full = 0.0

        accum = 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(self.train_loader, desc=f"{stage_name} | distill-train", leave=False)
        for batch in progress:
            if batch_is_empty(batch):
                print("Skipping empty batch")
                continue

            if self.scaler:
                with autocast(device_type=self.device.type):
                    res = self._forward_loss_and_metrics(batch)
                    if res is None:
                        continue
                    loss, correct, correct5, n, lp, lpr, H, logK, loss_support, loss_full, support_mass = res
                    loss_scaled = loss / self.grad_accum_steps
                self.scaler.scale(loss_scaled).backward()
            else:
                res = self._forward_loss_and_metrics(batch)
                if res is None:
                    continue
                loss, correct, correct5, n, lp, lpr, H, logK, loss_support, loss_full, support_mass = res
                (loss / self.grad_accum_steps).backward()

            accum += 1

            loss_val = float(loss.detach().cpu().item())
            total_loss += loss_val * n
            total_loss_support += loss_support*n
            total_loss_full += loss_full*n
            total_policy_loss += float(lp) * n
            total_promo_loss += float(lpr) * n
            total_H += float(H) * n
            total_logK += float(logK) * n
            total_correct += correct
            total_correct5 += correct5
            total_n += n

            acc_val = correct / max(1, n)
            acc5_val = correct5 / max(1, n)
            progress.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.3f}", acc5=f"{acc5_val:.3f}", loss_support=f"{loss_support:.2f}", loss_full=f"{loss_full:.2f}")

            if self.logger:
                wandb.log({
                    f"{stage_name}/loss_step": loss_val,
                    f"{stage_name}/policy_loss_step": float(lp),
                    f"{stage_name}/promo_loss_step": float(lpr),
                    f"{stage_name}/acc_step": acc_val,
                    f"{stage_name}/acc5_step": acc5_val,
                    f"{stage_name}/teacher_entropy_step": float(H),
                    f"{stage_name}/logK_step": float(logK),
                    f"{stage_name}/support_mass_step": float(support_mass),
                    f"{stage_name}/loss_support_step": float(loss_support),
                    f"{stage_name}/loss_full_step": float(loss_full),
                    "aggregate/loss_step": loss_val,
                })

            if accum % self.grad_accum_steps == 0:
                if self.scaler:
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

        return (
            total_loss / max(1, total_n),
            total_correct / max(1, total_n),
            total_correct5 / max(1, total_n),
            total_policy_loss / max(1, total_n),
            total_promo_loss / max(1, total_n),
            total_H / max(1, total_n),
            total_logK / max(1, total_n),
            total_loss_support / max(1, total_n),
            total_loss_full / max(1, total_n),
        )

    @torch.no_grad()
    def run_validation(self):
        self.model.eval()
        total_loss, total_correct, total_correct5, total_n = 0.0, 0, 0, 0
        total_policy_loss, total_promo_loss = 0.0, 0.0
        total_H, total_logK = 0.0, 0.0
        total_loss_support, total_loss_full = 0.0, 0.0

        for batch in self.val_loader:
            if batch_is_empty(batch):
                continue
            res = self._forward_loss_and_metrics(batch)
            if res is None:
                continue
            loss, correct, correct5, n, lp, lpr, H, logK, loss_support, loss_full, support_mass = res
            total_loss += float(loss.detach().cpu().item()) * n
            total_policy_loss += float(lp) * n
            total_promo_loss += float(lpr) * n
            total_H += float(H) * n
            total_logK += float(logK) * n
            total_loss_support += float(loss_support) * n
            total_loss_full += float(loss_full) * n
            total_correct += correct
            total_correct5 += correct5
            total_n += n

        return (
            total_loss / max(1, total_n),
            total_correct / max(1, total_n),
            total_correct5 / max(1, total_n),
            total_policy_loss / max(1, total_n),
            total_promo_loss / max(1, total_n),
            total_H / max(1, total_n),
            total_logK / max(1, total_n),
            total_loss_support / max(1, total_n),
            total_loss_full / max(1, total_n),
        )

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
            steps_per_epoch = len(self.train_loader) // self.grad_accum_steps
            num_training_steps = stage_dict["epochs"] * steps_per_epoch
            scheduler = build_scheduler_for_stage(
                self.optimizer,
                stage_dict=stage_dict,
                cfg=self.cfg,
                num_training_steps=num_training_steps,
            )

        best_val_loss, best_val_acc, epochs_no_improve = float("inf"), 0.0, 0

        for epoch in range(stage_dict["epochs"]):
            train_loss, train_acc, train_acc5, train_pl, train_pr, train_H, train_logK, train_loss_support, train_loss_full = self.run_epoch(stage_name, scheduler)
            val_loss, val_acc, val_acc5, val_pl, val_pr, val_H, val_logK, val_loss_support, val_loss_full = self.run_validation()

            print(
                f"[{stage_name}][Epoch {epoch}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_acc5={train_acc5:.4f} "
                f"(policy={train_pl:.4f}, promo={train_pr:.4f}) | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_acc5={val_acc5:.4f} "
                f"(policy={val_pl:.4f}, promo={val_pr:.4f}) | "
                f"train_H={train_H:.4f} train_logK={train_logK:.4f} | val_H={val_H:.4f} val_logK={val_logK:.4f} | "
                f"train_loss_support={train_loss_support:.2f} train_loss_full={train_loss_full:.2f} | "
                f"val_loss_support={val_loss_support:.2f} val_loss_full={val_loss_full:.2f}"
            )

            if self.logger:
                wandb.log({
                    "epoch": epoch,
                    f"{stage_name}/loss_epoch": train_loss,
                    f"{stage_name}/acc_epoch": train_acc,
                    f"{stage_name}/acc5_epoch": train_acc5,
                    f"{stage_name}/policy_loss_epoch": train_pl,
                    f"{stage_name}/promo_loss_epoch": train_pr,
                    f"{stage_name}/train_loss_support_epoch": train_loss_support,
                    f"{stage_name}/train_loss_full_epoch": train_loss_full,
                    f"{stage_name}/val_loss_epoch": val_loss,
                    f"{stage_name}/val_acc_epoch": val_acc,
                    f"{stage_name}/val_acc5_epoch": val_acc5,
                    f"{stage_name}/val_policy_loss_epoch": val_pl,
                    f"{stage_name}/val_promo_loss_epoch": val_pr,
                    f"{stage_name}/val_loss_support_epoch": val_loss_support,
                    f"{stage_name}/val_loss_full_epoch": val_loss_full,
                    
                    "aggregate/val_loss_epoch": val_loss,
                    "aggregate/val_acc_epoch": val_acc,
                    "aggregate/val_acc5_epoch": val_acc5,
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


@hydra.main(config_path="../configs", config_name="distillation_encoder_v4", version_base="1.1")
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
