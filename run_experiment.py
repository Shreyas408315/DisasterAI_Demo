import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src_disasterai import read_raster, synthetic_pair, discover_pairs
from models import UNet, ProposedAttentionUNet

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class FloodDataset(Dataset):
    def __init__(self, records, size=128, synthetic=False):
        self.records, self.size, self.synthetic = records, size, synthetic
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        if self.synthetic:
            x, y = synthetic_pair(idx, self.size, seed=11)
        else:
            _, ip, mp = self.records[idx]
            if Path(ip).suffix.lower() == ".npy":
                x = np.load(ip).astype(np.float32)
                if x.ndim == 2: x = np.stack([x, x])
                elif x.ndim == 3 and x.shape[-1] in (1,2,3,4): x = np.transpose(x, (2,0,1))
                x = x[:2]
                if x.shape[0] == 1: x = np.repeat(x, 2, axis=0)
                x = np.nan_to_num(x)
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                # Resize through torch to preserve channels.
                xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                x = torch.nn.functional.interpolate(xt, (self.size,self.size), mode="bilinear", align_corners=False)[0].numpy()
            else:
                x = read_raster(ip, self.size, is_mask=False)
            if Path(mp).suffix.lower() == ".npy":
                y = np.load(mp)
                if y.ndim == 3: y = y[0]
                yt = torch.tensor(y.astype(np.float32))[None,None]
                y = torch.nn.functional.interpolate(yt, (self.size,self.size), mode="nearest")[0,0].numpy().astype(np.int64)
            else:
                y = read_raster(mp, self.size, is_mask=True)
            # Normalize mask to binary: positive labels are foreground; negative labels ignored.
            y = np.where(y < 0, -1, (y > 0).astype(np.int64))
        x = np.nan_to_num(x.astype(np.float32))
        y = y.astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

def otsu_threshold(values, bins=256):
    values = values[np.isfinite(values)]
    if values.size == 0: return 0.5
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]
    denom = omega * (1 - omega)
    between = (mu_total * omega - mu) ** 2 / (denom + 1e-12)
    return float(centers[np.argmax(between)])

def threshold_predict(x, threshold, polarity):
    # x has been normalized to [0,1]; water often has lower backscatter,
    # but polarity is selected on validation data rather than assumed.
    ch = x[0]
    return (ch < threshold) if polarity == "dark" else (ch >= threshold)

def valid_pixels(y):
    return y >= 0

def metrics_for(pred, y):
    valid = valid_pixels(y)
    p = pred[valid].astype(bool); t = y[valid].astype(bool)
    if valid.sum() == 0:
        return {k: None for k in ["iou","dice","precision","recall","pixel_accuracy"]}
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum()); tn = int((~p & ~t).sum())
    return {
        "iou": float(tp / (tp+fp+fn+1e-9)),
        "dice": float(2*tp / (2*tp+fp+fn+1e-9)),
        "precision": float(tp / (tp+fp+1e-9)),
        "recall": float(tp / (tp+fn+1e-9)),
        "pixel_accuracy": float((tp+tn)/(tp+tn+fp+fn+1e-9))
    }

def evaluate_model(model, loader, device, criterion=None):
    model.eval(); all_metrics=[]; loss_sum=0; batches=0
    examples=[]
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            logits=model(x)
            safe_y=y.clamp(min=0).float().unsqueeze(1)
            valid=(y>=0).unsqueeze(1)
            loss=nn.functional.binary_cross_entropy_with_logits(logits, safe_y, reduction="none")
            loss=(loss*valid).sum()/(valid.sum()+1e-8)
            loss_sum += float(loss.item()); batches += 1
            probs = torch.sigmoid(logits)
            print(
                f"DEBUG probabilities: min={probs.min().item():.4f}, "
                f"mean={probs.mean().item():.4f}, "
                f"max={probs.max().item():.4f}"
            )
            pred = (probs[:, 0] >= 0.5).cpu().numpy()
            valid_count = int((y >= 0).sum().item())
            flood_count = int(((y == 1) & (y >= 0)).sum().item())
            pred_count = int((pred & (y.cpu().numpy() >= 0)).sum())
            print(
                f"DEBUG evaluation batch: valid={valid_count}, "
                f"actual_flood={flood_count}, predicted_flood={pred_count}"
            )
            ys=y.cpu().numpy()
            for i in range(len(ys)):
                all_metrics.append(metrics_for(pred[i], ys[i]))
                if len(examples)<4: examples.append((x[i].detach().cpu().numpy(), ys[i], pred[i]))
    keys=["iou","dice","precision","recall","pixel_accuracy"]
    avg={k:float(np.nanmean([m[k] for m in all_metrics if m[k] is not None])) if any(m[k] is not None for m in all_metrics) else None for k in keys}
    avg["loss"]=loss_sum/max(batches,1)
    return avg, examples

def train_model(model, train_loader, val_loader, device, epochs, lr, name, out_dir):
    optimizer=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best=float("inf"); history={"train_loss":[],"val_loss":[]}
    model.to(device)
    for epoch in range(1, epochs+1):
        model.train(); total=0; count=0
        for x,y in tqdm(train_loader, desc=f"{name} epoch {epoch}/{epochs}", leave=False):
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            safe_y = y.clamp(min=0).float().unsqueeze(1)
            valid = (y >= 0).unsqueeze(1)
            positive = ((y == 1) & (y >= 0)).sum().float()
            negative = ((y == 0) & (y >= 0)).sum().float()
            pos_weight = (negative / positive.clamp(min=1)).clamp(min=1.0, max=20.0)
            loss_map = nn.functional.binary_cross_entropy_with_logits(
                logits,
                safe_y,
                reduction="none",
                pos_weight=pos_weight
            )
            loss = (loss_map * valid).sum() / valid.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if epoch == 1 and count == 0:
                with torch.no_grad():
                    print(
                        f"DEBUG after first update: "
                        f"parameter_mean={next(model.parameters()).mean().item():.6f}, "
                        f"parameter_std={next(model.parameters()).std().item():.6f}"
                    )
            total += float(loss.item()); count += 1
        train_loss=total/max(count,1)
        val_metrics,_=evaluate_model(model,val_loader,device)
        val_loss=val_metrics["loss"]
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        print(f"{name} epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f}")
        if val_loss < best:
            best=val_loss
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"val_loss":val_loss},out_dir/"checkpoints"/f"{name}_best.pt")
    return history

def plot_examples(examples, out_path, title, model=None, device=None):
    if not examples: return
    n=len(examples)
    fig, axes=plt.subplots(n,4,figsize=(12,3*n))
    if n==1: axes=np.expand_dims(axes,0)
    for i,(x,y,pred) in enumerate(examples):
        axes[i,0].imshow(x[0],cmap="gray"); axes[i,0].set_title("SAR band 1")
        axes[i,1].imshow(y,cmap="gray",vmin=0,vmax=1); axes[i,1].set_title("Ground truth")
        axes[i,2].imshow(pred,cmap="gray",vmin=0,vmax=1); axes[i,2].set_title("Prediction")
        if model is not None:
            model.eval()
            xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            xt.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            score = model(xt).mean()
            grad = torch.autograd.grad(score, xt, retain_graph=False)[0]
            sal = (grad * xt).abs().sum(dim=1)[0].detach().cpu().numpy()
            sal = sal / (sal.max() + 1e-8)
            axes[i,3].imshow(x[0],cmap="gray")
            axes[i,3].imshow(sal,alpha=.55,cmap="hot",vmin=0,vmax=1)
            axes[i,3].set_title("Input-gradient saliency")
        else:
            axes[i,3].imshow(x[0],cmap="gray")
            axes[i,3].set_title("Input")
        for ax in axes[i]: ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_path,dpi=140); plt.close(fig)

def make_synthetic_records(n):
    return [(f"toy_{i}",None,None) for i in range(n)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",default="data")
    ap.add_argument("--demo-synthetic",action="store_true")
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--image-size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--max-train",type=int,default=0)
    ap.add_argument("--max-val",type=int,default=0)
    ap.add_argument("--max-test",type=int,default=0)
    args=ap.parse_args()
    set_seed(args.seed)
    out=Path("outputs"); (out/"checkpoints").mkdir(parents=True,exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"Device: {device} | GPU: {torch.xpu.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: cpu (no supported CUDA or Intel XPU device detected)")
    if args.demo_synthetic:
        train=make_synthetic_records(max(args.max_train,12))
        val=make_synthetic_records(max(args.max_val,4))
        test=make_synthetic_records(max(args.max_test,4))
        synthetic=True
    else:
        splits=discover_pairs(args.data_root)
        train,val,test=splits["train"],splits["val"],splits["test"]; synthetic=False
        for name,rec,mx in [("train",train,args.max_train),("val",val,args.max_val),("test",test,args.max_test)]:
            if mx>0: 
                if name=="train": train=rec[:mx]
                elif name=="val": val=rec[:mx]
                else: test=rec[:mx]
        if not train or not val or not test:
            raise RuntimeError(f"Need non-empty train/val/test splits; counts: train={len(train)}, val={len(val)}, test={len(test)}")
    # Keep batch size small for local GPU memory.
    train_ds=FloodDataset(train,args.image_size,synthetic)
    val_ds=FloodDataset(val,args.image_size,synthetic)
    test_ds=FloodDataset(test,args.image_size,synthetic)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)

    config=vars(args); config["device"]=str(device); config["synthetic_demo"]=synthetic
    (out/"run_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    results={}
    # Method 1: thresholding baseline. Pick threshold and polarity on validation only.
    print("\n[1/3] Traditional threshold baseline")
    candidates=[]
    val_arrays=[val_ds[i] for i in range(len(val_ds))]
    for polarity in ("dark","bright"):
        for t in np.linspace(.05,.95,37):
            ms=[]
            for x,y in val_arrays:
                pred=threshold_predict(x.numpy(),float(t),polarity)
                ms.append(metrics_for(pred,y.numpy())["dice"])
            candidates.append((float(np.nanmean(ms)),float(t),polarity))
    best_dice,best_t,best_pol=max(candidates,key=lambda q:q[0])
    test_ms=[]
    threshold_examples=[]
    for i in range(len(test_ds)):
        x,y=test_ds[i]; p=threshold_predict(x.numpy(),best_t,best_pol)
        test_ms.append(metrics_for(p,y.numpy()))
        if len(threshold_examples)<4: threshold_examples.append((x.numpy(),y.numpy(),p))
    results["threshold_baseline"]={k:float(np.nanmean([m[k] for m in test_ms])) for k in ["iou","dice","precision","recall","pixel_accuracy"]}
    results["threshold_baseline"].update({"validation_selected_threshold":best_t,"validation_selected_polarity":best_pol})
    plot_examples(threshold_examples,out/"threshold_predictions.png","Threshold baseline (test examples)")

    # Method 2: vanilla U-Net from scratch
    print("\n[2/3] Vanilla U-Net baseline (random initialization)")
    unet=UNet(base=16)
    h1=train_model(unet,train_loader,val_loader,device,args.epochs,args.lr,"unet",out)
    ck=torch.load(out/"checkpoints"/"unet_best.pt",map_location=device,weights_only=False)
    unet.load_state_dict(ck["model_state_dict"])
    m1,ex1=evaluate_model(unet,test_loader,device); results["vanilla_unet"]=m1
    plot_examples(ex1,out/"unet_predictions.png","Vanilla U-Net (test examples)",model=unet,device=device)

    # Method 3: proposed attention U-Net from scratch
    print("\n[3/3] Proposed attention U-Net (random initialization)")
    proposed=ProposedAttentionUNet(base=16)
    h2=train_model(proposed,train_loader,val_loader,device,args.epochs,args.lr,"proposed_attention_unet",out)
    ck=torch.load(out/"checkpoints"/"proposed_attention_unet_best.pt",map_location=device,weights_only=False)
    proposed.load_state_dict(ck["model_state_dict"])
    m2,ex2=evaluate_model(proposed,test_loader,device); results["proposed_attention_unet"]=m2
    plot_examples(ex2,out/"proposed_predictions.png","Proposed attention U-Net (test examples)",model=proposed,device=device)
    # Curves
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(h1["train_loss"],label="U-Net train"); ax.plot(h1["val_loss"],label="U-Net validation")
    ax.plot(h2["train_loss"],label="Proposed train"); ax.plot(h2["val_loss"],label="Proposed validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Masked BCE loss"); ax.legend(); ax.grid(True,alpha=.3)
    fig.tight_layout(); fig.savefig(out/"training_curves.png",dpi=140); plt.close(fig)
    (out/"metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    rows=[]
    for method,met in results.items():
        row={"method":method}; row.update(met); rows.append(row)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    print("\n=== TEST RESULTS FROM THIS RUN ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved outputs to: {out.resolve()}")
    if synthetic:
        print("WARNING: Synthetic smoke-test data used. These metrics are NOT real flood-detection results.")

if __name__=="__main__":
    main()
