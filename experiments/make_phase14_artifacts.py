from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def latest_run_all_dir(root: Path) -> Path:
    d = root / "results" / "run_all"
    runs = sorted([p for p in d.glob("*") if p.is_dir()])
    if not runs:
        raise RuntimeError("Không tìm thấy results/run_all/<run_id>. Hãy chạy experiments/run_all.py trước.")
    return runs[-1]

def main():
    root = Path(__file__).resolve().parents[1]
    run_dir = latest_run_all_dir(root)

    ab_path = run_dir / "ablation_summary.csv"
    ws_path = run_dir / "ws_sweep.csv"

    if not ab_path.exists():
        raise RuntimeError(f"Thiếu file: {ab_path}")

    ab = pd.read_csv(ab_path)
    ws = pd.read_csv(ws_path) if ws_path.exists() else None

    out_csv = root / "results" / "ablation_summary.csv"
    out_png = root / "results" / "ablation_plots.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # copy CSV ra root/results
    ab.to_csv(out_csv, index=False)

    # Vẽ 1 ảnh gồm 2 plot (b-sweep + heatmap WS)
    plt.figure(figsize=(10, 4))

    # Plot 1: b sweep
    plt.subplot(1, 2, 1)
    if "b" in ab.columns:
        ab2 = ab.sort_values("b")
        ycol = "acc_val" if "acc_val" in ab2.columns else ab2.columns[-1]
        plt.plot(ab2["b"].values, ab2[ycol].values, marker="o", label=ycol)
        if "acc_test" in ab2.columns:
            plt.plot(ab2["b"].values, ab2["acc_test"].values, marker="o", label="acc_test")
        plt.title("MBH-LPQ: b sweep")
        plt.xlabel("b (sub-block)")
        plt.ylabel("accuracy")
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "ablation_summary.csv thiếu cột b", ha="center", va="center")
        plt.axis("off")

    # Plot 2: WS heatmap
    plt.subplot(1, 2, 2)
    if ws is not None and len(ws) > 0 and ("b" in ws.columns) and ("alpha" in ws.columns):
        val_col = "acc_val" if "acc_val" in ws.columns else ws.columns[-1]
        piv = ws.pivot_table(index="b", columns="alpha", values=val_col, aggfunc="mean").sort_index()
        plt.imshow(piv.values, aspect="auto")
        plt.title(f"WS heatmap ({val_col})")
        plt.yticks(range(len(piv.index)), piv.index.astype(int))
        plt.xticks(range(len(piv.columns)), [f"{a:.1f}" for a in piv.columns], rotation=45, ha="right")
        plt.xlabel("alpha")
        plt.ylabel("b")
        plt.colorbar()
    else:
        plt.text(0.5, 0.5, "ws_sweep.csv not found / thiếu cột", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Source run_dir:", run_dir)
    print("Wrote:", out_csv)
    print("Wrote:", out_png)

if __name__ == "__main__":
    main()
