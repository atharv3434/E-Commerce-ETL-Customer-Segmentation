"""Read the loaded warehouse back and draw charts (proves the LOAD stage is queryable)."""
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SQLITE_DB, ROOT  # noqa: E402

out = ROOT / "reports"
out.mkdir(exist_ok=True)
con = sqlite3.connect(SQLITE_DB)

# 1. Monthly revenue by category (SQL on the warehouse)
monthly = pd.read_sql("""
    SELECT substr(order_date, 1, 7) AS month, category, SUM(revenue) AS revenue
    FROM agg_daily_sales GROUP BY 1, 2 ORDER BY 1""", con)
pivot = monthly.pivot(index="month", columns="category", values="revenue").fillna(0)
ax = pivot.plot(kind="area", stacked=True, figsize=(11, 5), colormap="viridis", alpha=0.9)
ax.set(title="Monthly net revenue by category", ylabel="Revenue", xlabel="")
plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(out / "monthly_revenue.png", dpi=130); plt.close()

# 2. Customer segments: recency vs monetary
seg = pd.read_sql("SELECT * FROM customer_segments", con)
fig, ax = plt.subplots(figsize=(8, 5.5))
for name, g in seg.groupby("segment"):
    ax.scatter(g["recency_days"], g["monetary"], s=14, alpha=0.6, label=f"{name} ({len(g)})")
ax.set(yscale="log", xlabel="Recency (days since last order)", ylabel="Total revenue (log)",
       title="Customer segments (KMeans, k=4)")
ax.legend(); plt.tight_layout(); plt.savefig(out / "customer_segments.png", dpi=130); plt.close()

# 3. Revenue concentration
share = seg.groupby("segment")["monetary"].sum()
print((share / share.sum() * 100).round(1).sort_values(ascending=False).to_string())
print(f"charts saved to {out}")
