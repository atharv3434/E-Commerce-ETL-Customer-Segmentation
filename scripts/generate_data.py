"""Generate a synthetic, deliberately *messy* e-commerce dataset.

Injected data-quality problems (to give the ETL something real to fix):
  - duplicate order rows
  - null / negative / non-numeric quantities and null prices
  - inconsistent country spellings and casing, stray whitespace
  - mixed timestamp formats
  - orders referencing unknown customers/products (orphans)
  - invalid email addresses
"""
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DIR, RANDOM_SEED  # noqa: E402

N_CUSTOMERS, N_PRODUCTS, N_ORDERS = 2_000, 150, 40_000

random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- customers ----------------
countries = ["India", "USA", "UK", "Germany", "France", "Canada"]
country_noise = {
    "India": ["India", "india", "IN", " India "],
    "USA": ["USA", "usa", "United States", "US"],
    "UK": ["UK", "uk", "United Kingdom"],
    "Germany": ["Germany", "germany", "DE"],
    "France": ["France", "FRANCE"],
    "Canada": ["Canada", "canada", "CA"],
}
cust_rows = []
for i in range(1, N_CUSTOMERS + 1):
    c = random.choice(countries)
    signup = datetime(2022, 1, 1) + timedelta(days=int(rng.integers(0, 900)))
    email = f"user{i}@example.com"
    if random.random() < 0.04:
        email = f"user{i}_at_example"  # invalid
    cust_rows.append({
        "customer_id": f"C{i:05d}",
        "name": f"  Customer {i} " if random.random() < 0.1 else f"Customer {i}",
        "email": email,
        "country": random.choice(country_noise[c]),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })
pd.DataFrame(cust_rows).to_csv(RAW_DIR / "customers.csv", index=False)

# ---------------- products (JSON lines) ----------------
categories = {"Electronics": (40, 900), "Clothing": (10, 120), "Home": (15, 300),
              "Books": (5, 50), "Sports": (20, 400), "Beauty": (8, 150)}
products = []
for i in range(1, N_PRODUCTS + 1):
    cat = random.choice(list(categories))
    lo, hi = categories[cat]
    list_price = round(float(rng.uniform(lo, hi)), 2)
    products.append({
        "product_id": f"P{i:04d}",
        "product_name": f"{cat} Item {i}",
        "category": cat.lower() if random.random() < 0.1 else cat,
        "unit_cost": round(list_price * 0.6, 2),
        "list_price": list_price,
    })
with open(RAW_DIR / "products.json", "w") as f:
    for p in products:
        f.write(json.dumps(p) + "\n")

# ---------------- orders ----------------
# Heavy-tailed customer activity so segmentation finds real structure.
weights = rng.pareto(1.15, N_CUSTOMERS) + 0.02
weights /= weights.sum()
cust_idx = rng.choice(N_CUSTOMERS, size=N_ORDERS, p=weights)
price_map = {p["product_id"]: p["list_price"] for p in products}
pids = list(price_map)

rows = []
start = datetime(2024, 1, 1)
for k in range(N_ORDERS):
    d = start + timedelta(days=int(rng.integers(0, 640)), minutes=int(rng.integers(0, 1440)))
    pid = random.choice(pids)
    qty = int(rng.integers(1, 6))
    price = price_map[pid]
    cid = f"C{cust_idx[k] + 1:05d}"

    r = random.random()
    if r < 0.01: qty = None
    elif r < 0.02: qty = -qty
    elif r < 0.025: qty = "two"
    if random.random() < 0.01: price = None
    if random.random() < 0.01: cid = "C99999"   # orphan customer
    if random.random() < 0.005: pid = "P9999"   # orphan product
    fmt = random.choice(["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"])
    rows.append({
        "order_id": f"O{k + 1:07d}",
        "customer_id": cid,
        "product_id": pid,
        "quantity": qty,
        "unit_price": price,
        "discount_pct": random.choice([0, 0, 0, 5, 10, 15, 20]),
        "order_ts": d.strftime(fmt),
        "status": random.choice(["completed"] * 8 + ["returned", "cancelled", "COMPLETED"]),
    })

df = pd.DataFrame(rows)
dupes = df.sample(frac=0.02, random_state=RANDOM_SEED)
df = pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
df.to_csv(RAW_DIR / "orders.csv", index=False)

print(f"customers={N_CUSTOMERS:,}  products={N_PRODUCTS}  orders={len(df):,} (incl. {len(dupes)} duplicates)")
print(f"written to {RAW_DIR}")
