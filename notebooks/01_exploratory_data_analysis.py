"""
notebooks/01_exploratory_data_analysis.py
------------------------------------------
Run as a plain script:
    python notebooks/01_exploratory_data_analysis.py

Or open in Jupyter:
    jupyter notebook

Sections
--------
1.  Load & overview the dataset
2.  Missing value analysis
3.  Top purchased products
4.  Customer purchase frequency distribution
5.  Product popularity distribution
6.  Monthly revenue trend
7.  Basket size analysis
8.  Country-level heatmap
9.  User–item interaction matrix sparsity
10. Top association rule pairs (bar chart)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".."))

# ── Style ──────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.dpi"       : 120,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "sans-serif",
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})
PALETTE = sns.color_palette("Blues_r", 15)
sns.set_style("whitegrid")

OUTPUT_DIR = "notebooks/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ 1. Loading data ═══")
raw_path = "data/raw/ecommerce_data.csv"
if not os.path.exists(raw_path):
    print("Raw data not found – generating …")
    from data.generate_dataset import generate_dataset
    generate_dataset()

df_raw = pd.read_csv(raw_path, parse_dates=["Timestamp"])
print(f"Shape            : {df_raw.shape}")
print(f"Columns          : {df_raw.columns.tolist()}")
print(f"Date range       : {df_raw['Timestamp'].min()} → {df_raw['Timestamp'].max()}")
print(f"Unique customers : {df_raw['CustomerID'].nunique():,}")
print(f"Unique products  : {df_raw['ProductID'].nunique():,}")
print(df_raw.head(3).to_string())

# ── Clean version ──────────────────────────────────────────────────────────────
clean_path = "data/processed/clean_transactions.csv"
if os.path.exists(clean_path):
    df = pd.read_csv(clean_path, parse_dates=["Timestamp"])
    print(f"\nClean shape: {df.shape}")
else:
    print("\nClean data not found – using raw with basic cleaning.")
    df = df_raw[
        (~df_raw["TransactionID"].astype(str).str.startswith("C"))
        & (df_raw["Quantity"] > 0)
        & (df_raw["UnitPrice"] > 0)
    ].dropna(subset=["CustomerID", "ProductID"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ 2. Missing values ═══")
miss = df_raw.isnull().sum()
miss = miss[miss > 0]
if miss.empty:
    print("  No missing values in raw data.")
else:
    print(miss.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TOP 15 PURCHASED PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ 3. Top purchased products ═══")
top_products = (
    df.groupby(["ProductID", "ProductName"])["Quantity"]
    .sum()
    .nlargest(15)
    .reset_index()
)
top_products["Label"] = top_products["ProductName"].str[:30]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_products["Label"][::-1], top_products["Quantity"][::-1], color=PALETTE[:15])
ax.set_xlabel("Total Units Sold")
ax.set_title("Top 15 Best-Selling Products")
ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=9)
fig.tight_layout()
save(fig, "top_products.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CUSTOMER PURCHASE FREQUENCY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ 4. Customer purchase frequency ═══")
order_freq = df.groupby("CustomerID")["TransactionID"].nunique()
print(f"  Median orders per customer : {order_freq.median():.0f}")
print(f"  Mean   orders per customer : {order_freq.mean():.1f}")
print(f"  Max    orders per customer : {order_freq.max():.0f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(order_freq, bins=30, color="#5b8dee", edgecolor="white")
ax1.set_xlabel("Number of Orders")
ax1.set_ylabel("Number of Customers")
ax1.set_title("Customer Order Frequency Distribution")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

ax2.boxplot(order_freq, vert=False, patch_artist=True,
            boxprops=dict(facecolor="#5b8dee", alpha=0.7))
ax2.set_xlabel("Number of Orders")
ax2.set_title("Boxplot: Orders per Customer")
fig.tight_layout()
save(fig, "customer_order_frequency.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PRODUCT POPULARITY (LOG SCALE)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ 5. Product popularity distribution ═══")
prod_pop = df.groupby("ProductID")["Quantity"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(prod_pop)), prod_pop.values, marker="o", ms=5, color="#5b8dee", linewidth=1.5)
ax.set_yscale("log")
ax.set_xlabel("Product Rank")
ax.set_ylabel("Total Units Sold (log scale)")
ax.set_title("Product Popularity Distribution (Long-Tail)")
ax.fill_between(range(len(prod_pop)), prod_pop.values, alpha=0.15, color="#5b8dee")
save(fig, "product_popularity_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MONTHLY REVENUE TREND
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ 6. Monthly revenue trend ═══")
if "Revenue" not in df.columns:
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

monthly = (
    df.groupby(df["Timestamp"].dt.to_period("M"))["Revenue"]
    .sum()
    .reset_index()
)
monthly["Timestamp"] = monthly["Timestamp"].astype(str)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(monthly["Timestamp"], monthly["Revenue"], color="#5b8dee", linewidth=2, marker="s", ms=6)
ax.fill_between(monthly["Timestamp"], monthly["Revenue"], alpha=0.15, color="#5b8dee")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.set_title("Monthly Revenue Trend")
ax.set_xticklabels(monthly["Timestamp"], rotation=45, ha="right")
fig.tight_layout()
save(fig, "monthly_revenue_trend.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BASKET SIZE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ 7. Basket size analysis ═══")
basket_size = df.groupby("TransactionID")["ProductID"].nunique()
print(f"  Average basket size   : {basket_size.mean():.2f} products")
print(f"  Median  basket size   : {basket_size.median():.0f} products")
print(f"  Max     basket size   : {basket_size.max():.0f} products")

fig, ax = plt.subplots(figsize=(9, 4))
basket_size.value_counts().sort_index().plot(kind="bar", ax=ax, color="#5b8dee", edgecolor="white")
ax.set_xlabel("Products per Basket")
ax.set_ylabel("Number of Baskets")
ax.set_title("Basket Size Distribution")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
save(fig, "basket_size_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. COUNTRY REVENUE HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
if "Country" in df.columns:
    print("\n═══ 8. Country revenue heatmap ═══")
    country_rev = (
        df.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=country_rev, y="Country", x="Revenue", palette="Blues_r", ax=ax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
    ax.set_title("Revenue by Country (Top 10)")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("")
    fig.tight_layout()
    save(fig, "country_revenue.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. INTERACTION MATRIX SPARSITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n═══ 9. Interaction matrix sparsity ═══")
matrix_path = "data/processed/interaction_matrix.npz"
if os.path.exists(matrix_path):
    from scipy.sparse import load_npz
    mat = load_npz(matrix_path)
    density = 100.0 * mat.nnz / (mat.shape[0] * mat.shape[1])
    print(f"  Shape   : {mat.shape[0]:,} users × {mat.shape[1]:,} products")
    print(f"  Non-zero: {mat.nnz:,}")
    print(f"  Density : {density:.4f}%")

    # Visualise a 100×30 sub-matrix
    sub = mat[:100, :].toarray()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(sub > 0, cmap="Blues", cbar=False, ax=ax, linewidths=0, linecolor="none")
    ax.set_xlabel("Products")
    ax.set_ylabel("Customers (first 100)")
    ax.set_title(f"User–Item Interaction Matrix Sparsity (density={density:.3f}%)")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.tight_layout()
    save(fig, "interaction_matrix_sparsity.png")


print("\n✅  EDA complete. Figures saved to notebooks/figures/")
