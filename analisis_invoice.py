"""
Studi Kasus: Analisis Keterlambatan Pembayaran Invoice PT Maju Bersama (fiktif)
Berdasarkan kuliah tamu Informatics Practitioner Lecture Series 2026 (Andria Arisal, BRIN)

Alur mengikuti CRISP-DM:
  1. Business Understanding  -> README.md
  2. Data Understanding      -> generate_raw_data() + eksplorasi singkat
  3. Data Preparation        -> clean_data(), build_features()
  4. Modeling                -> train_models()
  5. Evaluation              -> evaluasi di train_models()
  6. Deployment (simulasi)   -> prescriptive() menghasilkan rekomendasi tindakan

Empat level analitik:
  Deskriptif | Diagnostik | Prediktif | Preskriptif
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path("output")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


# ----------------------------------------------------------------------
# 2. DATA UNDERSTANDING: data sintetis yang sengaja "berantakan"
# ----------------------------------------------------------------------
def generate_raw_data(n=3000):
    """Membuat data invoice fiktif dengan masalah kualitas data seperti di lapangan."""
    n_cust = 150
    cust = pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(1, n_cust + 1)],
        "segment": rng.choice(["Korporasi", "UKM", "Pemerintah", "Ritel"], n_cust,
                              p=[.30, .35, .15, .20]),
        "region": rng.choice(["Sumatera", "Jawa", "Kalimantan", "Sulawesi"], n_cust),
        "cust_risk": rng.normal(0, 1, n_cust),  # sifat laten pelanggan
    })

    df = pd.DataFrame({"customer_id": rng.choice(cust.customer_id, n)}).merge(cust, on="customer_id")
    df["invoice_date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 730, len(df)), unit="D")
    df["payment_term"] = rng.choice([14, 30, 45, 60], len(df), p=[.2, .4, .25, .15])
    df["amount"] = np.round(rng.lognormal(16.5, 0.9, len(df)), -3)
    df["due_date"] = df["invoice_date"] + pd.to_timedelta(df["payment_term"], unit="D")

    # proses pembangkit label (tersembunyi): kapan invoice terlambat dibayar
    seg_eff = df["segment"].map({"UKM": 0.7, "Ritel": 0.3, "Korporasi": 0.0, "Pemerintah": 0.4})
    logit = (-1.3 + 0.8 * df["cust_risk"] + seg_eff
             + 0.35 * (np.log(df["amount"]) - 16.5)
             - 0.012 * (df["payment_term"] - 30))
    p_late = 1 / (1 + np.exp(-logit))
    late = rng.random(len(df)) < p_late
    days = np.where(late, 1 + rng.exponential(12, len(df)), -rng.integers(0, 10, len(df)))
    df["payment_date"] = df["due_date"] + pd.to_timedelta(np.round(days), unit="D")

    # invoice terbaru belum dibayar (outstanding)
    cutoff = pd.Timestamp("2025-12-31")
    df.loc[df["payment_date"] > cutoff, "payment_date"] = pd.NaT
    df = df.drop(columns=["cust_risk"])

    # ---- sengaja dirusak ----
    df["invoice_id"] = [f"INV-{i:05d}" for i in range(len(df))]
    df["segment"] = df["segment"].map(lambda s: rng.choice([s, s.upper(), s.lower(), f" {s} "],
                                                          p=[.7, .1, .1, .1]))
    df["amount"] = df["amount"].astype(object)
    miss = rng.choice(df.index, int(.02 * len(df)), replace=False)
    df.loc[miss, "amount"] = np.nan                              # nilai hilang
    txt = rng.choice(df.index.difference(miss), int(.05 * len(df)), replace=False)
    df.loc[txt, "amount"] = df.loc[txt, "amount"].map(lambda v: f"Rp {int(v):,}".replace(",", "."))
    typo = rng.choice(df.index.difference(miss).difference(txt), 8, replace=False)
    df.loc[typo, "amount"] = df.loc[typo, "amount"] * 1000       # salah ketik (outlier)
    df = pd.concat([df, df.sample(int(.01 * len(df)), random_state=1)])  # duplikat
    return df.sample(frac=1, random_state=7).reset_index(drop=True)


def explore(df):
    print("=== DATA UNDERSTANDING ===")
    print("Ukuran data      :", df.shape)
    print("Nilai hilang     :\n", df.isna().sum()[lambda s: s > 0])
    print("Baris duplikat   :", df.duplicated().sum())
    print("Variasi segment  :", sorted(df["segment"].unique()), "\n")


# ----------------------------------------------------------------------
# 3. DATA PREPARATION (tahap paling krusial & memakan waktu)
# ----------------------------------------------------------------------
def clean_data(df):
    df = df.drop_duplicates().copy()

    # standarisasi teks kategorikal
    df["segment"] = df["segment"].str.strip().str.title().replace({"Ukm": "UKM"})

    # amount: string "Rp 1.250.000" -> angka
    df["amount"] = (df["amount"].astype(str)
                    .str.replace("Rp", "", regex=False)
                    .str.replace(".", "", regex=False)
                    .str.strip()
                    .replace({"nan": np.nan}))
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # outlier salah ketik: batasi di persentil 99 per segmen
    cap = df.groupby("segment")["amount"].transform(lambda s: s.quantile(.99))
    df["amount"] = df["amount"].clip(upper=cap)

    # imputasi nilai hilang dengan median segmen
    df["amount"] = df["amount"].fillna(df.groupby("segment")["amount"].transform("median"))

    # label: terlambat jika dibayar setelah jatuh tempo (NaN = belum dibayar)
    df["days_late"] = (df["payment_date"] - df["due_date"]).dt.days
    df["is_late"] = np.where(df["payment_date"].isna(), np.nan, (df["days_late"] > 0).astype(float))
    df["is_paid"] = df["payment_date"].notna()
    return df.sort_values("invoice_date").reset_index(drop=True)


def build_features(df):
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df["invoice_month"] = df["invoice_date"].dt.month
    # riwayat pelanggan SEBELUM invoice ini (menghindari data leakage)
    df["prev_late_rate"] = df.groupby("customer_id")["is_late"].transform(
        lambda s: s.shift(1).expanding().mean())
    df["prev_late_rate"] = df["prev_late_rate"].fillna(df.loc[df.is_paid, "is_late"].mean())
    df["prev_invoice_count"] = df.groupby("customer_id").cumcount()
    return df


FEATURES_NUM = ["log_amount", "payment_term", "invoice_month", "prev_late_rate", "prev_invoice_count"]
FEATURES_CAT = ["segment", "region"]


# ----------------------------------------------------------------------
# 4 LEVEL ANALITIK
# ----------------------------------------------------------------------
def descriptive(df):
    """Apa yang terjadi? Tren pendapatan vs invoice terlambat / belum terbayar."""
    d = df.copy()
    d["ym"] = d["invoice_date"].dt.to_period("Q").astype(str)
    q = d.groupby("ym").agg(
        pendapatan=("amount", "sum"),
        terlambat=("amount", lambda s: s[d.loc[s.index, "is_late"] == 1].sum()),
        belum_dibayar=("amount", lambda s: s[~d.loc[s.index, "is_paid"]].sum()),
    ) / 1e9
    q["persen_terlambat"] = q["terlambat"] / q["pendapatan"] * 100
    q.round(2).to_csv(OUT / "1_deskriptif_per_kuartal.csv")
    print("=== 1. DESKRIPTIF (miliar Rp) ===\n", q.round(2), "\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    q[["pendapatan", "terlambat", "belum_dibayar"]].plot(kind="bar", ax=ax[0])
    ax[0].set_title("Pendapatan vs invoice terlambat/belum dibayar (Rp miliar)")
    ax[0].set_xlabel("Kuartal")
    q["persen_terlambat"].plot(marker="o", ax=ax[1], color="crimson")
    ax[1].set_title("% nilai invoice terlambat per kuartal")
    ax[1].set_xlabel("Kuartal")
    plt.tight_layout()
    plt.savefig(OUT / "1_deskriptif.png", dpi=130)
    plt.close()


def diagnostic(df):
    """Mengapa terjadi? Bedah tingkat keterlambatan menurut berbagai dimensi."""
    paid = df[df.is_paid].copy()
    paid["rentang_nominal"] = pd.qcut(paid["amount"], 4,
                                      labels=["Q1 (kecil)", "Q2", "Q3", "Q4 (besar)"])
    dims = ["segment", "payment_term", "rentang_nominal", "region"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    print("=== 2. DIAGNOSTIK (% invoice terlambat) ===")
    for ax, dim in zip(axes.ravel(), dims):
        rate = (paid.groupby(dim, observed=True)["is_late"].mean() * 100).round(1)
        print(f"\nBerdasarkan {dim}:\n{rate}")
        rate.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"% terlambat per {dim}")
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "2_diagnostik.png", dpi=130)
    plt.close()
    print()


def predictive(df):
    """Apa yang akan terjadi? Model prediksi invoice berpotensi late payment."""
    paid = df[df.is_paid].copy()
    X, y = paid[FEATURES_NUM + FEATURES_CAT], paid["is_late"].astype(int)
    # split berdasarkan waktu (data lama = latih, data baru = uji) -> lebih realistis
    cut = int(len(paid) * 0.8)
    X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

    prep = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                                class_weight="balanced", random_state=42),
    }
    print("=== 3. PREDIKTIF ===")
    print(f"Data latih: {len(X_tr)} | Data uji: {len(X_te)} | Proporsi terlambat: {y.mean():.1%}\n")
    results, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rep = classification_report(y_te, pred, output_dict=True)
        results.append({"model": name, "ROC-AUC": roc_auc_score(y_te, proba),
                        "precision": rep["1"]["precision"], "recall": rep["1"]["recall"],
                        "f1": rep["1"]["f1-score"], "accuracy": rep["accuracy"]})
        fitted[name] = pipe
        print(f"--- {name} ---\n{confusion_matrix(y_te, pred)}\n")
    res = pd.DataFrame(results).set_index("model").round(3)
    res.to_csv(OUT / "3_evaluasi_model.csv")
    print(res, "\n")

    # pentingnya fitur (Random Forest)
    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).nlargest(10)
    imp.sort_values().plot(kind="barh", figsize=(7, 4), color="darkorange")
    plt.title("10 fitur terpenting (Random Forest)")
    plt.tight_layout()
    plt.savefig(OUT / "3_feature_importance.png", dpi=130)
    plt.close()

    best = res["ROC-AUC"].idxmax()
    print(f"Model terbaik berdasarkan ROC-AUC: {best}\n")
    return fitted[best]


def prescriptive(df, model):
    """Apa yang harus dilakukan? Rekomendasi tindakan untuk invoice yang belum dibayar."""
    open_inv = df[~df.is_paid].copy()
    open_inv["risk_score"] = model.predict_proba(open_inv[FEATURES_NUM + FEATURES_CAT])[:, 1]

    def action(row):
        if row.risk_score >= 0.55 and row.amount >= open_inv.amount.median():
            return "PRIORITAS 1: telepon PIC + eskalasi ke manajer piutang"
        if row.risk_score >= 0.55:
            return "PRIORITAS 2: telepon PIC pelanggan"
        if row.risk_score >= 0.40:
            return "Kirim email pengingat H-7 dan H-1"
        return "Pengingat otomatis standar"

    open_inv["rekomendasi"] = open_inv.apply(action, axis=1)
    cols = ["invoice_id", "customer_id", "segment", "amount", "due_date", "risk_score", "rekomendasi"]
    result = open_inv.sort_values("risk_score", ascending=False)[cols]
    result.round({"risk_score": 3}).to_csv(OUT / "4_rekomendasi_tindakan.csv", index=False)
    print("=== 4. PRESKRIPTIF ===")
    print(f"Invoice belum dibayar: {len(result)}")
    print(result["rekomendasi"].value_counts(), "\n")
    print(result.head(5).to_string(index=False))


# ----------------------------------------------------------------------
def main():
    raw = generate_raw_data()
    raw.to_csv(OUT / "0_data_mentah.csv", index=False)
    explore(raw)

    df = build_features(clean_data(raw))
    df.to_csv(OUT / "0_data_bersih.csv", index=False)

    descriptive(df)
    diagnostic(df)
    model = predictive(df)
    prescriptive(df, model)
    print(f"\nSelesai. Semua hasil tersimpan di folder '{OUT}/'.")


if __name__ == "__main__":
    main()
