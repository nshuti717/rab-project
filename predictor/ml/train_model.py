"""
RAB Crop Price Prediction Model
================================
Run this once to train and save the model:
    python predictor/ml/train_model.py

It generates: predictor/ml/crop_price_model.pkl
"""

import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. RWANDA CROP PRICE KNOWLEDGE BASE ─────────────────────────────────────
# Prices in RWF/kg, based on Rwanda NAEB & MINAGRI market data patterns

CROPS = ['Maize', 'Beans', 'Irish Potato', 'Wheat', 'Sorghum', 'Rice']
DISTRICTS = ['Huye', 'Musanze', 'Karongi', 'Kirehe', 'Kigali', 'Nyagatare', 'Rubavu']

# Base price per crop (RWF/kg at neutral conditions)
BASE_PRICES = {
    'Maize':        320,
    'Beans':        780,
    'Irish Potato': 220,
    'Wheat':        430,
    'Sorghum':      280,
    'Rice':         880,
}

# District price multipliers (supply/demand factors)
DISTRICT_FACTORS = {
    'Maize':        {'Nyagatare': 0.82, 'Huye': 0.95, 'Musanze': 1.05, 'Karongi': 1.02, 'Kirehe': 0.90, 'Kigali': 1.12, 'Rubavu': 1.08},
    'Beans':        {'Nyagatare': 1.10, 'Huye': 0.90, 'Musanze': 0.95, 'Karongi': 1.05, 'Kirehe': 0.88, 'Kigali': 1.15, 'Rubavu': 1.05},
    'Irish Potato': {'Nyagatare': 1.15, 'Huye': 1.00, 'Musanze': 0.72, 'Karongi': 0.90, 'Kirehe': 1.10, 'Kigali': 1.18, 'Rubavu': 0.85},
    'Wheat':        {'Nyagatare': 0.95, 'Huye': 0.92, 'Musanze': 0.88, 'Karongi': 1.05, 'Kirehe': 1.00, 'Kigali': 1.10, 'Rubavu': 1.05},
    'Sorghum':      {'Nyagatare': 0.85, 'Huye': 0.90, 'Musanze': 1.05, 'Karongi': 1.00, 'Kirehe': 0.88, 'Kigali': 1.12, 'Rubavu': 1.00},
    'Rice':         {'Nyagatare': 0.95, 'Huye': 1.00, 'Musanze': 1.08, 'Karongi': 0.80, 'Kirehe': 0.82, 'Kigali': 1.15, 'Rubavu': 0.85},
}

# Monthly price multipliers (harvest months = low, scarcity months = high)
# Season A harvest: Jan-Feb  |  Season B harvest: Jun-Jul
MONTH_FACTORS = {
    'Maize':        [1.12, 0.80, 0.85, 0.92, 1.00, 1.08, 0.82, 0.90, 0.95, 1.05, 1.10, 1.15],
    'Beans':        [0.82, 0.88, 0.92, 1.00, 1.10, 0.80, 0.85, 0.90, 0.95, 1.08, 1.12, 1.05],
    'Irish Potato': [0.85, 0.88, 0.92, 1.00, 1.10, 0.78, 0.82, 0.90, 0.95, 1.05, 1.12, 1.08],
    'Wheat':        [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 0.82, 0.88, 0.92, 1.02, 1.08, 1.12],
    'Sorghum':      [0.88, 0.82, 0.90, 1.00, 1.05, 1.10, 0.78, 0.85, 0.92, 1.02, 1.08, 1.12],
    'Rice':         [0.90, 0.88, 0.92, 0.95, 1.00, 0.80, 0.82, 0.90, 0.95, 1.05, 1.10, 1.12],
}

# Season modifiers
SEASON_FACTORS = {'A': 1.02, 'B': 0.98}

# ── 2. GENERATE SYNTHETIC TRAINING DATA ─────────────────────────────────────
np.random.seed(42)
records = []

for _ in range(4000):
    crop     = np.random.choice(CROPS)
    district = np.random.choice(DISTRICTS)
    month    = np.random.randint(1, 13)
    season   = 'A' if month in [9, 10, 11, 12, 1, 2] else 'B'
    year     = np.random.randint(2020, 2026)

    # Compute price with all factors + slight yearly inflation + noise
    base   = BASE_PRICES[crop]
    d_fac  = DISTRICT_FACTORS[crop][district]
    m_fac  = MONTH_FACTORS[crop][month - 1]
    s_fac  = SEASON_FACTORS[season]
    yr_fac = 1 + (year - 2020) * 0.04   # ~4% annual inflation
    noise  = np.random.normal(1.0, 0.06)  # ±6% market noise

    price = base * d_fac * m_fac * s_fac * yr_fac * noise
    price = round(max(80, price), 1)

    records.append({
        'crop': crop,
        'district': district,
        'month': month,
        'season': season,
        'year': year,
        'price': price,
    })

# ── 3. ENCODE FEATURES ──────────────────────────────────────────────────────
crop_enc     = LabelEncoder().fit(CROPS)
district_enc = LabelEncoder().fit(DISTRICTS)
season_enc   = LabelEncoder().fit(['A', 'B'])

X, y = [], []
for r in records:
    X.append([
        crop_enc.transform([r['crop']])[0],
        district_enc.transform([r['district']])[0],
        r['month'],
        season_enc.transform([r['season']])[0],
        r['year'],
    ])
    y.append(r['price'])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 4. TRAIN MODEL ───────────────────────────────────────────────────────────
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"✅ Model trained  |  MAE: {mae:.1f} RWF/kg  |  R²: {r2:.3f}")

# ── 5. SAVE MODEL + ENCODERS ─────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
bundle = {
    'model':        model,
    'crop_enc':     crop_enc,
    'district_enc': district_enc,
    'season_enc':   season_enc,
    'crops':        CROPS,
    'districts':    DISTRICTS,
    'base_prices':  BASE_PRICES,
}
out_path = os.path.join(out_dir, 'crop_price_model.pkl')
joblib.dump(bundle, out_path)
print(f"💾 Saved → {out_path}")

# ── 6. QUICK SANITY CHECK ────────────────────────────────────────────────────
print("\n📊 Sample Predictions (RWF/kg):")
samples = [
    ('Maize',        'Nyagatare', 2, 'A', 2025),
    ('Beans',        'Kigali',    11, 'A', 2025),
    ('Irish Potato', 'Musanze',   6,  'B', 2025),
    ('Rice',         'Karongi',   7,  'B', 2025),
]
for crop, dist, month, season, year in samples:
    feat = [[
        crop_enc.transform([crop])[0],
        district_enc.transform([dist])[0],
        month,
        season_enc.transform([season])[0],
        year,
    ]]
    p = model.predict(feat)[0]
    print(f"  {crop:<14} {dist:<12} Month {month:02d} Season {season} → {p:,.0f} RWF/kg")
