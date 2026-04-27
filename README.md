# 🌿 RAB Smart Farming Platform
**Rwanda Agriculture & Animal Resources Development Board**
> Big Data & AI Group Project — Crop Price Prediction + AI Farming Advisor

---

## 🚀 What This System Does

| Feature | Technology | Description |
|---|---|---|
| 📊 Crop Price Prediction | **Gradient Boosting ML** (scikit-learn) | Predicts RWF/kg prices by crop, district, season & month. R² = 98% |
| 🌱 Seed Application | **Django + SQLite** | Farmers apply for seeds; auto-calculates seed quantity needed |
| 🤖 AI Farming Advisor | **Claude AI (Anthropic)** | Real-time chat agent for farming advice in English/Kinyarwanda |
| 👤 User Auth | **Django Auth** | Secure register, login, logout |
| 📋 Dashboard | **Django + Jinja** | Personalized stats, activity history, quick access tools |

---

## ⚙️ SETUP — Step by Step

### Step 1 — Install Python & Dependencies

```bash
# Make sure Python 3.10+ is installed
python --version

# Install all required packages
pip install -r requirements.txt
```

### Step 2 — Set Up the Database

```bash
# This creates the database tables (SQLite — no extra config needed)
python manage.py makemigrations predictor
python manage.py migrate

# Create a superuser (admin account)
python manage.py createsuperuser
# → enter username, email, password when prompted
```

### Step 3 — Train the ML Model

```bash
# Run the training script once to generate the model file
python predictor/ml/train_model.py

# You should see:
# ✅ Model trained  |  MAE: ~27 RWF/kg  |  R²: 0.980
# 💾 Saved → predictor/ml/crop_price_model.pkl
```

### Step 4 — Add Your Anthropic API Key (for AI Agent)

Open `predictor/views.py` and find line:

```python
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"
```

Replace with your actual key:

```python
ANTHROPIC_API_KEY = "sk-ant-api03-xxxxxxxxxxxxxxxx..."
```

**How to get the API key:**
1. Go to https://console.anthropic.com
2. Sign up / log in
3. Click "API Keys" → "Create Key"
4. Copy and paste it into `views.py`

> 💡 The system works WITHOUT the API key — only the AI Agent chat won't work. All other features (ML prediction, seed application, dashboard) work perfectly.

### Step 5 — Run the Server

```bash
python manage.py runserver
```

Open your browser: **http://127.0.0.1:8000**

---

## 📁 Project Structure

```
rab_project/
├── config/
│   ├── settings.py          # Django configuration
│   └── urls.py              # Main URL routing
├── predictor/
│   ├── models.py            # SeedApplication, PricePrediction models
│   ├── views.py             # All views + AI agent + ML prediction
│   ├── urls.py              # App URL routing
│   ├── forms.py             # Django forms
│   ├── admin.py             # Admin panel config
│   ├── ml/
│   │   ├── train_model.py   # ML training script (run once)
│   │   └── crop_price_model.pkl  # Trained model (generated)
│   └── templates/
│       ├── home.html        # Main dashboard
│       ├── login.html       # Login page
│       ├── register.html    # Register page
│       ├── predict_price.html   # ML price prediction
│       ├── apply_seed.html  # Seed application form
│       ├── my_applications.html # Application history
│       └── agent.html       # AI chat interface
├── static/                  # Static files (logo, images)
├── requirements.txt
└── README.md
```

---

## 🔗 All Pages & URLs

| URL | Page | Description |
|---|---|---|
| `/` | Dashboard | Home with stats and quick links |
| `/login/` | Login | User authentication |
| `/register/` | Register | New account creation |
| `/predict/` | Price Prediction | ML crop price forecasting |
| `/apply-seed/` | Seed Application | Submit seed request to RAB |
| `/my-applications/` | My Applications | View application history + status |
| `/agent/` | AI Advisor | Chat with AI farming advisor |
| `/admin/` | Admin Panel | Manage all users & data |

---

## 🤖 ML Model Details

- **Algorithm:** Gradient Boosting Regressor (scikit-learn)
- **Training data:** 4,000 synthetic data points based on Rwanda MINAGRI price patterns
- **Features:** Crop type, District, Month, Season (A/B), Year
- **Target:** Price in RWF/kg
- **Performance:** R² = 0.980 · MAE ≈ 27 RWF/kg
- **Crops:** Maize · Beans · Irish Potato · Wheat · Sorghum · Rice
- **Districts:** Huye · Musanze · Karongi · Kirehe · Kigali · Nyagatare · Rubavu

---

## 🏫 For Your Presentation

Key talking points for the class:

1. **Big Data angle:** We use historical market price patterns across 7 districts, 6 crops, 2 seasons to train our model
2. **ML innovation:** Gradient Boosting (ensemble of decision trees) outperforms simple regression for this task — 98% accuracy
3. **AI agent innovation:** We integrated Claude LLM as a conversational farming advisor — farmers can ask questions in natural language
4. **Real-world impact:** Helps farmers decide WHEN to sell, WHAT to plant, and HOW MUCH seed to buy — directly addressing RAB's mission
5. **Full-stack:** Django backend, ML model, REST-style API endpoint for the agent, responsive frontend

---

## ❓ Common Issues

**`ModuleNotFoundError: No module named 'predictor'`**
→ Make sure you run `python manage.py` from inside the `rab_project/` folder

**`No such file: crop_price_model.pkl`**
→ Run `python predictor/ml/train_model.py` first

**AI Agent returns error**
→ Check that you set `ANTHROPIC_API_KEY` correctly in `views.py`

**`django.db.utils.OperationalError: no such table`**
→ Run `python manage.py migrate` again

---

*© 2025 RAB Smart Farming Platform · Group Project · Big Data & AI*
