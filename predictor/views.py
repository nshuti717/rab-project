import os
import json
import datetime
import urllib.request
import urllib.error

import joblib
import numpy as np

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.contrib import messages

from .forms import UserRegistrationForm, SeedApplicationForm
from .models import SeedApplication, PricePrediction

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = "AIzaSyBMQav30_-dzt3LwYl3Zwad6e_L3lDCMnA"

# Path to trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'crop_price_model.pkl')

# Load ML bundle once at startup
_ml_bundle = None

def get_ml_bundle():
    global _ml_bundle
    if _ml_bundle is None and os.path.exists(MODEL_PATH):
        _ml_bundle = joblib.load(MODEL_PATH)
    return _ml_bundle


# Market advice thresholds per crop (RWF/kg)
PRICE_ADVICE = {
    'Maize':        {'sell_now': 380, 'hold': 300},
    'Beans':        {'sell_now': 950, 'hold': 750},
    'Irish Potato': {'sell_now': 300, 'hold': 200},
    'Wheat':        {'sell_now': 500, 'hold': 400},
    'Sorghum':      {'sell_now': 350, 'hold': 260},
    'Rice':         {'sell_now': 1000, 'hold': 850},
}

AGENT_SYSTEM_PROMPT = """
You are an expert AI Farming Advisor for Rwanda's Agriculture and Animal Resources Development Board (RAB).
You help Rwandan farmers with practical, accurate, localized advice.

Specializations:
- Crop price trends and best times to sell (Maize, Beans, Irish Potato, Wheat, Sorghum, Rice)
- Optimal planting/harvesting seasons: Season A (Sep–Feb, harvest Jan–Feb) and Season B (Mar–Jul, harvest Jun–Jul)
- Best crops per district: Nyagatare=Maize/Sorghum, Musanze=Potato/Wheat, Karongi/Rubavu=Rice/Beans, Huye=mixed, Kirehe=Rice/Beans, Kigali=all (urban premium prices)
- Seed rates per hectare: Maize 25kg, Beans 60kg, Irish Potato 1500kg, Wheat 120kg, Sorghum 10kg, Rice 80kg
- Soil, fertilizer, and climate-smart practices for Rwanda's hilly terrain
- RAB subsidy programs, cooperative systems, and nearby market centers

Response style:
- Simple, clear English (greet warmly in Kinyarwanda when appropriate: "Muraho", "Murakoze")
- Practical and actionable — include specific RWF price ranges when relevant
- Encouraging and supportive
- Keep answers concise (4–8 sentences usually enough)
"""

# ─────────────────────────────────────────────────────────────────────────────
# AUTH VIEWS
# ─────────────────────────────────────────────────────────────────────────────

def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        user = authenticate(
            request,
            username=request.POST.get('username'),
            password=request.POST.get('password'),
        )
        if user:
            login(request, user)
            return redirect(request.GET.get('next', 'home'))
        return render(request, 'login.html', {'error': 'Invalid username or password.'})
    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


def register(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
            messages.success(request, f"Welcome, {user.username}! Your account has been created.")
            return redirect('home')
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})


# ─────────────────────────────────────────────────────────────────────────────
# HOME / DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def home(request):
    user_apps   = SeedApplication.objects.filter(user=request.user)
    user_preds  = PricePrediction.objects.filter(user=request.user)
    latest_apps  = user_apps[:3]
    latest_preds = user_preds[:5]

    stats = {
        'total_apps':    user_apps.count(),
        'approved_apps': user_apps.filter(status='approved').count(),
        'pending_apps':  user_apps.filter(status='pending').count(),
        'total_preds':   user_preds.count(),
    }

    # Chart data for predictions bar chart
    chart_labels = json.dumps([f"{p.crop} / {p.district}" for p in latest_preds])
    chart_values = json.dumps([float(p.predicted_price) for p in latest_preds])

    return render(request, 'home.html', {
        'stats':        stats,
        'latest_apps':  latest_apps,
        'latest_preds': latest_preds,
        'chart_labels': chart_labels,
        'chart_values': chart_values,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SEED APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def apply_seed(request):
    if request.method == 'POST':
        form = SeedApplicationForm(request.POST)
        if form.is_valid():
            app = form.save(commit=False)
            app.user = request.user
            app.save()
            messages.success(request, "✅ Your seed application has been submitted successfully!")
            return redirect('my_applications')
        messages.error(request, "Please correct the errors below.")
    else:
        form = SeedApplicationForm()
    return render(request, 'apply_seed.html', {'form': form})


@login_required
def my_applications(request):
    apps = SeedApplication.objects.filter(user=request.user)
    return render(request, 'my_applications.html', {'apps': apps})


# ─────────────────────────────────────────────────────────────────────────────
# CROP PRICE PREDICTION (ML)
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def predict_price(request):
    bundle    = get_ml_bundle()
    crops     = bundle['crops']     if bundle else []
    districts = bundle['districts'] if bundle else []

    result = None
    if request.method == 'POST':
        crop     = request.POST.get('crop', '')
        district = request.POST.get('district', '')
        month    = int(request.POST.get('month', 1))
        season   = request.POST.get('season', 'A')
        year     = int(request.POST.get('year', datetime.date.today().year))

        if bundle and crop in crops and district in districts:
            feat = np.array([[
                bundle['crop_enc'].transform([crop])[0],
                bundle['district_enc'].transform([district])[0],
                month,
                bundle['season_enc'].transform([season])[0],
                year,
            ]])
            price = round(bundle['model'].predict(feat)[0])

            # Market advice
            thresholds = PRICE_ADVICE.get(crop, {'sell_now': 9999, 'hold': 0})
            if price >= thresholds['sell_now']:
                advice_label = 'excellent'
                advice_text  = f"🔥 Excellent time to sell! {crop} price is very high in {district}. Go to market now."
                advice_color = '#1a7e42'
            elif price >= thresholds['hold']:
                advice_label = 'good'
                advice_text  = f"✅ Good price. Consider selling now or wait 2–4 weeks for possible improvement."
                advice_color = '#2b8c5e'
            else:
                advice_label = 'wait'
                advice_text  = f"⏳ Price is below average. If possible, store your {crop} and wait for the market to improve."
                advice_color = '#e07b00'

            MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
                           'Jul','Aug','Sep','Oct','Nov','Dec']

            result = {
                'price':        price,
                'crop':         crop,
                'district':     district,
                'month_name':   MONTH_NAMES[month - 1],
                'year':         year,
                'season':       season,
                'advice_text':  advice_text,
                'advice_label': advice_label,
                'advice_color': advice_color,
            }

            # Save prediction to DB
            pred_obj = PricePrediction.objects.create(
                user=request.user,
                crop=crop,
                district=district,
                season=season,
                month=month,
                year=year,
                predicted_price=price,
                advice=advice_text,
            )
            result['pred_id'] = pred_obj.pk

    # History for the current user
    history = PricePrediction.objects.filter(user=request.user)[:8]

    return render(request, 'predict_price.html', {
        'crops':     crops,
        'districts': districts,
        'months':    list(range(1, 13)),
        'years':     list(range(2024, 2028)),
        'result':    result,
        'history':   history,
    })


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def export_prediction_pdf(request, pk):
    pred = get_object_or_404(PricePrediction, pk=pk, user=request.user)
    MONTH_NAMES = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']
    thresholds = {
        'Maize':        {'sell_now': 380,  'hold': 300},
        'Beans':        {'sell_now': 950,  'hold': 750},
        'Irish Potato': {'sell_now': 300,  'hold': 200},
        'Wheat':        {'sell_now': 500,  'hold': 400},
        'Sorghum':      {'sell_now': 350,  'hold': 260},
        'Rice':         {'sell_now': 1000, 'hold': 850},
    }
    t = thresholds.get(pred.crop, {'sell_now': 9999, 'hold': 0})
    if pred.predicted_price >= t['sell_now']:
        advice       = "SELL NOW — Price is at a high point. Take your crop to market immediately."
        advice_color = "#16a34a"
    elif pred.predicted_price >= t['hold']:
        advice       = "GOOD PRICE — Selling now is reasonable. You may also wait 2–4 weeks."
        advice_color = "#0369a1"
    else:
        advice       = "WAIT — Price is below average. Store your crop and wait for market recovery."
        advice_color = "#d97706"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>RAB Price Prediction Report</title>
<style>
  @media print {{ .no-print {{ display:none }} body {{ margin:0 }} }}
  body {{ font-family:Arial,sans-serif; color:#1e2a2f; padding:40px; max-width:700px; margin:0 auto }}
  .header {{ background:#16a34a; color:white; padding:24px 28px; border-radius:12px; margin-bottom:24px }}
  .header h1 {{ margin:0; font-size:22px }}
  .header p  {{ margin:6px 0 0; opacity:.85; font-size:13px }}
  .price-box {{ background:#f0fdf4; border:2px solid #16a34a; border-radius:12px; padding:20px; text-align:center; margin-bottom:20px }}
  .price-big {{ font-size:48px; font-weight:bold; color:#14532d; margin:0 }}
  .price-label {{ font-size:13px; color:#6b7280; margin-top:4px }}
  .advice {{ padding:14px 18px; border-radius:10px; font-weight:600; margin-bottom:20px;
             border-left:5px solid {advice_color}; background:{advice_color}18; color:{advice_color} }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:20px }}
  td {{ padding:10px 14px; border-bottom:1px solid #e5e7eb; font-size:14px }}
  td:first-child {{ font-weight:600; color:#166534; width:40% }}
  .footer {{ font-size:11px; color:#9ca3af; text-align:center; margin-top:32px;
             padding-top:16px; border-top:1px solid #e5e7eb }}
  .print-btn {{ background:#16a34a; color:white; border:none; padding:12px 28px;
                border-radius:8px; font-size:15px; font-weight:bold; cursor:pointer; margin-bottom:24px }}
</style></head><body>
<button class="print-btn no-print" onclick="window.print()">🖨️ Print / Save as PDF</button>
<div class="header">
  <h1>🌿 RAB Crop Price Prediction Report</h1>
  <p>Rwanda Agriculture &amp; Animal Resources Development Board &nbsp;·&nbsp;
     Generated {datetime.date.today().strftime("%d %B %Y")}</p>
</div>
<div class="price-box">
  <div class="price-label">Predicted Market Price</div>
  <div class="price-big">{int(pred.predicted_price):,}</div>
  <div class="price-label">RWF per kilogram</div>
</div>
<div class="advice">{advice}</div>
<table>
  <tr><td>Farmer</td><td>{pred.user.username}</td></tr>
  <tr><td>Crop</td><td>{pred.crop}</td></tr>
  <tr><td>District</td><td>{pred.district}</td></tr>
  <tr><td>Season</td><td>Season {pred.season}</td></tr>
  <tr><td>Month / Year</td><td>{MONTH_NAMES[pred.month - 1]} {pred.year}</td></tr>
  <tr><td>Predicted Price</td><td><strong>{int(pred.predicted_price):,} RWF/kg</strong></td></tr>
  <tr><td>Report Generated</td><td>{datetime.date.today().strftime("%d %B %Y")}</td></tr>
</table>
<div class="footer">
  RAB Smart Farming Platform &nbsp;·&nbsp; Rwanda Agriculture Board &nbsp;·&nbsp;
  Rubona, Huye District<br>
  Toll free: 4675 &nbsp;·&nbsp; info@rab.gov.rw &nbsp;·&nbsp;
  This prediction is generated by an ML model (R²=98%)
</div>
</body></html>"""
    return HttpResponse(html)


# ─────────────────────────────────────────────────────────────────────────────
# AI AGENT
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def agent_page(request):
    return render(request, 'agent.html')


@login_required
@require_POST
def agent_chat(request):
    try:
        body    = json.loads(request.body)
        message = body.get('message', '').strip()

        if not message:
            return JsonResponse({'error': 'Empty message'}, status=400)

        full_prompt = AGENT_SYSTEM_PROMPT + "\n\nFarmer question: " + message

        payload = json.dumps({
            "contents": [{
                "parts": [{"text": full_prompt}]
            }]
        }).encode('utf-8')

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        )

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        reply = data['candidates'][0]['content']['parts'][0]['text']
        return JsonResponse({'reply': reply})

    except urllib.error.HTTPError as e:
        code = e.code
        if code == 503:
            return JsonResponse({
                'reply': '⏳ The AI is temporarily busy. Please wait 10 seconds and try again.'
            })
        elif code == 429:
            return JsonResponse({
                'reply': '⏳ Too many requests sent. Please wait 30 seconds and try again.'
            })
        else:
            return JsonResponse({
                'reply': f'⚠️ API error {code}. Please try again in a moment.'
            })
    except Exception as e:
        return JsonResponse({
            'reply': f'⚠️ Something went wrong: {str(e)}'
        })
