from django.db import models
from django.contrib.auth.models import User


class SeedApplication(models.Model):
    STATUS_CHOICES = [
        ('pending',  '⏳ Pending'),
        ('approved', '✅ Approved'),
        ('rejected', '❌ Rejected'),
    ]
    SEED_CHOICES = [
        ('Maize', 'Maize'),
        ('Beans', 'Beans'),
        ('Irish Potato', 'Irish Potato'),
        ('Wheat', 'Wheat'),
        ('Sorghum', 'Sorghum'),
        ('Rice', 'Rice'),
    ]
    DISTRICT_CHOICES = [
        ('Huye', 'Huye'),
        ('Musanze', 'Musanze'),
        ('Karongi', 'Karongi'),
        ('Kirehe', 'Kirehe'),
        ('Kigali', 'Kigali'),
        ('Nyagatare', 'Nyagatare'),
        ('Rubavu', 'Rubavu'),
    ]

    user        = models.ForeignKey(User, on_delete=models.CASCADE, related_name='applications')
    full_name   = models.CharField(max_length=200)
    national_id = models.CharField(max_length=50)
    district    = models.CharField(max_length=100, choices=DISTRICT_CHOICES)
    seed_type   = models.CharField(max_length=100, choices=SEED_CHOICES)
    land_size   = models.FloatField(help_text="In hectares")
    notes       = models.TextField(blank=True)
    status      = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.full_name} — {self.seed_type} ({self.district})"

    def recommended_seeds_kg(self):
        """Rough seed quantity based on crop type and land size."""
        rates = {
            'Maize': 25, 'Beans': 60, 'Irish Potato': 1500,
            'Wheat': 120, 'Sorghum': 10, 'Rice': 80,
        }
        return round(rates.get(self.seed_type, 50) * self.land_size, 1)


class PricePrediction(models.Model):
    SEASON_CHOICES = [('A', 'Season A (Sep–Feb)'), ('B', 'Season B (Mar–Jul)')]
    MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    user            = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    crop            = models.CharField(max_length=100)
    district        = models.CharField(max_length=100)
    season          = models.CharField(max_length=2, choices=SEASON_CHOICES)
    month           = models.IntegerField()
    year            = models.IntegerField()
    predicted_price = models.FloatField(help_text="RWF per kg")
    advice          = models.TextField(blank=True)
    created_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def month_name(self):
        return self.MONTH_NAMES[self.month - 1]

    def __str__(self):
        return f"{self.crop} | {self.district} | {self.month_name()} {self.year} → {self.predicted_price:.0f} RWF/kg"
