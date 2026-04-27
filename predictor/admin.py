from django.contrib import admin
from .models import SeedApplication, PricePrediction


@admin.register(SeedApplication)
class SeedApplicationAdmin(admin.ModelAdmin):
    list_display  = ('full_name', 'seed_type', 'district', 'land_size', 'status', 'created_at')
    list_filter   = ('status', 'seed_type', 'district')
    search_fields = ('full_name', 'national_id')
    list_editable = ('status',)
    ordering      = ('-created_at',)


@admin.register(PricePrediction)
class PricePredictionAdmin(admin.ModelAdmin):
    list_display = ('user', 'crop', 'district', 'season', 'month', 'year', 'predicted_price', 'created_at')
    list_filter  = ('crop', 'district', 'season')
    ordering     = ('-created_at',)
