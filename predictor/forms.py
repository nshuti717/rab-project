from django import forms
from django.contrib.auth.models import User
from .models import SeedApplication


class UserRegistrationForm(forms.ModelForm):
    password  = forms.CharField(widget=forms.PasswordInput, min_length=6)
    password2 = forms.CharField(widget=forms.PasswordInput, label="Confirm Password")

    class Meta:
        model  = User
        fields = ['username', 'email', 'password']

    def clean(self):
        cleaned = super().clean()
        p1 = cleaned.get('password')
        p2 = cleaned.get('password2')
        if p1 and p2 and p1 != p2:
            raise forms.ValidationError("Passwords do not match.")
        return cleaned


class SeedApplicationForm(forms.ModelForm):
    class Meta:
        model  = SeedApplication
        fields = ['full_name', 'national_id', 'district', 'seed_type', 'land_size', 'notes']
        widgets = {
            'notes':     forms.Textarea(attrs={'rows': 3}),
            'land_size': forms.NumberInput(attrs={'step': '0.1', 'min': '0.1'}),
        }
