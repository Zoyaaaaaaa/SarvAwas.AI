import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, Any, List

def clean_text(text):
    if not text:
        return "Unknown"
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s-]', '', text)
    return text.title()

def preprocess_input(data: Dict[str, Any], metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform raw request data into robust model-compatible features.
    Matches the logic in train_robust.py
    """
    feature_cols = metadata['feature_cols']
    label_encoders = metadata['label_encoders']
    
    # helper for safe division
    def safe_div(a, b):
        return a / b if b and b > 0 else 0

    # 1. Parse Basics
    total_units = max(1, data.get('total_units', 1))
    booked_units = data.get('booked_units', 0)
    
    # 2. Variable Engineering
    
    # Market Demand
    booking_ratio = min(1.0, safe_div(booked_units, total_units))
    inventory_left = max(0, total_units - booked_units)
    market_absorption_risk = 1 if booking_ratio < 0.2 else 0
    high_demand_signal = 1 if booking_ratio > 0.8 else 0
    
    # Structural
    fsi_intensity = float(data.get('fsi', 0))
    # fsi_category logic from training: bins=[-1, 1, 2, 3, 100], labels=[0, 1, 2, 3]
    if fsi_intensity <= 1: fsi_cat = 0
    elif fsi_intensity <= 2: fsi_cat = 1
    elif fsi_intensity <= 3: fsi_cat = 2
    else: fsi_cat = 3
    
    vertical_complexity = int(data.get('floors', 1))
    is_high_rise = 1 if vertical_complexity > 10 else 0
    
    # Legal
    cases = int(data.get('cases', 0))
    legal_risk_score = cases
    has_litigation = 1 if cases > 0 else 0
    
    # Timeline
    # Try to calculate from dates first
    delay_days = 0
    extension_days = 0
    
    fmt = "%Y-%m-%d"
    today = datetime.now()
    
    proposed = data.get('proposed_completion_date')
    revised = data.get('revised_completion_date')
    extended = data.get('extended_completion_date')
    
    # Simple parsing if string
    def parse_date(d):
        if not d: return None
        try: return datetime.strptime(d[:10], fmt)
        except: return None
        
    d_prop = parse_date(proposed)
    d_rev = parse_date(revised)
    d_ext = parse_date(extended)
    
    if d_prop and d_rev:
        delay_days = max(0, (d_rev - d_prop).days)
    
    if d_prop and d_ext:
        extension_days = max(0, (d_ext - d_prop).days)
        
    # Manual overrides if provided in request
    if data.get('has_delay') is True:
        has_delay = 1
        if delay_days == 0: delay_days = 180 # assume 6 months diff if flag is strictly set
    else:
        has_delay = 1 if delay_days > 0 else 0
        
    if data.get('has_extension') is True:
        has_extension = 1
    else:
        has_extension = 1 if extension_days > 0 else 0
        
    # Delay Severity (bins=[-1, 0, 180, 365, 730, 10000], labels=[0, 1, 2, 3, 4])
    if delay_days <= 0: delay_sev = 0
    elif delay_days <= 180: delay_sev = 1
    elif delay_days <= 365: delay_sev = 2
    elif delay_days <= 730: delay_sev = 3
    else: delay_sev = 4
    
    # Project Scale (bins=[0, 20, 50, 100, 500, 10000], labels=[0, 1, 2, 3, 4])
    # Note: training handled NaN with -1 category, here we assume clean input
    if total_units <= 20: proj_scale = 0
    elif total_units <= 50: proj_scale = 1
    elif total_units <= 100: proj_scale = 2
    elif total_units <= 500: proj_scale = 3
    else: proj_scale = 4
    
    # Text Features
    p_name = clean_text(data.get('project_name', ''))
    proj_words = len(p_name.split())
    proj_len = len(p_name)
    
    # District Encoding & Aggregations
    district = data.get('district', 'Unknown')
    dist_val = -1
    if district in label_encoders['district'].classes_:
        dist_val = label_encoders['district'].transform([district])[0]
        
    # Heuristics for district averages (since we lack lookup table currently)
    # These are neutral/safe defaults to allow model to run
    dist_avg_delay = 0 
    dist_avg_booking = 0.5 
    # If we had the stats, we'd look them up here. 
    # For robust inference without external DB, we accept this approximation.
    
    # Construct DataFrame with exactly the column names model requires
    # We populate ALL possible columns to be safe, then only return the requested ones
    
    raw_features = {
        'booking_ratio': booking_ratio,
        'inventory_left': inventory_left,
        'fsi_intensity': fsi_intensity,
        'vertical_complexity': vertical_complexity,
        'legal_risk_score': legal_risk_score,
        'has_delay': has_delay,
        'has_extension': has_extension,
        'project_scale': proj_scale,
        'delay_severity': delay_sev,
        'market_absorption_risk': market_absorption_risk,
        'high_demand_signal': high_demand_signal,
        'is_high_rise': is_high_rise,
        'has_litigation': has_litigation,
        'fsi_category': fsi_cat,
        'project_name_word_count': proj_words,
        'project_name_length': proj_len,
        'district': dist_val,
        'district_avg_delay_days': dist_avg_delay,
        'district_avg_booking_ratio': dist_avg_booking
        # Add any others that might be in feature_cols
    }
    
    # Create DF
    df = pd.DataFrame([raw_features])
    
    # Ensure all required columns are present, fill with 0 if missing
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Return correctly ordered columns
    return df[feature_cols]

def generate_verdict_reasons(prob: float, data: Dict[str, Any]) -> List[str]:
    reasons = []
    
    # Booking
    total = max(1, data.get('total_units', 1))
    booked = data.get('booked_units', 0)
    ratio = booked / total
    
    if ratio > 0.8:
        reasons.append(f"Strong market confidence ({ratio*100:.0f}% sold)")
    elif ratio < 0.3:
        reasons.append(f"High inventory risk (only {ratio*100:.0f}% sold)")
        
    # Legal
    cases = data.get('cases', 0)
    if cases > 0:
        reasons.append(f"Legal flag: {cases} active litigation(s)")
    else:
        reasons.append("Clean legal compliance record")
        
    # Timeline
    dates_found = data.get('proposed_completion_date') and data.get('revised_completion_date')
    if data.get('has_delay') or (dates_found and data.get('proposed_completion_date') != data.get('revised_completion_date')):
        reasons.append("Project timeline has been revised")
        
    if prob > 0.80:
        reasons.append("High probability of on-time completion")
        
    return reasons[:4]
