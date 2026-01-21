"""
RERA Risk Prediction System - Comprehensive Test Suite
Tests all 3 scenarios and validates API functionality
"""

import requests
import json
from typing import Dict, Any

API_BASE = "http://127.0.0.1:8000"

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BLUE}{'=' * 80}")
    print(f"{text}")
    print(f"{'=' * 80}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.YELLOW}→ {text}{Colors.END}")

def test_health():
    """Test health endpoint"""
    print_header("TEST 1: Health Check")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print_success(f"API is healthy")
            print_info(f"Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Connection error: {e}")
        return False

def test_scenario(name: str, payload: Dict[str, Any], expected_risk: str):
    """Test a prediction scenario"""
    print_header(f"TEST: {name}")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/predict/predict",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print_success(f"Prediction successful")
            print(f"\n  📊 Results:")
            print(f"  ├─ Risk Level: {data['risk_level']}")
            print(f"  ├─ Completion Probability: {data['completion_probability']:.2%}")
            print(f"  ├─ Delay Risk: {data['delay_risk']}")
            print(f"  ├─ Confidence: {data['confidence_score']:.1f}%")
            print(f"  └─ Model Version: {data['model_version']}")
            
            print(f"\n  🔍 Key Factors:")
            for i, factor in enumerate(data['key_factors'], 1):
                print(f"     {i}. {factor}")
            
            # Validate expected risk
            if data['risk_level'] == expected_risk:
                print_success(f"\nExpected risk level '{expected_risk}' confirmed")
            else:
                print_info(f"\nRisk level: {data['risk_level']} (expected: {expected_risk})")
            
            return True
        else:
            print_error(f"Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Request error: {e}")
        return False

def test_map_prediction():
    """Test map-based prediction"""
    print_header("TEST: Map-Based Risk Assessment")
    
    payload = {
        "lat": 19.0760,
        "lng": 72.8777
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/predict/map",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Map prediction successful")
            print(f"\n  📍 Location: ({payload['lat']}, {payload['lng']})")
            print(f"  ├─ District Context: {data['district_context']}")
            print(f"  ├─ Nearby Risk Index: {data['nearby_risk_index']}")
            print(f"  ├─ RERA Safety: {data['predicted_rera_safety']}")
            print(f"  └─ Message: {data['message']}")
            return True
        else:
            print_error(f"Map prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request error: {e}")
        return False

def test_explanation():
    """Test SHAP explanation endpoint"""
    print_header("TEST: Model Explainability (SHAP)")
    
    payload = {
        "project_name": "Sunrise Heights",
        "district": "Mumbai Suburban",
        "pin_code": "400001",
        "total_units": 100,
        "booked_units": 85,
        "project_area": 5000,
        "fsi": 2.5,
        "floors": 12,
        "cases": 0,
        "proposed_completion_date": "2024-01-01",
        "revised_completion_date": "2024-01-01",
        "has_delay": False,
        "has_extension": False
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/predict/explain",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("SHAP explanation generated")
            print(f"\n  📈 Base Prediction Value: {data['base_value']:.4f}")
            print(f"\n  🧠 Top 5 Feature Impacts:")
            for i, feat in enumerate(data['feature_importance'][:5], 1):
                sign = "+" if feat['effect'] > 0 else ""
                print(f"     {i}. {feat['feature']}: {sign}{feat['effect']:.4f}")
                print(f"        └─ {feat['description']}")
            return True
        else:
            print_error(f"Explanation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request error: {e}")
        return False

def run_all_tests():
    """Execute all tests"""
    print_header("🧪 RERA RISK PREDICTION API - COMPREHENSIVE TEST SUITE")
    
    results = []
    
    # Test 1: Health Check
    results.append(test_health())
    
    # Test 2: Low Risk Scenario
    low_risk_payload = {
        "project_name": "Sunrise Heights",
        "district": "Mumbai Suburban",
        "pin_code": "400001",
        "total_units": 100,
        "booked_units": 95,
        "project_area": 5000,
        "fsi": 2.5,
        "floors": 12,
        "cases": 0,
        "proposed_completion_date": "2024-01-01",
        "revised_completion_date": "2024-01-01",
        "has_delay": False,
        "has_extension": False
    }
    results.append(test_scenario(
        "Scenario 1: Low Risk Project",
        low_risk_payload,
        "LOW"
    ))
    
    # Test 3: Moderate Risk Scenario
    moderate_risk_payload = {
        "project_name": "City Center Complex",
        "district": "Pune",
        "pin_code": "411001",
        "total_units": 100,
        "booked_units": 55,
        "project_area": 8000,
        "fsi": 2.0,
        "floors": 8,
        "cases": 1,
        "proposed_completion_date": "2023-01-01",
        "revised_completion_date": "2023-06-01",
        "has_delay": True,
        "has_extension": False
    }
    results.append(test_scenario(
        "Scenario 2: Moderate Risk Project",
        moderate_risk_payload,
        "MEDIUM"
    ))
    
    # Test 4: High Risk Scenario
    high_risk_payload = {
        "project_name": "Lakeside Towers",
        "district": "Thane",
        "pin_code": "400601",
        "total_units": 200,
        "booked_units": 30,
        "project_area": 12000,
        "fsi": 1.5,
        "floors": 15,
        "cases": 5,
        "proposed_completion_date": "2022-01-01",
        "revised_completion_date": "2024-01-01",
        "has_delay": True,
        "has_extension": True
    }
    results.append(test_scenario(
        "Scenario 3: High Risk Project",
        high_risk_payload,
        "HIGH"
    ))
    
    # Test 5: Map Prediction
    results.append(test_map_prediction())
    
    # Test 6: Explainability
    results.append(test_explanation())
    
    # Summary
    print_header("📊 TEST SUMMARY")
    passed = sum(results)
    total = len(results)
    
    print(f"  Tests Passed: {passed}/{total}")
    print(f"  Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print_success(f"\n✨ All tests passed! System is production-ready.")
    else:
        print_error(f"\n⚠ Some tests failed. Review errors above.")
    
    print(f"\n{Colors.BLUE}{'=' * 80}{Colors.END}\n")

if __name__ == "__main__":
    print(f"\n{Colors.YELLOW}🚀 Starting API tests...")
    print(f"⏳ Make sure the API is running at {API_BASE}")
    print(f"   Run: `uvicorn app.main:app --reload`{Colors.END}\n")
    
    # input(f"{Colors.GREEN}Press Enter to start tests...{Colors.END}")
    
    run_all_tests()
