"""
RERA Risk Prediction System - Pre-Deployment Verification Script
Checks all components are in place before deployment
"""

import sys
from pathlib import Path
from colorama import init, Fore, Style
import pickle

init(autoreset=True)

class Status:
    PASS = f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}"
    FAIL = f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"
    WARN = f"{Fore.YELLOW}⚠ WARN{Style.RESET_ALL}"
    INFO = f"{Fore.BLUE}→ INFO{Style.RESET_ALL}"

def print_header(text):
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{text}")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")

def check_file(path: Path, description: str) -> bool:
    if path.exists():
        size = path.stat().st_size
        print(f"{Status.PASS} {description}: {path.name} ({size:,} bytes)")
        return True
    else:
        print(f"{Status.FAIL} {description}: {path.name} NOT FOUND")
        return False

def check_directory(path: Path, description: str) -> bool:
    if path.exists() and path.is_dir():
        count = len(list(path.iterdir()))
        print(f"{Status.PASS} {description}: {path.name} ({count} items)")
        return True
    else:
        print(f"{Status.FAIL} {description}: {path.name} NOT FOUND")
        return False

def main():
    print_header("🔍 RERA RISK PREDICTION SYSTEM - PRE-DEPLOYMENT VERIFICATION")
    
    results = []
    base_dir = Path.cwd()
    
    # Check 1: Core Files
    print_header("1️⃣ CORE CONFIGURATION FILES")
    results.append(check_file(base_dir / "Dockerfile", "Dockerfile"))
    results.append(check_file(base_dir / "requirements.txt", "Requirements"))
    results.append(check_file(base_dir / ".gitignore", "Git ignore"))
    results.append(check_file(base_dir / "README.md", "Documentation"))
    results.append(check_file(base_dir / "train_robust.py", "Training script"))
    results.append(check_file(base_dir / "test_api.py", "Test suite"))
    
    # Check 2: Documentation
    print_header("2️⃣ DOCUMENTATION FILES")
    results.append(check_file(base_dir / "DEPLOYMENT_CHECKLIST.md", "Deployment guide"))
    results.append(check_file(base_dir / "DEPLOYMENT_SUMMARY.txt", "Deployment summary"))
    results.append(check_file(base_dir / "IMPROVEMENTS_LOG.txt", "Improvements log"))
    
    # Check 3: Model Artifacts
    print_header("3️⃣ TRAINED MODEL ARTIFACTS")
    models_dir = base_dir / "app" / "models"
    results.append(check_file(models_dir / "models.pkl", "Trained models"))
    results.append(check_file(models_dir / "metadata.pkl", "Model metadata"))
    results.append(check_file(models_dir / "shap_explainer.pkl", "SHAP explainer"))
    results.append(check_file(models_dir / "loader.py", "Model loader"))
    
    # Check 4: Application Structure
    print_header("4️⃣ APPLICATION STRUCTURE")
    results.append(check_directory(base_dir / "app", "App directory"))
    results.append(check_directory(base_dir / "app" / "routes", "Routes directory"))
    results.append(check_directory(base_dir / "app" / "services", "Services directory"))
    results.append(check_directory(base_dir / "app" / "schemas", "Schemas directory"))
    
    results.append(check_file(base_dir / "app" / "main.py", "FastAPI app"))
    results.append(check_file(base_dir / "app" / "routes" / "predict.py", "Prediction routes"))
    results.append(check_file(base_dir / "app" / "routes" / "map.py", "Map routes"))
    results.append(check_file(base_dir / "app" / "services" / "feature_engineering.py", "Feature engineering"))
    results.append(check_file(base_dir / "app" / "services" / "geo_utils.py", "Geo utilities"))
    results.append(check_file(base_dir / "app" / "schemas" / "request_response.py", "Pydantic schemas"))
    
    # Check 5: Data Files
    print_header("5️⃣ DATA FILES")
    data_dir = base_dir / "data"
    if check_directory(data_dir, "Data directory"):
        csv_files = list(data_dir.glob("*.csv"))
        print(f"{Status.INFO} Found {len(csv_files)} CSV files")
        if len(csv_files) >= 10:
            print(f"{Status.PASS} Sufficient RERA datasets")
            results.append(True)
        else:
            print(f"{Status.WARN} Only {len(csv_files)} datasets found (expected 11)")
            results.append(False)
    else:
        results.append(False)
    
    # Check 6: Training Results
    print_header("6️⃣ TRAINING RESULTS")
    results.append(check_file(base_dir / "model_training_results.txt", "Training metrics"))
    
    logs_dir = base_dir / "training_logs"
    if check_directory(logs_dir, "Training logs directory"):
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            print(f"{Status.INFO} Latest log: {latest_log.name}")
            results.append(True)
        else:
            print(f"{Status.WARN} No training logs found")
            results.append(False)
    else:
        results.append(False)
    
    # Check 7: Model Validation
    print_header("7️⃣ MODEL VALIDATION")
    try:
        with open(models_dir / "models.pkl", "rb") as f:
            models = pickle.load(f)
        
        if 'completion_probability' in models:
            print(f"{Status.PASS} Completion probability model loaded")
            results.append(True)
        else:
            print(f"{Status.FAIL} Completion probability model missing")
            results.append(False)
            
        if 'delay_risk_class' in models:
            print(f"{Status.PASS} Delay risk classifier loaded")
            results.append(True)
        else:
            print(f"{Status.FAIL} Delay risk classifier missing")
            results.append(False)
            
    except Exception as e:
        print(f"{Status.FAIL} Error loading models: {e}")
        results.append(False)
        results.append(False)
    
    try:
        with open(models_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        feature_count = len(metadata.get('feature_cols', []))
        print(f"{Status.INFO} Features in model: {feature_count}")
        
        if feature_count >= 10:
            print(f"{Status.PASS} Sufficient features ({feature_count})")
            results.append(True)
        else:
            print(f"{Status.WARN} Low feature count ({feature_count})")
            results.append(False)
            
    except Exception as e:
        print(f"{Status.FAIL} Error loading metadata: {e}")
        results.append(False)
    
    # Check 8: Dependencies
    print_header("8️⃣ DEPENDENCY CHECK")
    try:
        import fastapi
        print(f"{Status.PASS} FastAPI: {fastapi.__version__}")
        results.append(True)
    except ImportError:
        print(f"{Status.FAIL} FastAPI not installed")
        results.append(False)
    
    try:
        import lightgbm
        print(f"{Status.PASS} LightGBM: {lightgbm.__version__}")
        results.append(True)
    except ImportError:
        print(f"{Status.FAIL} LightGBM not installed")
        results.append(False)
    
    try:
        import shap
        print(f"{Status.PASS} SHAP: {shap.__version__}")
        results.append(True)
    except ImportError:
        print(f"{Status.FAIL} SHAP not installed")
        results.append(False)
    
    try:
        import geopy
        print(f"{Status.PASS} Geopy: {geopy.__version__}")
        results.append(True)
    except ImportError:
        print(f"{Status.FAIL} Geopy not installed")
        results.append(False)
    
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        print(f"{Status.PASS} Prometheus instrumentator")
        results.append(True)
    except ImportError:
        print(f"{Status.FAIL} Prometheus instrumentator not installed")
        results.append(False)
    
    # Final Summary
    print_header("📊 VERIFICATION SUMMARY")
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"  Checks Passed: {Fore.GREEN}{passed}{Style.RESET_ALL}/{total}")
    print(f"  Success Rate: {Fore.GREEN if success_rate == 100 else Fore.YELLOW}{success_rate:.1f}%{Style.RESET_ALL}")
    
    if success_rate == 100:
        print(f"\n{Fore.GREEN}✨ SYSTEM IS DEPLOYMENT READY!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}   → All components verified")
        print(f"{Fore.GREEN}   → Models trained and validated")
        print(f"{Fore.GREEN}   → Documentation complete")
        print(f"{Fore.GREEN}   → Dependencies satisfied{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Next step: Push to GitHub and deploy to Railway{Style.RESET_ALL}")
        return 0
    elif success_rate >= 80:
        print(f"\n{Fore.YELLOW}⚠ SYSTEM IS MOSTLY READY{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   → Review failed checks above")
        print(f"{Fore.YELLOW}   → Fix critical issues before deployment{Style.RESET_ALL}")
        return 1
    else:
        print(f"\n{Fore.RED}✗ SYSTEM NOT READY FOR DEPLOYMENT{Style.RESET_ALL}")
        print(f"{Fore.RED}   → Too many failed checks")
        print(f"{Fore.RED}   → Review and fix issues above{Style.RESET_ALL}")
        return 2

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Verification script error: {e}{Style.RESET_ALL}")
        sys.exit(3)
