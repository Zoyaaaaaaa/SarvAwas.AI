# RERA Risk Prediction System - Production Ready 🚀

## 📊 System Overview

This is a **production-grade, deployable ML system** that predicts RERA (Real Estate Regulatory Authority) project completion risk using explainable AI. The system is containerized, fully logged, and ready for cloud deployment on **Railway** or **Render**.

---

## 🏗️ Architecture

```
models_rera/
├── Dockerfile                      # Container config for cloud deployment
├── requirements.txt                # Python dependencies
├── train_robust.py                 # Robust training pipeline with logging
├── model_training_results.txt      # Latest training metrics
├── training_logs/                  # Timestamped training logs
├── data/                           # RERA datasets (11 CSVs)
└── app/
    ├── main.py                     # FastAPI application entry
    ├── models/
    │   ├── loader.py               # Model loading service
    │   ├── models.pkl              # Trained LightGBM models
    │   ├── metadata.pkl            # Feature metadata & encoders
    │   └── shap_explainer.pkl      # SHAP explainability engine
    ├── routes/
    │   ├── predict.py              # /api/v1/predict/* endpoints
    │   └── map.py                  # /api/v1/predict/map endpoint
    ├── services/
    │   ├── feature_engineering.py  # Input preprocessing logic
    │   └── geo_utils.py            # Geocoding & spatial intelligence
    └── schemas/
        └── request_response.py     # Pydantic validation schemas
```

---

## 🎯 Model Capabilities

### Multi-Task Prediction
1. **Completion Probability** (Regression): 0.0 to 1.0 likelihood of project success
2. **Delay Risk Classification**: Minimal | Short | Moderate | Severe
3. **Regulatory Risk Score**: 0-10 composite safety score

### Latest Performance Metrics
- **Completion Probability Model**: MAE: ~0.017, RMSE: ~0.13, R²: ~0.91
- **Delay Risk Classifier**: Accuracy: ~91%, Weighted F1: ~0.91
- **Dataset**: 44,000+ RERA projects across Maharashtra

### Top Predictive Features
1. `booking_ratio` - Market demand signal
2. `has_extension` - Timeline modification flag
3. `has_delay` - Historical delay indicator
4. `inventory_left` - Unsold units count
5. `legal_risk_score` - Active litigation count

---

## 🚀 Quick Start

### 1. Local Development
```bash
cd models_rera
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Visit: `http://localhost:8000/docs` for interactive API documentation.

### 2. Retrain Models (Optional)
```bash
python train_robust.py
```
This will:
- Load all RERA datasets from `data/`
- Clean project/promoter names
- Engineer 19+ features
- Train multi-task LightGBM models
- Save artifacts to `app/models/`
- Generate detailed logs in `training_logs/`
- Create `model_training_results.txt` with metrics

---

## 📡 API Endpoints

### Core Prediction
**POST** `/api/v1/predict/predict`
```json
{
  "district": "Mumbai Suburban",
  "pin_code": "400001",
  "total_units": 100,
  "booked_units": 85,
  "fsi": 2.5,
  "floors": 12,
  "cases": 0,
  "has_delay": false,
  "has_extension": false
}
```

**Response:**
```json
{
  "risk_level": "LOW",
  "completion_probability": 0.93,
  "delay_risk": "MINIMAL",
  "confidence_score": 93.0,
  "key_factors": [
    "High market demand (Strong booking ratio)",
    "Clean legal record (No active RERA cases)",
    "Statistically strong completion profile"
  ],
  "model_version": "1.0.0-prod"
}
```

### Map-Based Risk
**POST** `/api/v1/predict/map`
```json
{
  "lat": 19.0760,
  "lng": 72.8777
}
```

### Explainability
**POST** `/api/v1/predict/explain`
Same request as `/predict`, returns SHAP feature importance.

### Health Check
**GET** `/health`

---

## 🐳 Docker Deployment

### Build Locally
```bash
docker build -t rera-risk-api .
docker run -p 8000:8000 rera-risk-api
```

### Deploy to Railway (Recommended)

#### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial RERA Risk API"
git remote add origin <your-repo-url>
git push -u origin main
```

#### Step 2: Deploy on Railway
1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub Repo"
3. Select your repository
4. Railway auto-detects `Dockerfile` and deploys
5. Copy the public URL

#### Step 3: Test
```bash
curl https://your-app.railway.app/health
```

---

## 📊 Observability & Monitoring

### Built-in Features
- **Prometheus Metrics**: `/metrics` endpoint for latency, request count, etc.
- **Structured Logging**: All predictions logged with timestamps and latency
- **Health Checks**: `/health` endpoint reports model status

### Training Logs
Every training run creates a timestamped log in `training_logs/`:
```
training_20260119_151141.log
```

Contains:
- Data loading statistics
- Feature engineering steps
- Model training metrics
- Test scenario results

---

## 🧪 Test Scenarios

The training pipeline automatically tests 3 scenarios:

### 1. Low Risk Project
- Booking: 95%, No delays, No litigation
- **Predicted**: Completion Prob: ~95%, Delay Risk: Minimal

### 2. Moderate Risk Project
- Booking: 55%, Some delays, 1 case
- **Predicted**: Completion Prob: ~60%, Delay Risk: Moderate

### 3. High Risk Project
- Booking: 15%, Multiple delays, 5 cases
- **Predicted**: Completion Prob: ~25%, Delay Risk: Severe

---

## 🔧 Configuration

### Environment Variables
Create `.env` file (optional):
```env
LOG_LEVEL=INFO
MODEL_VERSION=1.0.0
MAX_WORKERS=4
```

### Feature Engineering
Edit `app/services/feature_engineering.py` to customize:
- Risk thresholds
- Verdict generation logic
- Feature transformations

---

## 📁 Data Requirements

Place RERA CSVs in `data/` directory:
- `mumbai-suburban-rera-dataset.csv`
- `pune-rera-dataset.csv`
- `thane-rera-dataset.csv`
- ... (11 total files)

Required columns:
- `district`, `project_name`, `promoter_name`
- `proposed_date_of_completion`, `revised_proposed_date_of_completion`
- `number_of_appartments`, `number_of_booked_appartments`
- `sanctioned_fsi`, `cases_count`

---

## 🛡️ Production Checklist

- [x] Dockerized
- [x] Pydantic validation
- [x] Structured logging
- [x] CORS configured
- [x] Health checks
- [x] Prometheus metrics
- [x] SHAP explainability
- [x] Error handling
- [x] Feature versioning
- [x] Model serialization

---

## 🧠 Machine Learning Details

### Preprocessing Pipeline
1. **Text Cleaning**: Standardize project/promoter names
2. **Date Engineering**: Calculate delays, extensions
3. **Market Signals**: Booking ratios, inventory
4. **Location Intelligence**: District-level aggregations
5. **Legal Risk**: Case count normalization

### Model Stack
- **Algorithm**: LightGBM (Gradient Boosting)
- **Training**: 80/20 split, stratified sampling
- **Hyperparameters**: 200 estimators, depth=7, balanced class weights
- **Explainability**: SHAP TreeExplainer

### Evaluation
- **Regression**: MAE, RMSE, R²
- **Classification**: Accuracy, Precision, Recall, F1
- **Cross-Validation**: 5-fold stratified

---

## 🚨 Troubleshooting

### Models Not Loading
- Check `app/models/` contains: `models.pkl`, `metadata.pkl`
- Retrain: `python train_robust.py`

### Training Fails
- Verify data in `data/` directory
- Check logs in `training_logs/`
- Ensure required columns exist in CSVs

### Docker Build Issues
- Update `requirements.txt`
- Use `--no-cache` flag: `docker build --no-cache -t rera-risk-api .`

---

## 📞 Support & Next Steps

### Enhancement Ideas
1. **PostgreSQL Integration**: Store predictions for analytics
2. **Real-time Monitoring**: Grafana dashboard for metrics
3. **A/B Testing**: Multiple model versions
4. **Batch Prediction**: CSV upload endpoint
5. **Frontend**: React dashboard with map visualization

### Model Improvements
- Add promoter reputation score
- Incorporate construction timeline data
- Sentiment analysis on complaints
- Geospatial clustering (DBSCAN)

---

## 📜 License & Credits

Built for production deployment. Uses:
- LightGBM, SHAP, FastAPI, Pydantic
- Prometheus, Geopy, Pandas, Scikit-learn

**Deployment Ready** ✅ Railway | Render | Any Docker-compatible platform

---

**Last Model Training**: Check `model_training_results.txt` for latest metrics.  
**Training Logs**: See `training_logs/` for detailed run history.
#   S a r v A w a s . A I  
 