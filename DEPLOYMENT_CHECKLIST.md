# 🚀 DEPLOYMENT CHECKLIST - RERA Risk Prediction System

## ✅ Pre-Deployment Validation

### 1. Model Training Complete
- [ ] Run `python train_robust.py`
- [ ] Check `model_training_results.txt` for metrics
- [ ] Verify `app/models/` contains:
  - [ ] `models.pkl`
  - [ ] `metadata.pkl`
  - [ ] `shap_explainer.pkl` (optional, large file)
- [ ] Review latest log in `training_logs/`

### 2. Local Testing
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Start API: `uvicorn app.main:app --reload`
- [ ] Run test suite: `python test_api.py`
- [ ] Verify all 6 tests pass
- [ ] Check `/docs` endpoint loads Swagger UI
- [ ] Test `/health` endpoint returns `model_loaded: true`

### 3. Code Quality
- [ ] No hardcoded credentials
- [ ] Logging configured properly
- [ ] Error handling in place
- [ ] CORS origins restricted (for production)
- [ ] Pydantic validation working

---

## 🐳 Dockerization

### 1. Build Docker Image
```bash
docker build -t rera-risk-api .
```
- [ ] Build completes successfully
- [ ] Image size reasonable (<500MB)

### 2. Test Docker Locally
```bash
docker run -p 8000:8000 rera-risk-api
```
- [ ] Container starts without errors
- [ ] API accessible at `localhost:8000`
- [ ] Health check passes

### 3. Optional: Push to Docker Hub
```bash
docker tag rera-risk-api <your-username>/rera-risk-api:v1.0.0
docker push <your-username>/rera-risk-api:v1.0.0
```

---

## ☁️ Railway Deployment

### Step 1: Prepare Repository
```bash
git init
git add .
git commit -m "Initial deployment - RERA risk prediction API"
```
- [ ] All necessary files committed
- [ ] `.gitignore` configured
- [ ] Large CSVs excluded if needed
- [ ] Model files included (or use separate storage)

### Step 2: Push to GitHub
```bash
git remote add origin <your-repo-url>
git push -u origin main
```
- [ ] Repository created on GitHub
- [ ] Code pushed successfully
- [ ] README displayed correctly

### Step 3: Connect to Railway
1. [ ] Go to [railway.app](https://railway.app)
2. [ ] Sign in with GitHub
3. [ ] Click "New Project"
4. [ ] Select "Deploy from GitHub Repo"
5. [ ] Choose your RERA repository
6. [ ] Railway auto-detects `Dockerfile`

### Step 4: Configure (if needed)
- [ ] Environment variables set (if any)
- [ ] Build command: (auto-detected)
- [ ] Start command: (auto-detected from Dockerfile)
- [ ] Health check path: `/health`

### Step 5: Deploy & Verify
- [ ] Deployment completes
- [ ] Logs show "Application startup complete"
- [ ] Visit `/health` endpoint
- [ ] Test `/docs` for interactive API
- [ ] Run test_api.py against live URL

### Step 6: Get Public URL
- [ ] Copy Railway-provided URL
- [ ] Test: `curl https://your-app.railway.app/health`
- [ ] Update frontend/docs with production URL

---

## 🔄 Alternative: Render Deployment

### Step 1-2: Same as Railway (GitHub push)

### Step 3: Render Setup
1. [ ] Go to [render.com](https://render.com)
2. [ ] Create "New Web Service"
3. [ ] Connect GitHub repository
4. [ ] Environment: Docker
5. [ ] Instance Type: Free (or starter)

### Step 4: Configure
- [ ] Docker command: (auto-detected)
- [ ] Health check: `/health`
- [ ] Auto-deploy: ON

### Step 5: Deploy
- [ ] Build starts automatically
- [ ] Check build logs
- [ ] Service goes live
- [ ] Test public URL

---

## 📊 Post-Deployment Monitoring

### Immediate Checks
- [ ] `/health` returns 200 OK
- [ ] `/metrics` shows Prometheus data
- [ ] Test prediction with sample data
- [ ] Check response times (<500ms typical)

### Logging
- [ ] View application logs in Railway/Render dashboard
- [ ] Verify structured logging format
- [ ] Check for any startup errors

### Performance
- [ ] Test cold start time (first request after idle)
- [ ] Measure average response time
- [ ] Monitor memory usage
- [ ] Check CPU utilization

---

## 🔐 Security Hardening (Production)

### API Security
- [ ] Update CORS to specific origins:
  ```python
  allow_origins=["https://your-frontend-domain.com"]
  ```
- [ ] Add rate limiting (if needed)
- [ ] Implement API key authentication (optional)
- [ ] Use HTTPS only

### Data Privacy
- [ ] No PII logged
- [ ] Sanitize inputs
- [ ] Validate all requests with Pydantic
- [ ] Error messages don't leak system info

---

## 📈 Ongoing Maintenance

### Weekly
- [ ] Check error logs
- [ ] Monitor request volume
- [ ] Review performance metrics

### Monthly
- [ ] Retrain models with new data
- [ ] Update dependencies
- [ ] Review SHAP explanations for drift
- [ ] Optimize feature engineering

### Quarterly
- [ ] Evaluate model performance
- [ ] A/B test new model versions
- [ ] Review security posture
- [ ] Optimize costs

---

## 🆘 Troubleshooting Guide

### Issue: Models not loading
**Solution**: 
- Check `app/models/*.pkl` files exist
- Verify pickle protocol compatibility
- Retrain models if needed

### Issue: High latency
**Solution**:
- Profile with `/metrics` endpoint
- Consider removing SHAP from default response
- Cache district-level stats
- Use Gunicorn with multiple workers

### Issue: Memory errors
**Solution**:
- Reduce `shap_explainer.pkl` size (sample fewer examples)
- Load SHAP on-demand only
- Increase instance RAM

### Issue: Geocoding fails
**Solution**:
- Check geopy/Nominatim service status
- Implement fallback to default district
- Add retry logic

---

## ✨ Enhancement Roadmap

### Short-term (1-2 weeks)
- [ ] Add batch prediction endpoint
- [ ] Implement caching (Redis)
- [ ] Create Grafana dashboard
- [ ] Add admin panel

### Medium-term (1-2 months)
- [ ] PostgreSQL integration
- [ ] Real-time model retraining pipeline
- [ ] A/B testing framework
- [ ] Mobile-optimized API responses

### Long-term (3-6 months)
- [ ] Multi-region deployment
- [ ] GraphQL API
- [ ] Real-time WebSocket updates
- [ ] ML model marketplace integration

---

## 📞 Support

**Documentation**: See `README.md`  
**Training Logs**: Check `training_logs/`  
**Metrics**: See `model_training_results.txt`  
**API Docs**: Visit `/docs` endpoint  

---

**Deployment Status**: 
- [ ] ✅ Ready for Railway/Render
- [ ] 🧪 Tested locally
- [ ] 🐳 Docker verified
- [ ] 📊 Metrics validated
- [ ] 🚀 PRODUCTION DEPLOYED

---

**Last Updated**: 2026-01-19  
**System Version**: 1.0.0-prod
