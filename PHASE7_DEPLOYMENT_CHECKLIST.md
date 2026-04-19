# Phase 7: Streamlit Cloud Deployment Checklist

## 🚀 Pre-Deployment Steps

### Step 1: GitHub Repository Setup (REQUIRED FIRST)
> ⚠️ Current Issue: Push permission denied to PJalgotrader/Deep_forecasting-USU  
> **Action Required**: Use Option A (Fork) or Option B (New Repo) below

#### Option A: Fork Existing Repository (Recommended)
```bash
# 1. Go to https://github.com/PJalgotrader/Deep_forecasting-USU
# 2. Click "Fork" button (top-right)
# 3. In your local repo, update remote:
git remote set-url origin https://github.com/YOUR_GITHUB_USERNAME/Deep_forecasting-USU.git
git push -u origin main

# 4. Verify push succeeded
git remote -v
```

**Expected Output:**
```
origin  https://github.com/YOUR_GITHUB_USERNAME/Deep_forecasting-USU.git (fetch)
origin  https://github.com/YOUR_GITHUB_USERNAME/Deep_forecasting-USU.git (push)
```

#### Option B: Create New Personal Repository
```bash
# 1. Create new repo: https://github.com/new
#    Name: nixtla-forecasting (or your preferred name)
#    Description: Time series forecasting with Nixtla ecosystem
# 
# 2. Update local remote:
git remote set-url origin https://github.com/YOUR_GITHUB_USERNAME/nixtla-forecasting.git
git push -u origin main

# 3. Verify
git remote -v
```

---

### Step 2: Verify All Files Pushed to GitHub
```bash
# Check local status
git status

# Verify pushed commits
git log --oneline -5
```

**Files that should be on GitHub:**
- ✅ streamlit_app.py (modified)
- ✅ requirements.txt
- ✅ full_pipeline/backtesting.py
- ✅ full_pipeline/config_manager.py
- ✅ full_pipeline/reporting.py
- ✅ full_pipeline/validators.py
- ✅ full_pipeline/df_statsforecast.py (existing)
- ✅ full_pipeline/df_mlforecast.py (existing)
- ✅ full_pipeline/df_neuralforecast.py (existing)
- ✅ README.md
- ✅ DEPLOYMENT.md
- ✅ ARCHITECTURE.md

---

## 📋 Streamlit Cloud Deployment

### Step 3: Create Streamlit Cloud Account
1. Go to https://share.streamlit.io
2. Click "Sign up"
3. Choose "Continue with GitHub"
4. Authorize Streamlit to access your GitHub account
5. ✅ Account created

### Step 4: Deploy Application
1. On Streamlit Cloud dashboard, click **"New app"**
2. Fill in deployment form:
   - **Repository**: `YOUR_USERNAME/Deep_forecasting-USU` (or your repo name)
   - **Branch**: `main`
   - **Main file path**: `Lectures and codes/miscellaneous/nixtla/streamlit_app.py`
   
   > ⚠️ **IMPORTANT**: The path includes spaces - make sure it's exactly:  
   > `Lectures and codes/miscellaneous/nixtla/streamlit_app.py`

3. Click **"Deploy"**
4. Wait 2-3 minutes for deployment to complete

### Step 5: Configure App Settings (Optional)
Once deployed, click "Settings" gear icon in top-right:

**Advanced Settings:**
- **Client max message size**: 200 MB (allows larger CSV uploads)
- **Logger level**: Info

**Secrets** (if needed later):
- API keys go here (currently none required)

---

## ✅ Post-Deployment Verification

### Step 6: Test Live Application
1. Once green checkmark appears, app URL will be displayed:
   ```
   https://share.streamlit.io/YOUR_USERNAME/REPO_NAME/main/Lectures%20and%20codes/miscellaneous/nixtla/streamlit_app.py
   ```

2. Test features:
   - ✅ Load sample data (AirPassengers)
   - ✅ Single forecast (one model)
   - ✅ Backtesting comparison (5 windows)
   - ✅ Config save/load
   - ✅ CSV upload functionality

3. Share URL with:
   - Class/instructor
   - Portfolio
   - GitHub README

---

## 🔧 Troubleshooting

### Issue: "Module not found" errors
**Solution**: Check `requirements.txt` includes all packages
```bash
# View installed packages locally
pip list | grep -E "streamlit|statsforecast|mlforecast|neuralforecast"
```

### Issue: "No such file or directory" in logs
**Solution**: Verify main file path includes correct directory structure
- ❌ Wrong: `streamlit_app.py`
- ✅ Correct: `Lectures and codes/miscellaneous/nixtla/streamlit_app.py`

### Issue: App times out during backtesting
**Solution**: Streamlit Cloud has 1GB memory limit. For large datasets:
1. Reduce number of windows: change line in code
2. Reduce model count: modify auto model selection
3. Use smaller sample CSV

### Issue: "Permission denied" on GitHub
**Solution**: 
- Ensure you forked/created YOUR OWN repository
- Push command uses YOUR GitHub username, not PJalgotrader
- Verify SSH key is added to GitHub (or use HTTPS with token)

---

## 📊 Performance Notes

**Backtesting Performance** (on Streamlit Cloud):
- 5-window backtest with 7 models: ~60-90 seconds
- Single forecast: ~5-10 seconds
- Memory usage: ~400-600 MB during backtest

**Optimization Tips:**
- Backtesting reduces window count if < 50 observations
- Disable unused models if speed is critical
- Cache results between runs (built into app)

---

## 🎓 Portfolio Presentation

### Markdown for Your Portfolio/Resume
```markdown
## Deep Forecasting Application

**Live Demo**: [Nixtla Forecasting App](https://share.streamlit.io/YOUR_USERNAME/YOUR_REPO/...)

**Technologies**: Python, Streamlit, Nixtla (StatsForecast, MLForecast, NeuralForecast)

**Features**:
- Multi-model time series forecasting with 17+ statistical, ML, and deep learning models
- 5-window rolling cross-validation backtesting for realistic performance evaluation
- Interactive visualizations with Plotly
- JSON-based model configuration persistence
- Comprehensive input validation and error handling
- Production-ready architecture with modular components

**Architecture**:
- `backtesting.py`: Rolling window cross-validation (5 windows, all models)
- `reporting.py`: Visualization and model comparison
- `config_manager.py`: Configuration persistence
- `validators.py`: Input validation and data quality checks
- `streamlit_app.py`: Interactive UI and orchestration

**Live Deployment**: Streamlit Cloud
```

---

## ✨ Final Checklist

- [ ] Step 1: Fork repo or create new repo with own GitHub account
- [ ] Step 2: Verify all files pushed to GitHub
- [ ] Step 3: Create Streamlit Cloud account
- [ ] Step 4: Deploy app from GitHub
- [ ] Step 5: Configure advanced settings (optional)
- [ ] Step 6: Test all features on live app
- [ ] Step 7: Share URL with class/portfolio

**Status**: Ready for deployment once GitHub is configured ✅

---

**Need Help?**  
See DEPLOYMENT.md for detailed step-by-step instructions.
