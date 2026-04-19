# 🚀 Deployment Guide: Streamlit Cloud

This guide walks you through deploying the Time Series Forecasting App to Streamlit Cloud for free, shareable access.

## 📋 Prerequisites

You'll need:
- GitHub account (free at [github.com](https://github.com))
- Streamlit Cloud account (free at [streamlit.io](https://streamlit.io/cloud))
- Git installed on your computer
- Your project pushed to GitHub

## Step 1: Prepare GitHub Repository

### If you already have the code locally:

```bash
# Navigate to your project directory
cd /path/to/nixtla

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Time series forecasting app"

# Create repository on GitHub
# Go to https://github.com/new and create a new repository

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Repository Structure

Your GitHub repo should look like:
```
your-repo/
├── streamlit_app.py
├── requirements.txt
├── sample_data.py
├── README.md
├── DEPLOYMENT.md
├── ARCHITECTURE.md
└── full_pipeline/
    ├── backtesting.py
    ├── reporting.py
    ├── config_manager.py
    ├── validators.py
    ├── df_statsforecast.py
    ├── df_mlforecast.py
    └── df_neuralforecast.py
```

## Step 2: Verify requirements.txt

Ensure your `requirements.txt` includes all dependencies:

```
# Streamlit App
streamlit>=1.28.0

# Data manipulation and visualization
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0

# Nixtla forecasting libraries (Core)
statsforecast>=1.5.0
mlforecast>=0.10.0
neuralforecast>=1.6.0
utilsforecast>=0.0.10

# Machine Learning libraries (for MLForecast)
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0

# Deep Learning libraries (for NeuralForecast)
torch>=2.0.0
pytorch-lightning>=2.0.0
pydantic>=2.0.0

# Ray for NeuralForecast distributed training (optional)
ray>=2.5.0

# Additional utilities
python-dateutil>=2.8.0
```

## Step 3: Create Streamlit Cloud Account

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click "Sign in with GitHub"
3. Authorize Streamlit to access your GitHub account
4. You're ready to deploy!

## Step 4: Deploy Your App

### Method 1: Deploy from GitHub (Recommended)

1. **On Streamlit Cloud**: Click "New app"
2. **Connect repository**:
   - Select your GitHub repo
   - Branch: `main`
   - File path: `streamlit_app.py`
3. **Click "Deploy"**

Streamlit will:
- Clone your repository
- Install dependencies from `requirements.txt`
- Launch your app
- Provide a public URL

### Method 2: Deploy from Local Computer

If you prefer manual deployment:

```bash
# Login to Streamlit Cloud
streamlit login

# Deploy
streamlit deploy \
  --title "Time Series Forecasting App" \
  streamlit_app.py
```

## Step 5: Access Your App

Once deployed, you'll get a URL like:
```
https://YOUR_USERNAME-YOUR_REPO_NAME-APPNAME.streamlit.app
```

Share this URL with anyone to let them use your app!

## 🔧 Advanced Configuration

### Environment Variables (Secrets)

If you need to store sensitive information:

1. Click app menu (⋮) → Settings
2. Go to "Secrets"
3. Add key-value pairs in TOML format:

```toml
# .streamlit/secrets.toml
api_key = "your_secret_key"
database_url = "postgresql://..."
```

Access in your app:
```python
import streamlit as st
secret = st.secrets["api_key"]
```

### Custom Configuration

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[client]
showErrorDetails = false
maxMessageSize = 200

[server]
maxUploadSize = 200
```

### Resource Limits

Free tier includes:
- **3 concurrent apps**
- **1 GB memory per app**
- **Shared CPU resources**
- **Public URL**

For production use, consider:
- Streamlit Cloud Pro for dedicated resources
- Private deployments on AWS/GCP/Azure
- On-premise deployments

## 📊 Monitoring & Logs

### View Logs

1. Click app menu (⋮) → Manage app
2. Scroll down to see recent logs
3. Expand "Logs" to see detailed output

### Debugging Common Issues

**App takes too long to load**
- Nixtla models have large dependencies
- First load may take 2-3 minutes
- Subsequent loads are cached

**Memory error**
- Reduce dataset size
- Use smaller models (Naive, SeasonalNaive)
- Limit number of backtest windows

**Model installation fails**
- Check `requirements.txt` versions
- Ensure all dependencies are listed
- Test locally first: `pip install -r requirements.txt`

## 🔄 Updates & Redeployment

### Push Updates to GitHub

```bash
# Make code changes
git add .
git commit -m "Update feature X"
git push origin main
```

Streamlit Cloud automatically redeploys within seconds!

### Clear Cache

If you want to force a full rebuild:

1. Click app menu (⋮) → Settings
2. Click "Reboot app"

Or in code:
```python
import streamlit as st
st.cache_data.clear()
st.cache_resource.clear()
```

## 📱 Sharing Your App

### Public URL
The app is public by default. Share the URL:
```
https://YOUR_USERNAME-YOUR_REPO_NAME-APPNAME.streamlit.app
```

### QR Code
Generate a QR code for easy mobile access:
1. Click app menu (⋮)
2. Copy app link to QR code generator
3. Share QR code

### Embed in Website
```html
<iframe
  src="https://YOUR_USERNAME-YOUR_REPO_NAME-APPNAME.streamlit.app"
  height="500"
  width="100%">
</iframe>
```

## ⚠️ Important Notes

### Performance Considerations

Backtesting with 7 models × 5 windows takes 2-3 minutes:
- First run will be slower (model loading)
- Subsequent runs use cached models
- Single forecasts are much faster (30 seconds)

### Data Privacy

- Free Streamlit Cloud apps are public
- Don't upload sensitive data
- For private apps, use Streamlit Cloud Pro

### Cost

- **Free**: Unlimited deployments, 3 concurrent apps
- **Pro**: $13/month per deployment, private URLs, more resources
- **Business**: Custom pricing for enterprise

## 🎯 Production Checklist

Before sharing your deployed app:

- [ ] README.md is clear and complete
- [ ] Sample datasets work correctly
- [ ] Validation messages are helpful
- [ ] Error messages are user-friendly
- [ ] All dependencies are listed in requirements.txt
- [ ] No hardcoded secrets in code
- [ ] App description is set in Streamlit settings
- [ ] Preview image is helpful (shows app UI)

## 📞 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Add missing package to `requirements.txt`

### Issue: "App takes >1 minute to load"
**Solution**: 
- First load installs packages (normal)
- Subsequent loads use cache
- Can't optimize this on free tier

### Issue: "Memory exceeded"
**Solution**:
- Use smaller dataset
- Reduce backtest windows
- Skip neural models (use StatsForecast/MLForecast)

### Issue: "Changes not showing"
**Solution**:
- Force refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
- Clear browser cache
- Wait 30 seconds for deployment

## 🔗 Useful Links

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Pricing](https://streamlit.io/pricing)
- [GitHub Desktop](https://desktop.github.com/) (if you prefer GUI over CLI)
- [Nixtla Documentation](https://nixtla.github.io/statsforecast/)

---

**Your app is now live! 🎉**

Share the URL with colleagues, supervisors, or add it to your portfolio. The app is production-ready and can handle real forecasting tasks.

Need help? Check Streamlit Cloud documentation or contact support at support@streamlit.io
