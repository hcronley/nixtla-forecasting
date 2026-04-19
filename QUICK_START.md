# Quick Reference: Running the App Locally

## One-Line Start
```bash
cd /Users/henrycronley/Classes/Current/DATA5630-Deep-Forecasting/Deep_forecasting-USU/"Lectures and codes"/miscellaneous/nixtla && streamlit run streamlit_app.py
```

## What to Test

### Basic Test (2 minutes)
1. **Load sample data** → AirPassengers should appear in preview
2. **Run single forecast** → Pick "StatsForecast - Auto ETS", click "Run Forecast"
3. **Check results** → Should see table and plot with metrics

### Full Feature Test (15 minutes)
1. **Single forecasts** → Try 3 different models (StatsForecast, MLForecast, NeuralForecast)
2. **Enable backtesting** → Check "Run Full Model Comparison" in sidebar, run forecast
3. **Review results** → Rankings table, comparison plots, heatmaps
4. **Save config** → After backtest, save best model config (JSON download)
5. **Load config** → Upload saved JSON, verify parameters load
6. **CSV upload** → Upload custom data (needs 'ds' and 'y' columns)

### Edge Cases (5 minutes)
1. **Bad CSV** → Upload file without 'ds' column → should show clear error
2. **Tiny dataset** → Upload 5-row CSV → should warn about data length
3. **Non-CSV file** → Try uploading .txt or .xlsx → should reject gracefully

---

## Expected Outputs

### ✅ Single Forecast Success
```
Model: AutoETS
Results:
- MAE: 12.34
- RMSE: 15.67
- MAPE: 2.89%
```

### ✅ Backtesting Success
```
5-window rolling CV results:
1. Model A: MAE=12.34 ± 1.23
2. Model B: MAE=13.45 ± 1.56
3. Model C: MAE=14.23 ± 2.10
[... 4 more models ...]
```

### ✅ Config Save/Load
```
Downloads: model_config_20260419_120000.json
Upload same file back: Parameters auto-fill
```

---

## If Something Goes Wrong

| Problem | Fix |
|---------|-----|
| Port already in use | `streamlit run streamlit_app.py --server.port 8502` |
| Module not found | `pip install -r requirements.txt` |
| Takes 5+ min to load | First run loads models, subsequent runs cached |
| Memory error in backtest | Reduce dataset size or disable NeuralForecast |
| CSV upload fails | Ensure columns named exactly 'ds' and 'y' |

---

## Success Criteria

The app is **ready for deployment** when:
- ✅ Sample data loads
- ✅ Single forecast works (all 3 model types)
- ✅ Backtesting completes without errors
- ✅ Config save/load works
- ✅ CSV upload accepts valid files
- ✅ CSV upload rejects bad files with clear error
- ✅ No crashes or unhandled exceptions
- ✅ Visualizations render correctly

---

## Next Steps After Testing

1. If tests pass → Push to GitHub
2. Create Streamlit Cloud account
3. Deploy to live URL
4. Share live app with instructors/portfolio

Estimated time from here to live: 15-20 minutes
