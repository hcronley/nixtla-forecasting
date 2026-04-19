# Local Testing Instructions

## Prerequisites
All dependencies are installed. To verify:
```bash
cd /Users/henrycronley/Classes/Current/DATA5630-Deep-Forecasting/Deep_forecasting-USU/"Lectures and codes"/miscellaneous/nixtla
python test_app.py
```

## Running the Streamlit App Locally

```bash
cd /Users/henrycronley/Classes/Current/DATA5630-Deep-Forecasting/Deep_forecasting-USU/"Lectures and codes"/miscellaneous/nixtla
streamlit run streamlit_app.py
```

This will open the app in your default browser at: `http://localhost:8501`

---

## Testing Checklist

### ✅ Test 1: UI and Navigation
- [ ] App loads without errors
- [ ] Sidebar displays correctly with all sections
- [ ] Page title and description visible
- [ ] Sample data loads when you click "Load Sample Data (AirPassengers)"

### ✅ Test 2: Data Quality Validation (with sample data)
- [ ] Data preview shows 144 observations
- [ ] Data quality warnings expander shows if any issues
- [ ] Validation checks pass (dates, values, length)
- [ ] Frequency detected as 'MS' (Monthly Start)

### ✅ Test 3: Single Forecast (StatsForecast)
**Steps:**
1. Make sure sample data is loaded
2. Keep default settings:
   - Model: StatsForecast - Auto ETS
   - Horizon: 12 months
   - Frequency: MS
3. Click "Run Forecast" button

**Expected Results:**
- Forecast executes without error (~5-10 seconds)
- Shows forecast table with dates and predictions
- Plots actual vs predicted with 95% confidence interval
- Displays metrics: MAE, RMSE, MAPE

### ✅ Test 4: Single Forecast (MLForecast)
**Steps:**
1. Change Model dropdown to "MLForecast - XGBoost"
2. Keep same settings
3. Click "Run Forecast"

**Expected Results:**
- Different forecast from StatsForecast
- Similar metrics displayed
- Visualization updates

### ✅ Test 5: Backtesting Comparison Mode
**Steps:**
1. Scroll down to Step 5 in sidebar
2. Check the checkbox: "Run Full Model Comparison (Backtesting)"
3. Click "Run Forecast" button

**Expected Results:**
- Takes 60-90 seconds to run (7 models × 5 windows)
- Progress shown in console/logs
- Shows "Model Comparison Results" section with:
  - **Model Rankings Table**: Top performers listed by MAE
  - **Comparison Plot**: Bar chart of model performance
  - **Metric Heatmap**: Window vs model performance matrix
  - **Performance Summary**: Best model details

### ✅ Test 6: Config Save/Load
**Steps:**
1. After running a forecast, scroll down
2. Click "Save Best Model Configuration" (if backtesting) or "Save Current Config"
3. Download JSON file
4. In sidebar, use "Load Previous Configuration"
5. Upload the JSON file
6. Config should auto-populate in the main area

**Expected Results:**
- Config saves successfully (downloads JSON)
- Config loads successfully with all parameters filled
- Model type and parameters visible in config area

### ✅ Test 7: CSV Upload
**Steps:**
1. Prepare a simple CSV with columns: ds, y
2. Click "Upload CSV File" in sidebar
3. Upload your CSV
4. Run forecast on custom data

**Expected Results:**
- CSV loads without error
- Data quality validation runs
- Forecast works on custom data
- Metrics calculated correctly

### ✅ Test 8: Error Handling
**Steps:**
1. Try uploading a CSV with missing 'ds' or 'y' column
2. Try uploading a CSV with only 5 rows
3. Try uploading non-CSV file

**Expected Results:**
- Clear error messages shown
- App doesn't crash
- User can fix issue and retry

---

## Key Features to Verify

| Feature | Status | Notes |
|---------|--------|-------|
| Load sample data | ✅ | AirPassengers dataset |
| Single forecast | ✅ | Multiple models available |
| Backtesting mode | ✅ | 5-window rolling CV, 7 models |
| Model comparison | ✅ | Rankings, heatmaps, plots |
| Config save | ✅ | JSON format |
| Config load | ✅ | File upload, auto-fill |
| CSV upload | ✅ | Custom data support |
| Validation | ✅ | Critical errors, warnings |
| Visualizations | ✅ | Plotly interactive charts |
| Error messages | ✅ | User-friendly feedback |

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: Reinstall requirements
```bash
pip install -r requirements.txt
```

### Issue: "Port 8501 already in use"
**Solution**: Use different port
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: "torch or pytorch-lightning not found"
**Solution**: These take time to install. You can comment out NeuralForecast models temporarily
```bash
pip install torch pytorch-lightning
```

### Issue: Backtesting takes too long or crashes
**Solution**: 
- Try with smaller dataset first
- Reduce number of models temporarily
- Check available memory: `top` or `Activity Monitor`

---

## Performance Benchmarks

**Expected Execution Times:**
- Single forecast (StatsForecast): 3-8 seconds
- Single forecast (MLForecast): 5-15 seconds
- Single forecast (NeuralForecast): 10-30 seconds
- Backtesting (7 models, 5 windows): 60-120 seconds
- Total backtest memory usage: ~600MB-1GB

---

## After Testing

Once all tests pass:
1. Note any error messages or unexpected behavior
2. Fix any issues found
3. Push to GitHub
4. Deploy to Streamlit Cloud
