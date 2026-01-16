# Patient Timeline Visualization App

Streamlit app for visualizing individual patient disease trajectories and signature loadings over time.

## Files

- `patient_timeline_app_compact.py` - Main Streamlit application
- `app_patients_compact_nolr.pt` - Pre-computed model data for sample patients (~5MB)
- `prs_names.csv` - Polygenic Risk Score feature names
- `disease_names.csv` - Disease names
- `requirements.txt` - Python dependencies

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run patient_timeline_app_compact.py
```

## Deployment

This app can be deployed to:
- **Streamlit Cloud** - Upload the entire folder to GitHub and connect via Streamlit Cloud
- **Heroku** - Use a Procfile with: `web: streamlit run patient_timeline_app_compact.py --server.port=$PORT --server.address=0.0.0.0`
- **AWS/GCP/Azure** - Deploy as a container or serverless function
- **Local server** - Run behind nginx/apache with reverse proxy

## Requirements

- Python 3.8+
- ~100MB disk space for dependencies
- ~10MB RAM minimum (more for smoother performance)

## Notes

- All sample patient data is pre-computed and included in `app_patients_compact_nolr.pt`
- No external API calls or database connections required
- All files must be in the same directory for the app to work

