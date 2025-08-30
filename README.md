# 🏠 SF Smart Rental Advisor

**End-to-end mini project**: clean Airbnb listings for San Francisco, train a predictive model, and serve a Streamlit UI with natural-language explanations (LLM-ready, default `gpt-5-nano`).

---

## 🚀 Features
- **Data pipeline**: raw CSV → cleaned dataset → optional one-hot encoded table.  
- **Modeling**: train baseline (RandomForest) + advanced (XGBoost / LightGBM / CatBoost).  
- **Evaluation**: reports **MAE** and **R²** for comparison.  
- **Streamlit App**: simple form → nightly price prediction.  
- **LLM explanation**: optional, short rationale from `gpt-5-nano`, with fallback if API key missing.  

---

## 📂 Project Structure
rental-advisor/
├─ data/
│ ├─ raw/ # place listings.csv here
│ ├─ processed/ # cleaned_data.csv, featured_data.csv
├─ models/ # model.pkl, model_schema.json
├─ src/
│ ├─ data/clean.py
│ ├─ features/build_features.py
│ ├─ models/{train.py, io.py, evaluate.py}
│ ├─ explain/{llm_openai.py, explain_fallback.py}
│ ├─ utils.py
│ ├─ config.py
│ └─ init.py
├─ scripts/
│ ├─ build_dataset.py
│ └─ train_model.py
├─ app/streamlit_app.py
├─ requirements.txt
├─ .env.sample
└─ README.md

yaml
Copy code

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/avivreu7/rental-advisor.git
cd rental-advisor
2. Create virtual environment & install dependencies
bash
Copy code
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # on Windows
pip install -r requirements.txt
3. Prepare your dataset
Place your Airbnb listings CSV inside:
data/raw/listings.csv

4. Build processed data
bash
Copy code
python scripts\build_dataset.py
5. Train a model
bash
Copy code
python scripts\train_model.py --model rf
# or: xgb | lgbm | cat
6. Run the Streamlit app
bash
Copy code
$env:PYTHONPATH = (Get-Location).Path
streamlit run app\streamlit_app.py
🧠 LLM Explanation (Optional)
To enable natural-language explanations:

Copy .env.sample → .env

Add your OpenAI key:

ini
Copy code
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-5-nano
Rerun the app. If no key is provided, the app falls back to a simple rule-based explanation.

📊 Example Results
Model	MAE ($)	R²
RandomForest	70.83	0.464
XGBoost	…	…
CatBoost	…	…
LightGBM	…	…

(Fill with your own results after training each model.)

✅ What I Learned
How data cleaning and feature engineering affect model stability.

How tree-based models provide a strong baseline, while boosting improves accuracy.

How to integrate LLMs to make model outputs more human-friendly.

How to structure a reproducible ML project (data, scripts, src, app).

🔧 Troubleshooting
ModuleNotFoundError: src → ensure you run from project root or add:

python
Copy code
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
LightGBM build errors on Windows → skip and use XGBoost or CatBoost.

OpenAI error: temperature unsupported → fixed by not passing custom temperature to gpt-5-nano.

📜 License
MIT License © 2025 Aviv Reuven
