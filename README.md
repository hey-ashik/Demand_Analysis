<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyQt5-GUI-green?style=for-the-badge&logo=qt&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-Web-orange?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-red?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-AI-purple?style=for-the-badge&logo=openai&logoColor=white" />
</p>

<h1 align="center">Ontario Demand Analysis</h1>
<p align="center">
  <strong>A Hybrid Machine Learning Experimentation Platform for Regression Model Training, Evaluation & AI-Powered Insights</strong>
</p>
<p align="center">
  Developed by <strong>Ashikul Islam</strong> — v21.0
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
  - [Desktop Application](#desktop-application-pyqt5)
  - [Web Application](#web-application-flask)
- [Supported Regressors](#supported-regressors)
- [AI Predictor (Groq Integration)](#ai-predictor-groq-integration)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

**Ontario Demand Analysis** is a professional-grade, dual-interface machine learning platform designed for training, evaluating, and comparing regression models on time-series energy demand data. The platform provides:

- **Desktop Application** — A feature-rich PyQt5 GUI with interactive charts, metric cards, and real-time experiment tracking.
- **Web Application** — A modern Flask-powered dashboard accessible from any browser, fully responsive across desktop, tablet, and mobile devices.

Both interfaces share the same ML pipeline, enabling seamless experimentation and consistent results across platforms.

---

## Key Features

| Feature | Description |
|---|---|
| **Dual Interface** | Choose between Desktop (PyQt5) or Web (Flask) at launch |
| **Lag-based Feature Engineering** | Automatic generation of time-series lag features with configurable history window |
| **Multiple Regressors** | Linear Regression, Random Forest, and Gradient Boosting out of the box |
| **K-Fold Cross Validation** | Robust model evaluation with configurable number of folds |
| **Multiple Run Experiments** | Run parallel experiments across different regressors, history sizes, and fold counts simultaneously |
| **Interactive Visualizations** | Bar charts (MAE, MSE, R²), scatter plots (Actual vs. Predicted), powered by Chart.js and Matplotlib |
| **AI-Powered Insights** | Integrated Groq Cloud API (OpenAI-compatible) for intelligent, context-aware experiment analysis |
| **Experiment Persistence** | All results saved to CSV for historical tracking and comparison |
| **Responsive Web Design** | Mobile-first layout with hamburger menu, collapsible sidebar, and adaptive chart sizing |
| **Professional Icons** | FontAwesome 6 (Web) and QtAwesome (Desktop) vector icons throughout the UI |

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   main.py                        │
│            (Startup Dialog - Mode Selection)      │
│                                                  │
│         ┌──────────┐     ┌──────────┐            │
│         │ Desktop  │     │   Web    │            │
│         │  (PyQt5) │     │ (Flask)  │            │
│         └────┬─────┘     └────┬─────┘            │
│              │                │                  │
│         desktop_app.py    app.py                  │
│              │                │                  │
│              └───────┬────────┘                  │
│                      │                           │
│              ┌───────┴────────┐                  │
│              │  ml_pipeline.py │                  │
│              │  (Core Engine)  │                  │
│              └───────┬────────┘                  │
│                      │                           │
│              ┌───────┴────────┐                  │
│              │ ai_predictor.py│                  │
│              │  (Groq Cloud)  │                  │
│              └────────────────┘                  │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
Ontario_Demand/
│
├── main.py                 # Entry point — Startup dialog and mode selection
├── desktop_app.py          # PyQt5 desktop application (GUI)
├── app.py                  # Flask web application (routes & API)
├── ml_pipeline.py          # Core ML engine (feature engineering, training, evaluation)
├── ai_predictor.py         # AI analysis module (Groq Cloud API integration)
│
├── templates/
│   └── dashboard.html      # Web dashboard (single-page application)
│
├── static/
│   ├── css/                # Static CSS assets (if any)
│   └── js/                 # Static JS assets (if any)
│
├── data/                   # Place your CSV datasets here
├── results/
│   └── experiments.csv     # Auto-generated experiment results log
├── logs/
│   └── app.log             # Application logs
│
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Prerequisites

Before setting up the project, ensure the following are installed on your system:

| Requirement | Version | Purpose |
|---|---|---|
| **Python** | 3.10 or higher | Core runtime |
| **pip** | Latest | Package management |
| **Git** | Latest | Version control (optional) |

> **Note:** On Windows, make sure Python is added to your system PATH during installation.

---

## Installation

Follow these steps *exactly* to set up the project in a virtual environment:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Ontario_Demand.git
cd Ontario_Demand
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
```

**macOS / Linux:**
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

> After activation, your terminal prompt should show `(venv)` at the beginning.

### 4. Upgrade pip (Recommended)

```bash
pip install --upgrade pip
```

### 5. Install All Dependencies

```bash
pip install -r requirements.txt
```

This will install the following libraries into your virtual environment:

| Package | Version | Purpose |
|---|---|---|
| `PyQt5` | >= 5.15.0 | Desktop GUI framework |
| `Flask` | >= 2.3.0 | Web application framework |
| `pandas` | >= 1.5.0 | Data manipulation and CSV handling |
| `numpy` | >= 1.24.0 | Numerical computations |
| `scikit-learn` | >= 1.2.0 | Machine learning models and evaluation |
| `matplotlib` | >= 3.7.0 | Chart generation (desktop app) |
| `requests` | >= 2.28.0 | HTTP requests for Groq API |
| `groq` | >= 0.4.0 | Groq Cloud SDK |
| `python-dotenv` | >= 1.0.0 | Load environment variables from `.env` |
| `qtawesome` | Latest | Professional FontAwesome icons in PyQt5 |

### 6. Verify Installation

```bash
pip list
```

Ensure all packages listed above appear in the output without errors.

---

## Configuration

### Groq API Key (Required for AI Predictor)

The AI Predictor feature requires a Groq Cloud API key. To configure it:

1. **Get your API key** from [console.groq.com](https://console.groq.com)

2. **Create a `.env` file** in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

3. The application will automatically load this key at startup using `python-dotenv`.

> **Important:** Never commit your `.env` file to version control. Add it to `.gitignore`:
> ```
> .env
> venv/
> __pycache__/
> logs/
> ```

### Dataset Setup

Place your CSV dataset files in either:
- The **project root directory** (`Ontario_Demand/`)
- The **`data/` directory** (`Ontario_Demand/data/`)

The application will automatically detect all `.csv` files in both locations.

---

## Running the Application

### Start the Application

With your virtual environment activated, run:

```bash
python main.py
```

A startup dialog will appear with two options:

| Mode | Description |
|---|---|
| **Desktop Application** | Launches the full PyQt5 GUI with tabs for experiments, results, visualizations, and AI analysis |
| **Web Application** | Starts a Flask server and opens `http://127.0.0.1:5000` in your default browser |

### Quick Launch (Web Only)

If you only need the web dashboard:

```bash
python app.py
```

---

## Usage Guide

### Desktop Application (PyQt5)

The desktop application features four main tabs:

#### 1. Experiment Tab
- **Load Dataset** — Browse and select a CSV file
- **Configure Parameters** — Choose target column, history window size, regressor, and number of CV folds
- **Run Experiment** — Execute a single experiment
- **Multiple Run Experiment** — Run parallel experiments across various combinations
- **Show Metrics** — Display MAE, MSE, and R² with standard deviations
- **Plot Results** — Generate Actual vs. Predicted scatter plots
- **Save Results** — Persist experiment results to CSV

#### 2. Results Tab
- View all historical experiment results in a sortable table
- Refresh to see newly added results

#### 3. Visualization Tab
- Select chart type: Actual vs. Predicted, MAE Comparison, MSE Comparison, or R² Score Comparison
- Interactive bar charts with hover tooltips showing exact values

#### 4. AI Predictor Tab
- Enter custom queries about your experiments
- Auto-analyze all results with one click
- Powered by Groq Cloud API (model: `openai/gpt-oss-120b`)

---

### Web Application (Flask)

The web dashboard provides five main sections accessible via the sidebar:

#### Dashboard
- **Metric Cards** — Total Experiments, Best MAE, Best R² Score, Best Model
- **Charts** — MAE by Experiment, R² Score by Experiment (interactive bar charts)
- **Results Table** — Quick overview of all logged experiments

#### Run Experiment
- Dataset selection with automatic column detection
- Configurable history window, regressor, and CV folds
- Single or Multiple Run Experiment modes

#### Results
- Full results table with all experiment metrics
- Clear results functionality

#### Graph Analytics
- MAE Comparison chart
- MSE Comparison chart
- R² Score Comparison chart
- Actual vs. Predicted scatter plot (Plotly.js)

#### AI Predictor
- Natural language queries about experiment data
- Automatic comprehensive analysis
- Responses in clean, professional plain text

---

## Supported Regressors

| Regressor | Description | Key Parameters |
|---|---|---|
| **Linear Regression** | Simple, interpretable baseline model | Default settings |
| **Random Forest Regressor** | Ensemble method using decision trees | `n_estimators=100`, `random_state=42` |
| **Gradient Boosting Regressor** | Sequential tree boosting for high accuracy | `n_estimators=100`, `max_depth=5`, `random_state=42` |

---

## AI Predictor (Groq Integration)

The AI Predictor module connects to the **Groq Cloud API** for intelligent experiment analysis:

- **Model:** `openai/gpt-oss-120b`
- **Temperature:** `0.5` (balanced creativity and consistency)
- **Max Tokens:** `500` (concise professional responses)
- **Persona:** MLOps Analyst — Professional, data-driven, plain-text responses
- **Fallback:** Offline statistical analysis when API is unavailable

### What the AI Can Do:
- Analyze and compare model performance metrics
- Explain why certain regressors outperform others
- Suggest hyperparameter tuning strategies
- Recommend next steps for improving model accuracy
- Answer general ML and data science questions

---

## API Endpoints

The Flask web application exposes the following REST API:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Render the dashboard HTML |
| `GET` | `/api/datasets` | List available CSV datasets |
| `POST` | `/api/columns` | Get numeric columns from a dataset |
| `POST` | `/api/run_experiment` | Run a single ML experiment |
| `POST` | `/api/run_multiple_experiments` | Run parallel experiments |
| `GET` | `/api/results` | Retrieve all experiment results |
| `POST` | `/api/ai_analyze` | Run AI analysis on experiment data |
| `POST` | `/api/clear_results` | Clear all saved results |

---

## Technologies Used

| Category | Technology | Version |
|---|---|---|
| **Language** | Python | 3.10+ |
| **Desktop GUI** | PyQt5 | 5.15+ |
| **Web Framework** | Flask | 2.3+ |
| **ML Library** | scikit-learn | 1.2+ |
| **Data Processing** | pandas, NumPy | 1.5+, 1.24+ |
| **Charting (Desktop)** | Matplotlib | 3.7+ |
| **Charting (Web)** | Chart.js, Plotly.js | 4.4+, 2.27+ |
| **Icons (Desktop)** | QtAwesome (FontAwesome 5) | Latest |
| **Icons (Web)** | FontAwesome 6 CDN | 6.5.1 |
| **AI/LLM** | Groq Cloud API | Latest |
| **AI Model** | openai/gpt-oss-120b | Latest |
| **Environment** | python-dotenv | 1.0+ |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Ensure your virtual environment is activated and all dependencies are installed |
| `GROQ_API_KEY not found` | Create a `.env` file in the project root with `GROQ_API_KEY=your_key` |
| Charts not rendering (Web) | Hard-refresh the browser with `Ctrl+Shift+R` |
| PyQt5 import error on macOS | Run `brew install pyqt5` or install via `pip install PyQt5` |
| Port 5000 already in use | Change the port in `app.py` (`app.run(port=5001)`) or kill the existing process |

---

## License

This project is developed for academic and research purposes.

**Developed by Ashikul Islam**

---

<p align="center">
  <em>Built with care using Python, PyQt5, Flask, scikit-learn, and Groq AI</em>
</p>
