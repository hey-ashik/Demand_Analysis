"""
Flask Web Application Module
Professional dashboard-style web interface for regression experiments.
"""

import os
import json
import logging
import webbrowser
import threading
import pandas as pd

from flask import Flask, render_template, request, jsonify

from ml_pipeline import MLPipeline
from ai_predictor import AIPredictor

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Globals
pipeline = MLPipeline()
ai_predictor = AIPredictor()
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'results', 'experiments.csv')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


@app.route('/')
def index():
    """Render the dashboard."""
    return render_template('dashboard.html')


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List CSV files in data directory and project root."""
    files = []
    search_dirs = [DATA_DIR, os.path.dirname(__file__)]
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith('.csv') and f != 'experiments.csv':
                    full_path = os.path.join(d, f)
                    files.append({
                        'name': f,
                        'path': full_path,
                        'size': os.path.getsize(full_path)
                    })
    # De-duplicate by name
    seen = set()
    unique = []
    for f in files:
        if f['name'] not in seen:
            seen.add(f['name'])
            unique.append(f)
    return jsonify(unique)


@app.route('/api/columns', methods=['POST'])
def get_columns():
    """Get numeric columns from a dataset."""
    data = request.json
    filepath = data.get('filepath', '')
    try:
        pipeline.load_dataset(filepath)
        cols = pipeline.get_numeric_columns()
        return jsonify({'columns': cols, 'rows': len(pipeline.dataset)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/run_experiment', methods=['POST'])
def run_experiment():
    """Run an ML experiment."""
    data = request.json
    filepath = data.get('filepath', '')
    target = data.get('target', '')
    history = int(data.get('history', 5))
    regressor = data.get('regressor', 'Linear Regression')
    n_folds = int(data.get('n_folds', 5))

    try:
        metrics = pipeline.run_experiment(
            filepath, target, history, regressor, n_folds, RESULTS_FILE
        )
        actual, predicted = pipeline.get_predictions()
        return jsonify({
            'metrics': metrics,
            'actual': actual[:200] if actual else [],
            'predicted': predicted[:200] if predicted else [],
            'samples': len(actual) if actual else 0
        })
    except Exception as e:
        logger.error(f"Experiment error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/run_multiple_experiments', methods=['POST'])
def run_multiple_experiments():
    """Run multiple ML experiments in parallel."""
    import concurrent.futures
    data = request.json
    filepath = data.get('filepath', '')
    target = data.get('target', '')
    histories = data.get('histories', [5])
    regressors = data.get('regressors', ['Linear Regression'])
    folds = data.get('folds', [5])

    combos = []
    for r in regressors:
        for h in histories:
            for f in folds:
                combos.append((r, h, f))

    if not combos:
        return jsonify({'error': 'No experiment combinations provided.'}), 400

    def do_work(combo):
        reg, hist, fld = combo
        pipe = MLPipeline()
        pipe.load_dataset(filepath)
        pipe.generate_features(target, hist)
        metrics = pipe.train_and_evaluate(reg, fld)
        filename = os.path.basename(filepath)
        pipe.save_results(RESULTS_FILE, filename, hist, reg)
        return metrics

    try:
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_combo = {executor.submit(do_work, c): c for c in combos}
            for future in concurrent.futures.as_completed(future_to_combo):
                future.result()
                completed += 1
        return jsonify({
            'status': 'ok',
            'completed': completed,
            'total': len(combos)
        })
    except Exception as e:
        logger.error(f"Multiple experiment error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get all experiment results."""
    try:
        df = pipeline.load_results(RESULTS_FILE)
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/ai_analyze', methods=['POST'])
def ai_analyze():
    """Run AI analysis on experiment data."""
    data = request.json
    api_key = data.get('api_key', '')
    query = data.get('query', '')
    experiment_data = data.get('experiment_data', None)

    if api_key:
        ai_predictor.set_api_key(api_key)

    if experiment_data is None:
        df = pipeline.load_results(RESULTS_FILE)
        if not df.empty:
            experiment_data = df.to_dict('records')
        elif pipeline.metrics:
            experiment_data = pipeline.metrics
        else:
            experiment_data = {"message": "No experiment data available."}

    response = ai_predictor.analyze_results(experiment_data, query)
    return jsonify({'response': response})


@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """Clear all experiment results."""
    try:
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        pipeline.experiment_history = []
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def open_browser():
    """Open the browser after a short delay."""
    import time
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')


def launch_web():
    """Launch the Flask web application."""
    threading.Thread(target=open_browser, daemon=True).start()
    print("\n" + "=" * 60)
    print("  🌐  Ontario Demand ML Lab — Web Dashboard")
    print("  📍  http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False)


if __name__ == '__main__':
    launch_web()
