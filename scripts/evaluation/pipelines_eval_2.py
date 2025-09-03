import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from brainbuilding.train.pipelines import (
    GENERAL_PIPELINE_STEPS
)
from tqdm import tqdm
from brainbuilding.train.evaluation import evaluate_pipeline
import itertools

# Load preprocessed data
DATASET_FNAME = 'data/preprocessed/motor-imagery-2.npy'
data = np.load(DATASET_FNAME, allow_pickle=True)

# Prepare data for evaluation
X = data
y = data['label']
X = X[['sample', 'subject_id', 'event_id', 'is_background']]

OUTPUT_DIR = Path('results-final/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_FNAME = 'pipeline_comparison.pdf'
SUMMARY_STATS_FNAME = 'pipeline_summary_stats.csv'
FULL_RESULTS_FNAME = 'pipeline_full_results.csv'

def save_results(combined_results):
    # Create box plot if results exist
    if not combined_results.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=combined_results,
            x='measure',
            y='value',
            hue='pipeline'
        )
        plt.title('Pipeline Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plt.savefig(OUTPUT_DIR / PLOT_FNAME)
        plt.close()

        # Calculate and save summary statistics
        summary_stats = combined_results.groupby(['pipeline', 'measure'])['value'].describe()
        summary_stats.to_csv(OUTPUT_DIR / SUMMARY_STATS_FNAME)

        # Save full combined results
        combined_results.to_csv(OUTPUT_DIR / FULL_RESULTS_FNAME, index=False)
    else:
        print("No results to plot or save.")

# Generate all combinations of pipeline steps
pipeline_combinations = list(itertools.product(*GENERAL_PIPELINE_STEPS))

# Helper to flatten steps and create a name
all_pipelines = {}
for combo in pipeline_combinations:
    # Flatten the steps (each is a list of (name, obj) tuples)
    steps = [step for group in combo for step in group]
    # Create a name by joining step names
    name = '+'.join([s[0] for s in steps]) if steps else 'empty'
    all_pipelines[name] = steps

# Store results for all pipelines
all_results = []

# Evaluate each pipeline
for pipeline_name, pipeline_steps in tqdm(all_pipelines.items(), desc="Evaluating pipelines"):
    print(f"Evaluating pipeline: {pipeline_name}")
    metrics_df = evaluate_pipeline(
        X=X.copy(),
        y=y,
        pipeline_steps=pipeline_steps,
    )
    metrics_df['pipeline'] = pipeline_name
    all_results.append(metrics_df)
    combined_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    save_results(combined_results)