import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from brainbuilding.train.pipelines import (
    PT_TANGENT_KNN_STEPS,
    PT_TANGENT_KPCA_KNN_STEPS,
    PT_CSP_KNN_STEPS,
    PT_CSP_SVC_STEPS,
    CSP_SVC_STEPS
)
from tqdm import tqdm
from brainbuilding.train.evaluation import evaluate_pipeline_with_adaptation
from brainbuilding.core.transformers import AUGMENTED_COVARIANCE_TRANSFORMER_STEPS, COVARIANCE_TRANSFORMER_STEPS

# Load preprocessed data
DATASET_FNAME = 'data/preprocessed/motor-imagery-2.npy'
data = np.load(DATASET_FNAME, allow_pickle=True)

# Prepare data for evaluation
X = data
y = data['label']
X = X[['sample', 'subject_id', 'event_id', 'is_background']]

# Initialize covariance transformer
# covariance_transformer = Pipeline(AUGMENTED_COVARIANCE_TRANSFORMER_STEPS)
covariance_transformer = Pipeline(COVARIANCE_TRANSFORMER_STEPS)

# Define pipelines to evaluate
pipelines = {
    'PT_CSP_SVC': PT_CSP_SVC_STEPS,
    'CSP_SVC': CSP_SVC_STEPS,
    'PT_TANGENT_KNN': PT_TANGENT_KNN_STEPS,
    'PT_TANGENT_KPCA_KNN': PT_TANGENT_KPCA_KNN_STEPS,
    'PT_CSP_KNN': PT_CSP_KNN_STEPS,
}

# Store results for all pipelines
all_results = []

# Evaluate each pipeline
for pipeline_name, pipeline_steps in tqdm(pipelines.items(), desc="Evaluating pipelines"):
    print(f"Evaluating pipeline: {pipeline_name}")
    
    # Evaluate pipeline with adaptation
    metrics_df, _ = evaluate_pipeline_with_adaptation(
        X=X,
        y=y,
        covariance_transformer=covariance_transformer,
        pipeline_steps=pipeline_steps,
        use_background=False
    )
    
    # Add pipeline name to results
    metrics_df['pipeline'] = pipeline_name
    all_results.append(metrics_df)

# Combine all results
combined_results = pd.concat(all_results, ignore_index=True)

# Create box plot
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
output_dir = Path('results-no-background-3-segments/plots')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'pipeline_comparison.pdf')
plt.close()

# Calculate and save summary statistics
summary_stats = combined_results.groupby(['pipeline', 'measure'])['value'].describe()
summary_stats.to_csv(output_dir / 'pipeline_summary_stats.csv')

# Save full combined results
combined_results.to_csv(output_dir / 'pipeline_full_results.csv', index=False)
