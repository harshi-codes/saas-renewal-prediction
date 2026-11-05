"""Builds and executes SaaS_Renewal_Prediction.ipynb (cell-by-cell walkthrough of the pipeline)."""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
C, M = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell

nb.cells = [
M("""# SaaS Subscription Renewal Prediction Using Decision Tree Classification
**Student:** Harshita Singh (Reg No. 2428010025, Section C) &nbsp;|&nbsp; **Supervisor:** Dr. Anshika Malsaria  
Department of Computer and Communication Engineering, Manipal University Jaipur

This notebook runs the project pipeline (`src/saas_renewal_prediction.py`) **cell by cell**, showing intermediate results."""),
M("## Step 0 — Imports & setup"),
C("""import os, sys, json, logging
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
%matplotlib inline
sys.path.insert(0, 'src')
import saas_renewal_prediction as srp
logging.getLogger().setLevel(logging.WARNING); logging.getLogger('SaaSRenewal').setLevel(logging.WARNING)
plt.rcParams['figure.dpi'] = 100
pd.set_option('display.max_columns', 30)

RAW_DIR, PROC_PATH = 'data/raw', 'data/processed/saas_processed.csv'
OUT_DIR, RES_DIR, FIG_DIR = 'outputs', 'results', 'figures'
for d in [RAW_DIR, os.path.dirname(PROC_PATH), OUT_DIR, RES_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)
print('Setup complete')"""),
M("## Step 1 — Load & validate the raw multi-table dataset (RavenStack)"),
C("""dfs = srp.load_and_validate_raw_data(RAW_DIR)
for name, d in dfs.items():
    print(f'{name:<20} rows={d.shape[0]:>6}  cols={d.shape[1]}')"""),
C("""display(dfs['subscriptions'].head()) if 'subscriptions' in dfs else display(list(dfs.values())[0].head())"""),
M("## Step 2 — Feature engineering within the observation window"),
C("""df = srp.process_and_engineer_features(dfs, PROC_PATH)
print('Processed dataset shape:', df.shape)
display(df.head())"""),
C("""print('Target distribution (renewed):')
vc = df['renewed'].value_counts()
display(pd.DataFrame({'count': vc, 'percent': (vc / len(df) * 100).round(2)}))"""),
M("## Step 3 — Customer-level train/test split (GroupShuffleSplit on `account_id`)"),
C("""train_df, test_df = srp.split_data_customer_level(df, test_size=0.2, random_state=42)
overlap = set(train_df.account_id) & set(test_df.account_id)
print(f'Train: {len(train_df)} subscriptions / {train_df.account_id.nunique()} accounts')
print(f'Test : {len(test_df)} subscriptions / {test_df.account_id.nunique()} accounts')
print('Accounts shared between train and test:', len(overlap))"""),
M("## Step 4 — Feature lists & preprocessing pipeline"),
C("""num_features, cat_features = srp.get_feature_lists()
preprocessor = srp.build_preprocessing_pipeline(num_features, cat_features)
X_train, y_train, groups_train = train_df[num_features + cat_features], train_df['renewed'], train_df['account_id']
X_test, y_test = test_df[num_features + cat_features], test_df['renewed']
print('Numerical features :', num_features)
print('Categorical features:', cat_features)"""),
M("## Step 5 — Tune the Decision Tree (GridSearchCV + StratifiedGroupKFold)"),
C("""best_dt_pipe, best_dt_params, best_dt_cv_score = srp.tune_decision_tree_group_aware(
    X_train, y_train, groups_train, preprocessor, random_state=42)
print('Best hyperparameters:', best_dt_params)
print(f'Group-aware CV ROC-AUC: {best_dt_cv_score:.4f}')"""),
M("## Step 6 — Train comparative models (Logistic Regression, Random Forest, Majority Baseline)"),
C("""comparative_models = srp.train_comparative_models_group_aware(
    X_train, y_train, groups_train, preprocessor, random_state=42)
all_models = {'Decision Tree': best_dt_pipe, **comparative_models}
print('Models trained:', list(all_models))"""),
M("## Step 7 — Evaluate on the untouched test set"),
C("""metrics_summary, predictions_df = srp.evaluate_all_models(all_models, X_test, y_test)
predictions_df['subscription_id'] = test_df['subscription_id'].values
predictions_df['account_id'] = test_df['account_id'].values
predictions_df.to_csv(os.path.join(RES_DIR, 'test_predictions.csv'), index=False)
with open(os.path.join(RES_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics_summary, f, indent=4)

rows = {m: {k: v for k, v in d.items() if isinstance(v, (int, float))} for m, d in metrics_summary.items()}
display(pd.DataFrame(rows).T.round(4))"""),
C("""from sklearn.metrics import classification_report
print(classification_report(y_test, best_dt_pipe.predict(X_test), target_names=['Not Renewed (0)', 'Renewed (1)'], zero_division=0))"""),
M("## Step 8 — Decision rules & feature importance"),
C("""import joblib
imp_df, all_feat_names, rules_txt = srp.extract_tree_rules_and_importance(best_dt_pipe, num_features, cat_features, OUT_DIR)
joblib.dump(best_dt_pipe, os.path.join(OUT_DIR, 'decision_tree_model.joblib'))
print(rules_txt)
display(imp_df.head(10))"""),
M("## Step 9 — Generate & display publication figures"),
C("""srp.generate_publication_figures(df, all_models, X_test, y_test, imp_df, all_feat_names, FIG_DIR)
print(sorted(os.listdir(FIG_DIR)))"""),
]
figs = [('fig1_class_distribution.png', 'Class distribution'), ('fig2_eda_features.png', 'EDA features'),
        ('fig3_confusion_matrices.png', 'Confusion matrices'), ('fig4_roc_curves.png', 'ROC curves'),
        ('fig5_decision_tree_visualization.png', 'Decision tree'), ('fig6_feature_importance.png', 'Feature importance')]
for fn, title in figs:
    nb.cells.append(M(f"### {title}"))
    nb.cells.append(C(f"display(Image(filename=os.path.join(FIG_DIR, '{fn}'), width=850))"))
nb.cells.append(M("## Summary"))
nb.cells.append(C("""dt = metrics_summary['Decision Tree']
print(f"Decision Tree -> Accuracy: {dt['accuracy']:.4f} | ROC-AUC: {dt['roc_auc']:.4f} | Macro-F1: {dt['f1_macro']:.4f}")
print(f"CV ROC-AUC (group-aware): {best_dt_cv_score:.4f}")"""))

nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
nbf.write(nb, 'SaaS_Renewal_Prediction.ipynb')
print('Notebook executed and saved: SaaS_Renewal_Prediction.ipynb')