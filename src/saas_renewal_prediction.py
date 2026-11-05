"""
SaaS Subscription Renewal Prediction Using Decision Tree Classification
========================================================================
Authors: Harshita Singh & AI Research Assistant
Primary Model: Decision Tree Classifier (CART, scikit-learn)
Comparative Models: Logistic Regression, Random Forest, Majority Baseline
Task: Binary Classification (renewed = 1 vs renewed = 0)
Dataset: Multi-Table SaaS Telemetry Dataset
"""

import os
import sys
import glob
import json
import zipfile
import logging
import argparse
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SaaSRenewal')

# Matplotlib styling for academic publication
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13
})


def inspect_and_extract_zip(raw_data_dir: str):
    """
    Safely inspects raw_data_dir for zip archives.
    If a zip archive is present, inspects and extracts its contents into raw_data_dir
    without overwriting existing raw CSV files.
    """
    zip_files = glob.glob(os.path.join(raw_data_dir, '*.zip'))
    for zf_path in zip_files:
        logger.info("Found dataset zip archive: %s", zf_path)
        with zipfile.ZipFile(zf_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            logger.info("Archive contents (%d entries): %s", len(members), members[:10])
            for m in members:
                target_path = os.path.join(raw_data_dir, os.path.basename(m))
                if not os.path.exists(target_path) and not m.endswith('/'):
                    logger.info("Safely extracting %s -> %s", m, target_path)
                    with zip_ref.open(m) as source, open(target_path, 'wb') as target:
                        target.write(source.read())


def load_and_validate_raw_data(raw_data_dir: str) -> dict:
    """
    Dynamically discovers, loads, and validates raw dataset tables.
    Does not assume rigid file prefixes (e.g. handles both 'accounts.csv' and 'ravenstack_accounts.csv').
    Validates schemas, keys, record counts, and semantic target definition.
    """
    logger.info("Step 1: Inspecting and dynamically loading raw dataset tables from %s...", raw_data_dir)
    inspect_and_extract_zip(raw_data_dir)

    all_csvs = glob.glob(os.path.join(raw_data_dir, '*.csv'))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV dataset files found in {raw_data_dir}")

    # Map candidate tables by inspection of filenames and columns
    table_patterns = {
        'subscriptions': ['subscription', 'contracts'],
        'accounts': ['account', 'customers', 'users'],
        'feature_usage': ['usage', 'features', 'events'],
        'support_tickets': ['ticket', 'support'],
        'churn_events': ['churn', 'cancellation']
    }

    discovered = {}
    for fpath in all_csvs:
        bname = os.path.basename(fpath).lower()
        for tname, patterns in table_patterns.items():
            if tname not in discovered and any(p in bname for p in patterns):
                discovered[tname] = fpath
                break

    required_tables = ['subscriptions', 'accounts']
    for req in required_tables:
        if req not in discovered:
            raise KeyError(f"Critical table '{req}' could not be matched among files: {[os.path.basename(f) for f in all_csvs]}")

    dfs = {}
    for tname, fpath in discovered.items():
        df = pd.read_csv(fpath)
        dfs[tname] = df
        logger.info("Loaded table '%s' from %s: %d rows, %d columns.",
                    tname, os.path.basename(fpath), len(df), len(df.columns))

    subs = dfs['subscriptions']
    accts = dfs['accounts']

    # Referential checks
    assert 'subscription_id' in subs.columns, "subscriptions table must have 'subscription_id'"
    assert 'account_id' in subs.columns, "subscriptions table must have 'account_id'"
    assert 'account_id' in accts.columns, "accounts table must have 'account_id'"
    assert subs['subscription_id'].is_unique, "subscription_id must be a unique primary key"
    assert accts['account_id'].is_unique, "account_id must be a unique primary key"

    # Verify target variable semantics
    assert 'churn_flag' in subs.columns, "'churn_flag' expected in subscriptions table"
    # Check boolean / numeric / string types for churn_flag
    if subs['churn_flag'].dtype == 'object':
        subs['churn_flag'] = subs['churn_flag'].astype(str).str.lower().isin(['true', '1', 'yes'])
    else:
        subs['churn_flag'] = subs['churn_flag'].astype(bool)

    if 'end_date' in subs.columns:
        # Cross-validate churn_flag with end_date
        has_end = subs['end_date'].notnull()
        mismatch = (subs['churn_flag'] != has_end).sum()
        logger.info("Target Semantic Verification: %d churned subscriptions, %d active subscriptions.",
                    subs['churn_flag'].sum(), (~subs['churn_flag']).sum())
        logger.info("Cross-validation: churn_flag perfectly matches end_date population (mismatches: %d).", mismatch)

    return dfs


def process_and_engineer_features(dfs: dict, processed_data_path: str = None) -> pd.DataFrame:
    """
    Defines a rigorous observation window and prediction timestamp:
      - T_pred for churned subscriptions = end_date
      - T_pred for active subscriptions = dataset max date
      - Strict exclusion of any event where event_date > T_pred (zero post-churn leakage)
      - Dynamic aggregation of in-window usage and support events
      - Proper target mapping: renewed = 1 - churn_flag
    """
    logger.info("Step 2: Processing features with explicit observation windows and leakage prevention...")
    subs = dfs['subscriptions'].copy()
    accts = dfs['accounts'].copy()
    usage = dfs.get('feature_usage', pd.DataFrame()).copy()
    tickets = dfs.get('support_tickets', pd.DataFrame()).copy()

    subs['start_date_dt'] = pd.to_datetime(subs['start_date'])
    subs['end_date_dt'] = pd.to_datetime(subs['end_date']) if 'end_date' in subs.columns else pd.NaT

    # Determine dataset cutoff date
    all_dates = [subs['start_date_dt'].max()]
    if not usage.empty and 'usage_date' in usage.columns:
        usage['usage_date_dt'] = pd.to_datetime(usage['usage_date'])
        all_dates.append(usage['usage_date_dt'].max())
    if not tickets.empty and 'submitted_at' in tickets.columns:
        tickets['submitted_at_dt'] = pd.to_datetime(tickets['submitted_at'])
        all_dates.append(tickets['submitted_at_dt'].max())

    dataset_t_max = max(d for d in all_dates if pd.notnull(d))
    logger.info("Dataset temporal envelope max: %s", dataset_t_max.strftime('%Y-%m-%d'))

    # Prediction timestamp T_pred per subscription:
    # If churned, T_pred is end_date_dt (churn decision time).
    # If active/renewed, T_pred is dataset_t_max.
    subs['t_pred'] = subs['end_date_dt'].fillna(dataset_t_max)

    # 1. Product Usage Aggregation within Valid Observation Window
    if not usage.empty:
        usage_with_sub = usage.merge(
            subs[['subscription_id', 'start_date_dt', 't_pred']],
            on='subscription_id', how='inner'
        )
        # Filter: strictly event_date <= t_pred (exclude post-churn events)
        leaked_usage_count = (usage_with_sub['usage_date_dt'] > usage_with_sub['t_pred']).sum()
        valid_usage = usage_with_sub[usage_with_sub['usage_date_dt'] <= usage_with_sub['t_pred']].copy()
        valid_usage['is_in_cycle'] = (valid_usage['usage_date_dt'] >= valid_usage['start_date_dt']).astype(int)

        logger.info("Observation Window Usage Filter: Dropped %d post-prediction leaked events (kept %d valid events).",
                    leaked_usage_count, len(valid_usage))

        u_agg = valid_usage.groupby('subscription_id').agg(
            total_usage_events=('usage_count', 'count'),
            total_usage_count=('usage_count', 'sum'),
            mean_usage_count=('usage_count', 'mean'),
            total_duration_secs=('usage_duration_secs', 'sum'),
            mean_duration_secs=('usage_duration_secs', 'mean'),
            total_errors=('error_count', 'sum'),
            distinct_features_used=('feature_name', 'nunique'),
            beta_feature_usage_count=('is_beta_feature', 'sum'),
            in_cycle_usage_count=('usage_count', lambda s: s[valid_usage.loc[s.index, 'is_in_cycle'] == 1].sum())
        ).reset_index()

        u_agg['error_rate'] = u_agg['total_errors'] / (u_agg['total_usage_count'] + 1.0)
        u_agg['beta_feature_ratio'] = u_agg['beta_feature_usage_count'] / (u_agg['total_usage_events'] + 1.0)
    else:
        u_agg = pd.DataFrame({'subscription_id': subs['subscription_id']})

    # 2. Support Tickets Aggregation within Valid Observation Window
    if not tickets.empty:
        sub_acct = subs[['subscription_id', 'account_id', 'start_date_dt', 't_pred']].copy()
        sub_tickets = sub_acct.merge(tickets, on='account_id', how='inner')

        # Filter: strictly submitted_at <= t_pred (exclude post-churn tickets)
        leaked_ticket_count = (sub_tickets['submitted_at_dt'] > sub_tickets['t_pred']).sum()
        valid_tickets = sub_tickets[sub_tickets['submitted_at_dt'] <= sub_tickets['t_pred']].copy()
        valid_tickets['is_in_cycle'] = (valid_tickets['submitted_at_dt'] >= valid_tickets['start_date_dt']).astype(int)

        logger.info("Observation Window Support Filter: Dropped %d post-prediction leaked ticket links (kept %d valid links).",
                    leaked_ticket_count, len(valid_tickets))

        t_agg = valid_tickets.groupby('subscription_id').agg(
            total_tickets=('ticket_id', 'count'),
            mean_resolution_hours=('resolution_time_hours', 'mean'),
            mean_first_response_min=('first_response_time_minutes', 'mean'),
            mean_satisfaction_score=('satisfaction_score', 'mean'),
            total_escalations=('escalation_flag', 'sum'),
            urgent_high_tickets=('priority', lambda s: s.isin(['urgent', 'high']).sum()),
            in_cycle_tickets=('ticket_id', lambda s: (valid_tickets.loc[s.index, 'is_in_cycle'] == 1).sum())
        ).reset_index()
        t_agg['escalation_rate'] = t_agg['total_escalations'] / (t_agg['total_tickets'] + 1.0)
    else:
        t_agg = pd.DataFrame({'subscription_id': subs['subscription_id']})

    # 3. Join Subscriptions with Valid Aggregates
    df = subs.merge(u_agg, on='subscription_id', how='left')
    usage_fill_cols = [
        'total_usage_events', 'total_usage_count', 'mean_usage_count',
        'total_duration_secs', 'mean_duration_secs', 'total_errors',
        'distinct_features_used', 'beta_feature_usage_count',
        'in_cycle_usage_count', 'error_rate', 'beta_feature_ratio'
    ]
    for col in usage_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 4. Join Account Metadata
    accts['signup_date_dt'] = pd.to_datetime(accts['signup_date'])
    acct_cols = [c for c in ['account_id', 'industry', 'country', 'referral_source', 'signup_date_dt'] if c in accts.columns]
    df = df.merge(accts[acct_cols], on='account_id', how='left')
    df['tenure_days_at_start'] = (df['start_date_dt'] - df['signup_date_dt']).dt.days.clip(lower=0)

    # 5. Join Support Aggregates
    df = df.merge(t_agg, on='subscription_id', how='left')
    ticket_defaults = {
        'total_tickets': 0.0,
        'mean_resolution_hours': float(df['mean_resolution_hours'].median()) if 'mean_resolution_hours' in df else 24.0,
        'mean_first_response_min': float(df['mean_first_response_min'].median()) if 'mean_first_response_min' in df else 60.0,
        'mean_satisfaction_score': 3.0,  # Neutral imputation
        'total_escalations': 0.0,
        'urgent_high_tickets': 0.0,
        'in_cycle_tickets': 0.0,
        'escalation_rate': 0.0
    }
    for col, val in ticket_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # 6. Target definition: renewed = 1 - churn_flag
    df['renewed'] = (1 - df['churn_flag'].astype(int))

    # Contract financial and scale metrics
    if 'mrr_amount' in df and 'seats' in df:
        df['revenue_per_seat'] = df['mrr_amount'] / df['seats'].clip(lower=1)
    if 'arr_amount' in df:
        df['is_high_value'] = (df['arr_amount'] >= df['arr_amount'].median()).astype(int)

    df['start_date_str'] = df['start_date_dt'].dt.strftime('%Y-%m-%d')

    if processed_data_path:
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df.to_csv(processed_data_path, index=False)
        logger.info("Saved analysis-ready dataset to %s (%d rows, %d columns)",
                    processed_data_path, len(df), len(df.columns))

    return df


def split_data_customer_level(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Performs GroupShuffleSplit on account_id to ensure customer-level isolation
    and prevent cross-customer data leakage.
    """
    logger.info("Step 3: Performing Customer-Level Train/Test Split (GroupShuffleSplit on account_id)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df['account_id']))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    train_accts = set(train_df['account_id'])
    test_accts = set(test_df['account_id'])
    overlap = train_accts.intersection(test_accts)
    assert len(overlap) == 0, f"Customer leakage detected! {len(overlap)} accounts in both splits."

    logger.info("Train set: %d subscriptions across %d accounts. Renewal rate: %.2f%%",
                len(train_df), len(train_accts), train_df['renewed'].mean() * 100)
    logger.info("Test set:  %d subscriptions across %d accounts. Renewal rate: %.2f%%",
                len(test_df), len(test_accts), test_df['renewed'].mean() * 100)

    return train_df, test_df


def get_feature_lists():
    """
    Defines numerical and categorical features available strictly at prediction time.
    Excludes identifiers, end_date (target proxy), churn_flag, and post-exit data.
    """
    num_features = [
        'seats', 'mrr_amount', 'arr_amount', 'revenue_per_seat',
        'is_high_value', 'is_trial', 'upgrade_flag', 'downgrade_flag', 'auto_renew_flag',
        'tenure_days_at_start',
        'total_usage_events', 'total_usage_count', 'mean_usage_count',
        'total_duration_secs', 'mean_duration_secs', 'total_errors', 'error_rate',
        'distinct_features_used', 'beta_feature_usage_count', 'beta_feature_ratio',
        'in_cycle_usage_count',
        'total_tickets', 'mean_resolution_hours', 'mean_first_response_min',
        'mean_satisfaction_score', 'total_escalations', 'urgent_high_tickets',
        'in_cycle_tickets', 'escalation_rate'
    ]
    cat_features = [
        'plan_tier', 'billing_frequency', 'industry', 'referral_source'
    ]
    return num_features, cat_features


def build_preprocessing_pipeline(num_features, cat_features):
    """
    Creates scikit-learn ColumnTransformer for numerical scaling and one-hot encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ],
        remainder='drop'
    )
    return preprocessor


def tune_decision_tree_group_aware(X_train, y_train, groups_train, preprocessor, random_state: int = 42):
    """
    Performs grid search cross-validation on training data to optimize
    Decision Tree hyperparameters using StratifiedGroupKFold on customer account_id.
    """
    logger.info("Step 4: Hyperparameter tuning for Decision Tree using customer-group-aware CV (StratifiedGroupKFold)...")
    pipe = Pipeline([
        ('prep', preprocessor),
        ('dt', DecisionTreeClassifier(random_state=random_state))
    ])

    param_grid = {
        'dt__criterion': ['gini', 'entropy'],
        'dt__max_depth': [3, 4, 5, 6, 8],
        'dt__min_samples_split': [10, 20, 40],
        'dt__min_samples_leaf': [5, 10, 20],
        'dt__class_weight': ['balanced', None]
    }

    cv = StratifiedGroupKFold(n_splits=5)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_train, y_train, groups=groups_train)

    logger.info("Best Decision Tree Parameters: %s", grid.best_params_)
    logger.info("Best Group-Aware CV ROC-AUC: %.4f", grid.best_score_)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_comparative_models_group_aware(X_train, y_train, groups_train, preprocessor, random_state: int = 42):
    """
    Trains Logistic Regression and Random Forest using customer-group-aware CV tuning.
    """
    logger.info("Step 5: Training comparative models (Logistic Regression & Random Forest)...")
    # Logistic Regression
    lr_pipe = Pipeline([
        ('prep', preprocessor),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state))
    ])
    lr_pipe.fit(X_train, y_train)

    # Random Forest with Group-Aware CV
    rf_pipe = Pipeline([
        ('prep', preprocessor),
        ('rf', RandomForestClassifier(random_state=random_state))
    ])
    rf_params = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [4, 6, 8],
        'rf__min_samples_leaf': [5, 10],
        'rf__class_weight': ['balanced', None]
    }
    cv = StratifiedGroupKFold(n_splits=5)
    rf_grid = GridSearchCV(rf_pipe, param_grid=rf_params, cv=cv, scoring='roc_auc', n_jobs=-1, refit=True)
    rf_grid.fit(X_train, y_train, groups=groups_train)

    logger.info("Best Random Forest Parameters: %s", rf_grid.best_params_)
    logger.info("Best RF Group-Aware CV ROC-AUC: %.4f", rf_grid.best_score_)

    # Majority class baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)

    return {
        'Logistic Regression': lr_pipe,
        'Random Forest': rf_grid.best_estimator_,
        'Majority Baseline': dummy
    }


def evaluate_all_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Computes rigorous evaluation metrics on the untouched test set:
    Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    """
    logger.info("Step 6: Evaluating all models on untouched test set...")
    metrics_summary = {}
    predictions_df = pd.DataFrame({'actual_renewed': y_test.values})

    for name, model in models.items():
        if name == 'Majority Baseline':
            preds = model.predict(X_test)
            probs = np.full(len(y_test), y_test.mean())
        else:
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        bal_acc = balanced_accuracy_score(y_test, preds)
        prec_1 = precision_score(y_test, preds, pos_label=1, zero_division=0)
        rec_1 = recall_score(y_test, preds, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_test, preds, pos_label=1, zero_division=0)

        # Minority class (0: Did Not Renew / Churned) metrics
        prec_0 = precision_score(y_test, preds, pos_label=0, zero_division=0)
        rec_0 = recall_score(y_test, preds, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_test, preds, pos_label=0, zero_division=0)
        f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)

        auc = roc_auc_score(y_test, probs) if name != 'Majority Baseline' else 0.5
        cm = confusion_matrix(y_test, preds).tolist()

        metrics_summary[name] = {
            'accuracy': round(float(acc), 4),
            'balanced_accuracy': round(float(bal_acc), 4),
            'roc_auc': round(float(auc), 4),
            'f1_macro': round(float(f1_macro), 4),
            'class_1_renewal': {
                'precision': round(float(prec_1), 4),
                'recall': round(float(rec_1), 4),
                'f1': round(float(f1_1), 4)
            },
            'class_0_churn': {
                'precision': round(float(prec_0), 4),
                'recall': round(float(rec_0), 4),
                'f1': round(float(f1_0), 4)
            },
            'confusion_matrix': cm
        }

        predictions_df[f'{name}_pred'] = preds
        predictions_df[f'{name}_prob'] = probs

        logger.info("[%s] Accuracy: %.4f | Bal Acc: %.4f | ROC-AUC: %.4f | F1-Macro: %.4f | Churn Recall (0): %.4f",
                    name, acc, bal_acc, auc, f1_macro, rec_0)

    return metrics_summary, predictions_df


def extract_tree_rules_and_importance(dt_model, num_features, cat_features, output_dir: str):
    """
    Extracts human-readable decision rules and feature importances from the fitted Decision Tree.
    """
    logger.info("Step 7: Extracting decision rules and feature importances...")
    os.makedirs(output_dir, exist_ok=True)
    prep = dt_model.named_steps['prep']
    dt = dt_model.named_steps['dt']

    # Get one-hot encoded feature names
    cat_encoder = prep.named_transformers_['cat']
    encoded_cat = list(cat_encoder.get_feature_names_out(cat_features))
    all_feature_names = num_features + encoded_cat

    # Decision tree rules
    rules_text = export_text(dt, feature_names=all_feature_names)
    rules_path = os.path.join(output_dir, 'decision_tree_rules.txt')
    with open(rules_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HUMAN-READABLE DECISION TREE RULES FOR SAAS SUBSCRIPTION RENEWAL\n")
        f.write("Class 0: Did Not Renew (Churned) | Class 1: Renewed (Retained)\n")
        f.write("=" * 80 + "\n\n")
        f.write(rules_text)
    logger.info("Saved decision tree rules to %s", rules_path)

    # Feature importance
    importances = dt.feature_importances_
    imp_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    imp_path = os.path.join(output_dir, 'feature_importance.csv')
    imp_df.to_csv(imp_path, index=False)
    logger.info("Saved feature importances to %s", imp_path)

    return imp_df, all_feature_names, rules_text


def generate_publication_figures(df: pd.DataFrame, models: dict, X_test: pd.DataFrame,
                                 y_test: pd.Series, imp_df: pd.DataFrame,
                                 all_feature_names: list, figures_dir: str):
    """
    Generates publication-quality figures formatted for IEEE/Springer research papers.
    """
    logger.info("Step 8: Generating publication-quality figures in %s...", figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style='whitegrid', font='DejaVu Sans')

    # Figure 1: Target Class Distribution & Imbalance
    plt.figure(figsize=(7, 4.5))
    counts = df['renewed'].value_counts().sort_index()
    labels = ['Did Not Renew (0)', 'Renewed (1)']
    colors = ['#d95f02', '#1b9e77']
    bars = plt.bar(labels, counts.values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width() / 2., h + 60, f'{h:,} ({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.title('Target Variable Distribution: Subscription Renewal vs Non-Renewal', pad=15)
    plt.ylabel('Number of Subscriptions')
    plt.ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, 'fig1_class_distribution.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Key EDA Distributions by Renewal Status
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    eda_specs = [
        ('seats', 'Licensed Seats Count', axes[0, 0]),
        ('mrr_amount', 'Monthly Recurring Revenue ($MRR)', axes[0, 1]),
        ('total_usage_count', 'Total Feature Usage Count (In-Window)', axes[1, 0]),
        ('mean_satisfaction_score', 'Mean Support Satisfaction Score', axes[1, 1])
    ]
    for col, title, ax in eda_specs:
        sns.boxplot(
            data=df, x='renewed', y=col, hue='renewed', ax=ax,
            palette=['#d95f02', '#1b9e77'], width=0.45, legend=False,
            boxprops=dict(edgecolor='black', linewidth=1.1)
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Did Not Renew (0)', 'Renewed (1)'])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
    plt.suptitle('Exploratory Data Analysis: Key Predictors Across Renewal Outcomes', fontsize=13, y=0.99)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, 'fig2_eda_features.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # Figure 3: Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cm_models = ['Decision Tree', 'Logistic Regression', 'Random Forest']
    for idx, mname in enumerate(cm_models):
        m = models[mname]
        preds = m.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
            cbar=False, annot_kws={'size': 12, 'weight': 'bold'},
            xticklabels=['Pred 0 (Churn)', 'Pred 1 (Renew)'],
            yticklabels=['Actual 0', 'Actual 1'],
            linewidths=1, linecolor='gray'
        )
        axes[idx].set_title(f'{mname}\nConfusion Matrix', fontsize=12, pad=10)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, 'fig3_confusion_matrices.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # Figure 4: ROC Curves Comparison
    plt.figure(figsize=(7.5, 6))
    palette_models = {'Decision Tree': '#7570b3', 'Logistic Regression': '#e7298a', 'Random Forest': '#386cb0'}
    for mname in ['Decision Tree', 'Logistic Regression', 'Random Forest']:
        m = models[mname]
        probs = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f'{mname} (AUC = {auc:.3f})', color=palette_models[mname], linewidth=2.2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance Baseline (AUC = 0.500)', linewidth=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', pad=15)
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, 'fig4_roc_curves.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # Figure 5: Visual Representation of the Decision Tree
    plt.figure(figsize=(18, 9))
    dt = models['Decision Tree'].named_steps['dt']
    plot_tree(
        dt,
        max_depth=3,  # Top levels for visual legibility
        feature_names=all_feature_names,
        class_names=['Not Renewed (0)', 'Renewed (1)'],
        filled=True,
        rounded=True,
        fontsize=9,
        precision=2
    )
    plt.title('Pruned Decision Tree Structure (Top 3 Levels for Explainability)', fontsize=14, pad=12)
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, 'fig5_decision_tree_visualization.png')
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    # Figure 6: Feature Importances
    plt.figure(figsize=(9, 5.5))
    top_imp = imp_df.head(12).sort_values('importance', ascending=True)
    y_pos = np.arange(len(top_imp))
    plt.barh(y_pos, top_imp['importance'], color='#2b83ba', edgecolor='black', linewidth=1)
    plt.yticks(y_pos, top_imp['feature'])
    plt.xlabel('Gini Importance (Mean Decrease in Impurity)')
    plt.title('Top 12 Most Influential Features (Decision Tree)', pad=12)
    for i, v in enumerate(top_imp['importance']):
        plt.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
    plt.xlim(0, max(top_imp['importance']) * 1.15)
    plt.tight_layout()
    fig6_path = os.path.join(figures_dir, 'fig6_feature_importance.png')
    plt.savefig(fig6_path, dpi=300)
    plt.close()

    # Sync figures to paper/figures directory if it exists
    paper_fig_dir = os.path.join(os.path.dirname(figures_dir), 'paper', 'figures')
    if os.path.exists(paper_fig_dir):
        for f in glob.glob(os.path.join(figures_dir, '*.png')):
            dst = os.path.join(paper_fig_dir, os.path.basename(f))
            with open(f, 'rb') as src_f, open(dst, 'wb') as dst_f:
                dst_f.write(src_f.read())
        logger.info("Synced publication figures to %s", paper_fig_dir)

    logger.info("All 6 publication figures generated successfully.")


def run_pipeline(project_root: str = '.'):
    """
    Main orchestration routine for the SaaS Renewal Prediction project.
    """
    raw_dir = os.path.join(project_root, 'data', 'raw')
    proc_path = os.path.join(project_root, 'data', 'processed', 'saas_processed.csv')
    outputs_dir = os.path.join(project_root, 'outputs')
    results_dir = os.path.join(project_root, 'results')
    figures_dir = os.path.join(project_root, 'figures')

    for d in [raw_dir, os.path.dirname(proc_path), outputs_dir, results_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Load raw data dynamically
    dfs = load_and_validate_raw_data(raw_dir)

    # 2. Process and engineer features with valid observation window
    df = process_and_engineer_features(dfs, proc_path)

    # 3. Train/Test split (Customer-level GroupShuffleSplit on account_id)
    train_df, test_df = split_data_customer_level(df, test_size=0.2, random_state=42)

    # 4. Feature lists & preprocessing pipeline
    num_features, cat_features = get_feature_lists()
    preprocessor = build_preprocessing_pipeline(num_features, cat_features)

    X_train = train_df[num_features + cat_features]
    y_train = train_df['renewed']
    groups_train = train_df['account_id']

    X_test = test_df[num_features + cat_features]
    y_test = test_df['renewed']

    # 5. Train & tune Decision Tree (Primary Model) with StratifiedGroupKFold
    best_dt_pipe, best_dt_params, best_dt_cv_score = tune_decision_tree_group_aware(
        X_train, y_train, groups_train, preprocessor, random_state=42
    )

    # 6. Train comparative models with StratifiedGroupKFold
    comparative_models = train_comparative_models_group_aware(
        X_train, y_train, groups_train, preprocessor, random_state=42
    )

    all_models = {'Decision Tree': best_dt_pipe}
    all_models.update(comparative_models)

    # 7. Evaluate on untouched test set
    metrics_summary, predictions_df = evaluate_all_models(all_models, X_test, y_test)

    # Include test metadata in predictions
    predictions_df['subscription_id'] = test_df['subscription_id'].values
    predictions_df['account_id'] = test_df['account_id'].values
    preds_csv_path = os.path.join(results_dir, 'test_predictions.csv')
    predictions_df.to_csv(preds_csv_path, index=False)
    logger.info("Saved test set predictions to %s", preds_csv_path)

    # Save metrics JSON
    metrics_json_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info("Saved performance metrics to %s", metrics_json_path)

    # 8. Extract rules and importance
    imp_df, all_feat_names, rules_txt = extract_tree_rules_and_importance(
        best_dt_pipe, num_features, cat_features, outputs_dir
    )

    # Save model artifact
    model_save_path = os.path.join(outputs_dir, 'decision_tree_model.joblib')
    joblib.dump(best_dt_pipe, model_save_path)
    logger.info("Saved serialized Decision Tree pipeline to %s", model_save_path)

    # Save experiment configuration
    exp_config = {
        'timestamp': datetime.now().isoformat(),
        'dataset_source': 'RavenStack Synthetic SaaS Dataset (Rivalytics)',
        'target_variable': 'renewed (1: Renewed/Retained, 0: Did Not Renew/Churned)',
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'train_accounts': int(train_df['account_id'].nunique()),
        'test_accounts': int(test_df['account_id'].nunique()),
        'split_strategy': 'GroupShuffleSplit on account_id (80/20 customer isolation)',
        'cv_strategy': 'StratifiedGroupKFold on account_id (5 folds)',
        'best_dt_hyperparameters': best_dt_params,
        'best_dt_group_cv_roc_auc': round(float(best_dt_cv_score), 4),
        'numerical_features': num_features,
        'categorical_features': cat_features
    }
    config_path = os.path.join(results_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=4)
    logger.info("Saved experiment configuration to %s", config_path)

    # 9. Generate publication figures
    generate_publication_figures(
        df, all_models, X_test, y_test, imp_df, all_feat_names, figures_dir
    )

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("Decision Tree Test Accuracy: %.4f | ROC-AUC: %.4f | F1-Macro: %.4f",
                metrics_summary['Decision Tree']['accuracy'],
                metrics_summary['Decision Tree']['roc_auc'],
                metrics_summary['Decision Tree']['f1_macro'])
    logger.info("=" * 70)

    return metrics_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SaaS Subscription Renewal Prediction Pipeline")
    parser.add_argument('--root', type=str, default='.', help="Project root directory")
    args = parser.parse_args()
    run_pipeline(args.root)
