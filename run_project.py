#!/usr/bin/env python3
"""
Entry point script for reproducing the SaaS Subscription Renewal Prediction experiment.
Executes data validation, feature engineering, model training, evaluation, rule extraction,
and publication figure generation.
"""

import sys
import os

# Ensure current directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.saas_renewal_prediction import run_pipeline

if __name__ == '__main__':
    print("=" * 80)
    print("STARTING SAAS SUBSCRIPTION RENEWAL PREDICTION PIPELINE")
    print("Author: Harshita Singh & AI Research Assistant")
    print("Algorithm: Decision Tree Classifier (with Logistic Regression & Random Forest)")
    print("=" * 80)
    
    project_root = os.path.abspath(os.path.dirname(__file__))
    metrics = run_pipeline(project_root)
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    for model_name, m in metrics.items():
        print(f"\nModel: {model_name}")
        print(f"  Accuracy:          {m['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {m['balanced_accuracy']:.4f}")
        print(f"  ROC-AUC:           {m['roc_auc']:.4f}")
        print(f"  F1-Score (Macro):  {m['f1_macro']:.4f}")
        print(f"  Renewal (Class 1) - Precision: {m['class_1_renewal']['precision']:.4f}, Recall: {m['class_1_renewal']['recall']:.4f}, F1: {m['class_1_renewal']['f1']:.4f}")
        print(f"  Churn (Class 0)   - Precision: {m['class_0_churn']['precision']:.4f}, Recall: {m['class_0_churn']['recall']:.4f}, F1: {m['class_0_churn']['f1']:.4f}")
        print(f"  Confusion Matrix:  {m['confusion_matrix']}")
    print("\nAll outputs and figures generated in 'outputs/', 'results/', and 'figures/'.")
