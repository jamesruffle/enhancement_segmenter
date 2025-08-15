#!/usr/bin/env python3
"""
Model Performance Comparison Analysis
Compares performance metrics between two nnUNet models:
- Dataset002_enhance_and_abnormality
- Dataset003_enhance_and_abnormality_batchconfig
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Performance metrics from the models
model_results = {
    "Dataset002_enhance_and_abnormality": {
        "overall": {
            "Dice": 0.776692894521691,
            "IoU": 0.7091560916341774
        },
        "class_metrics": {
            "Class_1": {"Dice": 0.9867030431441742, "IoU": 0.9745627208620088},
            "Class_2": {"Dice": 0.810775627322343, "IoU": 0.7280622839928657},
            "Class_3": {"Dice": 0.5326000130985554, "IoU": 0.4248432700476577}
        }
    },
    "Dataset003_enhance_and_abnormality_batchconfig": {
        "overall": {
            "Dice": 0.7883432787028758,
            "IoU": 0.7232662996092536
        },
        "class_metrics": {
            "Class_1": {"Dice": 0.9869062035305266, "IoU": 0.974971800867428},
            "Class_2": {"Dice": 0.8208156992734613, "IoU": 0.7419296907885664},
            "Class_3": {"Dice": 0.5573079333046396, "IoU": 0.4528974071717664}
        }
    }
}

# Calculate improvements
def calculate_improvements():
    improvements = {}
    base = model_results["Dataset002_enhance_and_abnormality"]
    batch = model_results["Dataset003_enhance_and_abnormality_batchconfig"]
    
    # Overall improvements
    improvements["overall"] = {
        "Dice": (batch["overall"]["Dice"] - base["overall"]["Dice"]) * 100,
        "IoU": (batch["overall"]["IoU"] - base["overall"]["IoU"]) * 100
    }
    
    # Class-wise improvements
    improvements["classes"] = {}
    for class_name in base["class_metrics"]:
        improvements["classes"][class_name] = {
            "Dice": (batch["class_metrics"][class_name]["Dice"] - 
                    base["class_metrics"][class_name]["Dice"]) * 100,
            "IoU": (batch["class_metrics"][class_name]["IoU"] - 
                   base["class_metrics"][class_name]["IoU"]) * 100
        }
    
    return improvements

# Save evaluation results to file
def save_evaluation_results():
    improvements = calculate_improvements()
    
    evaluation_text = f"""Model Performance Comparison Report
=====================================

Model 1: Dataset002_enhance_and_abnormality
Model 2: Dataset003_enhance_and_abnormality_batchconfig (with batch configuration)

OVERALL PERFORMANCE
-------------------
Model 1 - Dice: {model_results['Dataset002_enhance_and_abnormality']['overall']['Dice']:.4f} ({model_results['Dataset002_enhance_and_abnormality']['overall']['Dice']*100:.2f}%)
Model 2 - Dice: {model_results['Dataset003_enhance_and_abnormality_batchconfig']['overall']['Dice']:.4f} ({model_results['Dataset003_enhance_and_abnormality_batchconfig']['overall']['Dice']*100:.2f}%)
Improvement: +{improvements['overall']['Dice']:.2f}%

Model 1 - IoU: {model_results['Dataset002_enhance_and_abnormality']['overall']['IoU']:.4f} ({model_results['Dataset002_enhance_and_abnormality']['overall']['IoU']*100:.2f}%)
Model 2 - IoU: {model_results['Dataset003_enhance_and_abnormality_batchconfig']['overall']['IoU']:.4f} ({model_results['Dataset003_enhance_and_abnormality_batchconfig']['overall']['IoU']*100:.2f}%)
Improvement: +{improvements['overall']['IoU']:.2f}%

CLASS-WISE PERFORMANCE
----------------------
Class 1 (Main tissue):
  Model 1 - Dice: {model_results['Dataset002_enhance_and_abnormality']['class_metrics']['Class_1']['Dice']:.4f} ({model_results['Dataset002_enhance_and_abnormality']['class_metrics']['Class_1']['Dice']*100:.2f}%)
  Model 2 - Dice: {model_results['Dataset003_enhance_and_abnormality_batchconfig']['class_metrics']['Class_1']['Dice']:.4f} ({model_results['Dataset003_enhance_and_abnormality_batchconfig']['class_metrics']['Class_1']['Dice']*100:.2f}%)
  Improvement: +{improvements['classes']['Class_1']['Dice']:.2f}%

Class 2:
  Model 1 - Dice: {model_results['Dataset002_enhance_and_abnormality']['class_metrics']['Class_2']['Dice']:.4f} ({model_results['Dataset002_enhance_and_abnormality']['class_metrics']['Class_2']['Dice']*100:.2f}%)
  Model 2 - Dice: {model_results['Dataset003_enhance_and_abnormality_batchconfig']['class_metrics']['Class_2']['Dice']:.4f} ({model_results['Dataset003_enhance_and_abnormality_batchconfig']['class_metrics']['Class_2']['Dice']*100:.2f}%)
  Improvement: +{improvements['classes']['Class_2']['Dice']:.2f}%

Class 3:
  Model 1 - Dice: {model_results['Dataset002_enhance_and_abnormality']['class_metrics']['Class_3']['Dice']:.4f} ({model_results['Dataset002_enhance_and_abnormality']['class_metrics']['Class_3']['Dice']*100:.2f}%)
  Model 2 - Dice: {model_results['Dataset003_enhance_and_abnormality_batchconfig']['class_metrics']['Class_3']['Dice']:.4f} ({model_results['Dataset003_enhance_and_abnormality_batchconfig']['class_metrics']['Class_3']['Dice']*100:.2f}%)
  Improvement: +{improvements['classes']['Class_3']['Dice']:.2f}%

CONCLUSION
----------
The model with batch configuration (Dataset003) consistently outperforms the baseline model (Dataset002) 
across all metrics and classes. The most significant improvements are observed in Class 3 (+{improvements['classes']['Class_3']['Dice']:.2f}% Dice score),
while Class 1 shows minimal but positive improvement (+{improvements['classes']['Class_1']['Dice']:.2f}% Dice score).
"""
    
    with open("/home/jruffle/Downloads/model_comparison_evaluation.txt", "w") as f:
        f.write(evaluation_text)
    
    # Also save as JSON
    json_results = {
        "models": model_results,
        "improvements": improvements,
        "winner": "Dataset003_enhance_and_abnormality_batchconfig"
    }
    
    with open("/home/jruffle/Downloads/model_comparison_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print("Evaluation results saved to:")
    print("- model_comparison_evaluation.txt")
    print("- model_comparison_results.json")

# Generate comparison figures
def generate_comparison_figures():
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Figure 1: Overall Performance Comparison
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ['Dataset002\n(baseline)', 'Dataset003\n(batch config)']
    dice_scores = [
        model_results['Dataset002_enhance_and_abnormality']['overall']['Dice'] * 100,
        model_results['Dataset003_enhance_and_abnormality_batchconfig']['overall']['Dice'] * 100
    ]
    iou_scores = [
        model_results['Dataset002_enhance_and_abnormality']['overall']['IoU'] * 100,
        model_results['Dataset003_enhance_and_abnormality_batchconfig']['overall']['IoU'] * 100
    ]
    
    # Dice comparison
    bars1 = ax1.bar(models, dice_scores, color=['#3498db', '#2ecc71'], alpha=0.8)
    ax1.set_ylabel('Dice Score (%)')
    ax1.set_title('Overall Dice Score Comparison')
    ax1.set_ylim(70, 80)
    
    # Add value labels on bars
    for bar, score in zip(bars1, dice_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%', ha='center', va='bottom')
    
    # IoU comparison
    bars2 = ax2.bar(models, iou_scores, color=['#e74c3c', '#f39c12'], alpha=0.8)
    ax2.set_ylabel('IoU Score (%)')
    ax2.set_title('Overall IoU Score Comparison')
    ax2.set_ylim(65, 75)
    
    # Add value labels on bars
    for bar, score in zip(bars2, iou_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/home/jruffle/Downloads/model_comparison_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Class-wise Performance Comparison
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    classes = ['Class 1', 'Class 2', 'Class 3']
    x = np.arange(len(classes))
    width = 0.35
    
    dice_baseline = [
        model_results['Dataset002_enhance_and_abnormality']['class_metrics'][f'Class_{i+1}']['Dice'] * 100
        for i in range(3)
    ]
    dice_batch = [
        model_results['Dataset003_enhance_and_abnormality_batchconfig']['class_metrics'][f'Class_{i+1}']['Dice'] * 100
        for i in range(3)
    ]
    
    bars1 = ax.bar(x - width/2, dice_baseline, width, label='Dataset002 (baseline)', 
                    color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, dice_batch, width, label='Dataset003 (batch config)', 
                    color='#2ecc71', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Dice Score (%)')
    ax.set_title('Class-wise Dice Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('/home/jruffle/Downloads/model_comparison_classwise.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Improvement Analysis
    fig3, ax = plt.subplots(figsize=(8, 6))
    
    improvements = calculate_improvements()
    metrics = ['Overall Dice', 'Overall IoU', 'Class 1 Dice', 'Class 2 Dice', 'Class 3 Dice']
    improvement_values = [
        improvements['overall']['Dice'],
        improvements['overall']['IoU'],
        improvements['classes']['Class_1']['Dice'],
        improvements['classes']['Class_2']['Dice'],
        improvements['classes']['Class_3']['Dice']
    ]
    
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in improvement_values]
    bars = ax.barh(metrics, improvement_values, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars, improvement_values):
        width = bar.get_width()
        label_x = width if width > 0 else width - 0.1
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2.,
               f'{value:+.2f}%', ha=ha, va='center')
    
    ax.set_xlabel('Improvement (%)')
    ax.set_title('Performance Improvements: Batch Config vs Baseline')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlim(-0.5, 3)
    
    plt.tight_layout()
    plt.savefig('/home/jruffle/Downloads/model_comparison_improvements.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Figures saved:")
    print("- model_comparison_overall.png")
    print("- model_comparison_classwise.png")
    print("- model_comparison_improvements.png")

def main():
    """Run the complete analysis"""
    print("Model Performance Comparison Analysis")
    print("=====================================\n")
    
    save_evaluation_results()
    print()
    generate_comparison_figures()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()