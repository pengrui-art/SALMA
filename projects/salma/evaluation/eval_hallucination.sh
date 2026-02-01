#!/bin/bash
# Hallucination Evaluation Script for Sa2VA-Struct
# This script runs DCR, SRA, and ADS evaluation on RefCOCOg

# =============================================================================
# Configuration
# =============================================================================

# Model paths (modify these)
BASELINE_MODEL="/path/to/sa2va-1b-baseline"
OURS_MODEL="/path/to/sa2va-1b-struct"

# Data paths (set via environment or modify here)
export COCO2014_TRAIN="${COCO2014_TRAIN:-/data/coco/train2014}"
export REFCOCO_DATA="${REFCOCO_DATA:-/data/refcoco}"

# Output directory
WORK_DIR="./work_dirs/hallucination_eval"

# =============================================================================
# Run Evaluation
# =============================================================================

echo "=============================================="
echo "Hallucination Evaluation for Sa2VA-Struct"
echo "=============================================="

# Create output directory
mkdir -p ${WORK_DIR}

# Evaluate Baseline
echo ""
echo "[1/2] Evaluating Baseline Model..."
echo "----------------------------------------------"
python hallucination_eval.py ${BASELINE_MODEL} \
    --metric all \
    --dataset refcocog \
    --split val \
    --work-dir ${WORK_DIR}/baseline

# Evaluate Ours
echo ""
echo "[2/2] Evaluating Our Model (Sa2VA-Struct)..."
echo "----------------------------------------------"
python hallucination_eval.py ${OURS_MODEL} \
    --metric all \
    --dataset refcocog \
    --split val \
    --work-dir ${WORK_DIR}/ours

# =============================================================================
# Generate Comparison Report
# =============================================================================

echo ""
echo "=============================================="
echo "Generating Comparison Report..."
echo "=============================================="

python -c "
import json
import os

work_dir = '${WORK_DIR}'

# Load results
baseline_path = os.path.join(work_dir, 'baseline', 'hallucination_eval_refcocog_val.json')
ours_path = os.path.join(work_dir, 'ours', 'hallucination_eval_refcocog_val.json')

with open(baseline_path) as f:
    baseline = json.load(f)
with open(ours_path) as f:
    ours = json.load(f)

# Generate comparison table
print()
print('=' * 70)
print('HALLUCINATION METRICS COMPARISON')
print('=' * 70)
print(f'{\"Metric\":<35} {\"Baseline\":<12} {\"Ours\":<12} {\"Delta\":<12}')
print('-' * 70)

# DCR
if 'dcr' in baseline and 'dcr' in ours:
    b_dcr = baseline['dcr']['dcr']
    o_dcr = ours['dcr']['dcr']
    delta = o_dcr - b_dcr
    print(f'{\"DCR (Distractor Confusion) ↓\":<35} {b_dcr:>10.2f}% {o_dcr:>10.2f}% {delta:>+10.2f}%')

# SRA Overall
if 'sra' in baseline and 'sra' in ours:
    b_sra = baseline['sra']['overall_sra']
    o_sra = ours['sra']['overall_sra']
    delta = o_sra - b_sra
    print(f'{\"SRA (Overall) ↑\":<35} {b_sra:>10.2f}% {o_sra:>10.2f}% {delta:>+10.2f}%')
    
    # SRA by category
    for cat in ['positional', 'relational', 'ordinal', 'size_comparative']:
        key = f'sra_{cat}'
        if key in baseline['sra'] and key in ours['sra']:
            b_val = baseline['sra'][key]
            o_val = ours['sra'][key]
            delta = o_val - b_val
            print(f'{f\"  - {cat.capitalize()}\":<35} {b_val:>10.2f}% {o_val:>10.2f}% {delta:>+10.2f}%')

# ADS
if 'ads' in baseline and 'ads' in ours:
    b_ads = baseline['ads']['proxy_ads']
    o_ads = ours['ads']['proxy_ads']
    delta = o_ads - b_ads
    print(f'{\"ADS (Attention Drift) ↓\":<35} {b_ads:>10.4f}  {o_ads:>10.4f}  {delta:>+10.4f}')

print('=' * 70)

# Save to file
report = {
    'baseline': baseline,
    'ours': ours,
}
report_path = os.path.join(work_dir, 'comparison_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f'Report saved to {report_path}')
"

echo ""
echo "Evaluation complete!"
