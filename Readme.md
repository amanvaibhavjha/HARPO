# HARPO: Hierarchical Agentic Reasoning with Preference Optimization

Official implementation of **"Optimizing Conversational Recommendation for User-Aligned Quality via Hierarchical Preference Learning"** (ACL 2025 Submission).

## Overview

HARPO is a framework for conversational recommender systems that explicitly optimizes for user-aligned recommendation quality rather than proxy metrics. The framework consists of four components:

- **CHARM** (Contrastive Hierarchical Alignment with Reward Marginalization): Multi-dimensional reward decomposition across relevance, diversity, satisfaction, and engagement
- **STAR** (Structured Tree-of-Thought Agentic Reasoning): Quality-aware tree search with learned value function
- **BRIDGE** (Bidirectional Reasoning-Informed Domain-Generalized Embeddings): Cross-domain transfer via adversarial domain adaptation
- **MAVEN** (Multi-Agent Virtual Environment for Recommendations): Multi-agent refinement through collaborative critique

## Results

### Main Results on ReDial (Table 3)

| Method | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | User Sat. | Engage. |
|--------|-----|------|------|--------|---------|-----------|---------|
| UniCRS | 4.8±0.3 | 21.2±0.5 | 40.8±0.8 | 10.1±0.3 | 13.8±0.4 | 0.51±0.02 | 0.47±0.02 |
| DCRS | 7.5±0.3 | 25.1±0.6 | 43.6±0.9 | 12.2±0.4 | 15.2±0.5 | 0.56±0.02 | 0.52±0.02 |
| GPT-4 | 4.5±0.4 | 19.4±0.8 | 40.2±1.2 | 9.6±0.5 | 13.2±0.6 | 0.55±0.03 | 0.51±0.03 |
| RecMind | 5.8±0.3 | 22.6±0.6 | 42.2±0.9 | 11.2±0.4 | 15.3±0.5 | 0.54±0.02 | 0.50±0.02 |
| **HARPO** | **9.1±0.3** | **29.8±0.7** | **50.2±1.0** | **15.6±0.5** | **21.2±0.6** | **0.68±0.02** | **0.64±0.02** |

User Sat./Engage. are normalized to [0,1] and computed via CHARM reward model. Human evaluation confirms correlation with human judgments (r=0.64-0.73).

### Ablation Study on ReDial (Table 4)

| Variant | R@10 | MRR@10 | NDCG@10 | Sat. | Eng. |
|---------|------|--------|---------|------|------|
| HARPO (Full) | 29.8 | 15.6 | 21.2 | 0.68 | 0.64 |
| w/o CHARM | 24.6 | 12.6 | 17.2 | 0.55 | 0.51 |
| w/o STAR | 27.0 | 14.0 | 19.0 | 0.63 | 0.59 |
| w/o BRIDGE | 28.4 | 14.9 | 20.2 | 0.66 | 0.62 |
| w/o MAVEN | 28.0 | 14.7 | 19.9 | 0.65 | 0.61 |
| w/o VTOs | 23.4 | 12.0 | 16.4 | 0.53 | 0.49 |
| SFT Only | 21.6 | 10.6 | 14.6 | 0.50 | 0.46 |

### Human Evaluation on ReDial (Table 8)

| Method | Rec. Quality | Exp. Quality | Overall | κ |
|--------|--------------|--------------|---------|---|
| UniCRS | 3.18±0.12 | 2.86±0.14 | 3.04±0.11 | 0.73 |
| GPT-4 | 3.48±0.11 | 3.42±0.13 | 3.46±0.10 | 0.74 |
| **HARPO** | **4.08±0.10** | **3.92±0.12** | **4.01±0.09** | 0.78 |

Scores on 1-5 scale. κ = Fleiss' kappa. 200 samples rated by 3 annotators.

## Repository Structure

```
harpo/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   └── data_processing.py
├── scripts/
│   ├── convert_redial.py
│   ├── convert_inspired.py
│   ├── train.py
│   ├── evaluate.py
│   └── chat.py
├── configs/
│   ├── accelerate_config.yaml
│   └── training_config.yaml
├── data/
│   └── (see Data Preparation)
└── tests/
    └── test_model.py
```

## Installation

```bash
git clone https://github.com/anonymous/harpo.git
cd harpo

conda create -n harpo python=3.10
conda activate harpo

pip install -r requirements.txt

# Optional: Flash Attention 2 (requires CUDA)
pip install flash-attn --no-build-isolation
```

## Data Preparation

Download and process datasets:

```bash
# ReDial
python scripts/convert_redial.py --output data/redial

# INSPIRED  
python scripts/convert_inspired.py --output data/inspired
```

## Training

### Single GPU

```bash
python scripts/train.py \
    --sft data/redial/sft_data.json \
    --pref data/redial/preference_data.json \
    --output outputs/redial
```

### Multi-GPU (Recommended)

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train.py \
    --sft data/redial/sft_data.json \
    --pref data/redial/preference_data.json \
    --output outputs/redial
```

## Configuration

### Model

- Backbone: `DeepSeek-R1-Distill-Qwen-7B`
- Hidden size: 3584
- Number of VTOs: 21
- LoRA rank: 16, alpha: 32

### Training Hyperparameters

| Stage | Learning Rate | Epochs |
|-------|---------------|--------|
| SFT | 5e-5 | 3 |
| CHARM | 2e-5 | 2 |
| STAR | 1e-5 | 2 |
| MAVEN | 2e-6 | 1 |

- Batch size: 4 per GPU
- Gradient accumulation: 4 steps
- Effective batch size: 32 (with 2 GPUs)
- Max sequence length: 512

### STAR Configuration

- Beam width: 3
- Max depth: 3
- Backtrack threshold: 0.3

### CHARM Configuration

- Reward dimensions: 4 (relevance, diversity, satisfaction, engagement)
- β (preference strength): 0.5

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/redial/checkpoints/final \
    --test-data data/redial/test.json \
    --output results/redial
```

## Computational Requirements

Training on 2× NVIDIA A100 80GB:

| Stage | Time | Memory |
|-------|------|--------|
| SFT | 52 min | 68 GB |
| CHARM | 42 min | 62 GB |
| STAR | 46 min | 58 GB |
| MAVEN | 36 min | 56 GB |
| **Total** | **~2.9 hours** | 68 GB |

Inference latency per turn:
- Full HARPO: 298 ms
- Without STAR: 88 ms

## Virtual Tool Operations (VTOs)

HARPO uses 21 domain-agnostic VTOs organized into 7 categories:

| Category | Operations |
|----------|------------|
| Extraction | `analyze_sentiment`, `extract_context`, `extract_entities` |
| User Modeling | `retrieve_preferences`, `identify_constraints`, `model_user_state` |
| Retrieval | `search_candidates`, `filter_results`, `match_attributes` |
| Ranking | `rank_options`, `compare_options`, `select_best` |
| Reasoning | `query_knowledge`, `reason_over_graph`, `infer_implicit` |
| Interaction | `explain_choice`, `refine_query`, `handle_rejection` |
| Memory | `track_history`, `update_beliefs`, `recall_context` |

## License

MIT License

## Acknowledgments

- DeepSeek for the R1-Distill-Qwen-7B model
- ReDial, INSPIRED, and MUSE dataset creators
- HuggingFace Transformers library
