# HARPO: Hierarchical Agentic Reasoning with Preference Optimization for Conversational Recommendation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **HARPO: Optimizing Conversational Recommendation for User-Aligned Quality via Hierarchical Preference Learning** (ACL 2025 Submission).

## 🎯 Overview

HARPO is a novel framework for conversational recommender systems that explicitly optimizes for **user-aligned recommendation quality** rather than proxy metrics. Our key contributions:

1. **CHARM** (Contrastive Hierarchical Alignment with Reward Marginalization): Multi-dimensional reward decomposition that captures relevance, diversity, satisfaction, and engagement
2. **STAR** (Structured Tree-of-thought Agentic Reasoning): Quality-aware tree search with learned value function for recommendation reasoning  
3. **BRIDGE** (Bidirectional Reasoning-Informed Domain-Generalized Embeddings): Cross-domain transfer via adversarial domain adaptation
4. **MAVEN** (Multi-Agent Virtual Environment for Recommendations): Multi-agent refinement through collaborative critique

## 📊 Results

Performance on ReDial, INSPIRED, and MUSE datasets:

| Method | R@1 | R@10 | R@50 | MRR | User Sat. | Engage. |
|--------|-----|------|------|-----|-----------|---------|
| UniCRS | 7.8 | 21.2 | 38.4 | 0.127 | - | - |
| DCRS | 9.4 | 25.1 | 42.7 | 0.152 | - | - |
| **HARPO** | **12.1** | **29.8** | **51.2** | **0.189** | **0.847** | **0.823** |

*Note: User Sat./Engage. computed via CHARM reward model; human evaluation (Table 8) confirms correlation with human judgments (r=0.64-0.73).*

## 📁 Repository Structure

```
harpo/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
│
├── src/                         # Core source code
│   ├── __init__.py
│   ├── config.py               # Configuration & data structures
│   ├── model.py                # HARPO model (BRIDGE, STAR, CHARM, MAVEN)
│   ├── training.py             # 4-stage training pipeline
│   ├── evaluation.py           # Comprehensive evaluation metrics
│   └── data_generation.py      # Data processing utilities
│
├── scripts/                     # Executable scripts
│   ├── convert_redial.py       # ReDial dataset converter
│   ├── convert_inspired.py     # INSPIRED dataset converter
│   ├── convert_muse.py         # MUSE dataset converter (coming soon)
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Standalone evaluation
│   └── chat.py                 # Interactive demo
│
├── configs/                     # Configuration files
│   ├── accelerate_config.yaml  # Multi-GPU settings
│   └── training_config.yaml    # Training hyperparameters
│
├── data/                        # Data directory (create after download)
│   ├── redial/                 # ReDial dataset
│   │   ├── sft_data.json
│   │   ├── preference_data.json
│   │   └── movie_list.json
│   ├── inspired/               # INSPIRED dataset
│   │   ├── sft_data.json
│   │   └── preference_data.json
│   └── muse/                   # MUSE dataset
│       ├── sft_data.json
│       └── preference_data.json
│
├── outputs/                     # Training outputs (generated)
│   └── checkpoints/
│
└── tests/                       # Unit tests
    └── test_model.py
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/anonymous/harpo.git
cd harpo

# Create environment
conda create -n harpo python=3.10
conda activate harpo

# Install dependencies
pip install -r requirements.txt

# Optional: Flash Attention 2 for 2-3x speedup (requires CUDA)
pip install flash-attn --no-build-isolation
```

### 2. Data Preparation

**Option A: Download pre-processed data (Recommended)**
```bash
# Download from anonymous link (will be provided)
wget https://anonymous-link/harpo_data.tar.gz
tar -xzf harpo_data.tar.gz -C data/
```

**Option B: Process from raw datasets**
```bash
# ReDial (Movies - conversational)
python scripts/convert_redial.py \
    --output data/redial \
    --use-llm \
    --api-key YOUR_OPENAI_KEY

# INSPIRED (Movies - sociable)
python scripts/convert_inspired.py \
    --output data/inspired \
    --use-llm \
    --api-key YOUR_OPENAI_KEY
```

### 3. Training

**Single GPU:**
```bash
python scripts/train.py \
    --sft data/redial/sft_data.json \
    --pref data/redial/preference_data.json \
    --output outputs/redial
```

**Multi-GPU with Accelerate (Recommended):**
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train.py \
    --sft data/redial/sft_data.json \
    --pref data/redial/preference_data.json \
    --output outputs/redial
```

**Resume from checkpoint:**
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train.py \
    --sft data/redial/sft_data.json \
    --pref data/redial/preference_data.json \
    --output outputs/redial \
    --resume sft_final \
    --skip-stages sft
```

### 4. Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/redial/checkpoints/final \
    --test-data data/redial/test_sft.json \
    --output results/redial
```

### 5. Interactive Demo

```bash
python scripts/chat.py --checkpoint outputs/redial/checkpoints/final
```

## 📋 Training Pipeline

HARPO uses a 4-stage curriculum training:

| Stage | Component | Objective | Epochs |
|-------|-----------|-----------|--------|
| 1 | SFT | Supervised fine-tuning with VTO prediction | 3 |
| 2 | CHARM | Hierarchical preference optimization | 2 |
| 3 | STAR | Tree-of-thought reasoning training | 2 |
| 4 | MAVEN | Multi-agent self-play refinement | 1 |

**Training Time Estimates (2x A100 80GB):**
- ReDial (~10K examples): ~2-3 hours
- INSPIRED (~5K examples): ~1-2 hours
- Full pipeline with evaluation: ~4 hours

## 🔧 Configuration

### Model Configuration (src/config.py)

```python
@dataclass
class ModelConfig:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    hidden_size: int = 3584
    num_vtos: int = 24
    use_flash_attention: bool = True  # Requires flash-attn package
```

### Training Configuration

```python
@dataclass  
class TrainingConfig:
    # Stage 1: SFT
    sft_epochs: int = 3
    sft_lr: float = 2e-5
    
    # Stage 2: CHARM
    charm_epochs: int = 2
    charm_lr: float = 1e-5
    
    # Stage 3: STAR  
    star_epochs: int = 2
    star_lr: float = 5e-6
    
    # Stage 4: MAVEN
    maven_epochs: int = 1
    maven_lr: float = 2e-6
    
    # General
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
```

### Multi-GPU Configuration (configs/accelerate_config.yaml)

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 2  # Number of GPUs
```

## 📊 Evaluation Metrics

### Primary Metrics (User-Aligned Quality)
- **User Satisfaction**: CHARM-predicted satisfaction (validated r=0.73 with human judgments)
- **Engagement Score**: CHARM-predicted engagement

### Ranking Metrics (Standard)
- Recall@K (K=1,5,10,20,50)
- MRR, MRR@K
- NDCG@K
- Hit Rate@K

### Generation Metrics
- BLEU-1/2/3/4
- ROUGE-L
- Distinct-1/2

### Novel Metrics
- Reasoning Depth (STAR)
- Thought Quality (STAR)
- Agent Agreement Rate (MAVEN)

## 📝 Data Format

### SFT Data (sft_data.json)
```json
[
  {
    "input": "<|domain:movies|>\n\nUser: I'm looking for a comedy movie",
    "output": "<|think|>extract_context, retrieve_preferences, search_candidates<|/think|>\nI'd recommend \"The Grand Budapest Hotel\" - it's a visually stunning comedy...",
    "vtos": ["extract_context", "retrieve_preferences", "search_candidates"],
    "domain": "movies",
    "ground_truth_item": "The Grand Budapest Hotel",
    "conversation_id": "conv_001"
  }
]
```

### Preference Data (preference_data.json)
```json
[
  {
    "context": "<|domain:movies|>\n\nUser: I want something funny",
    "chosen": "<|think|>extract_context, search_candidates<|/think|>\nBased on your preference for comedy...",
    "rejected": "I'm not sure what to recommend. There are many options.",
    "chosen_vtos": ["extract_context", "search_candidates"],
    "hierarchical_rewards": {
      "relevance": {"chosen": 0.9, "rejected": 0.3},
      "diversity": {"chosen": 0.8, "rejected": 0.5},
      "user_satisfaction": {"chosen": 0.85, "rejected": 0.4},
      "engagement": {"chosen": 0.9, "rejected": 0.6}
    },
    "reward_margin": 0.5
  }
]
```

## 🔬 Ablation Study

Component contributions on ReDial (Recall@10):

| Configuration | R@10 | Δ |
|--------------|------|---|
| HARPO (Full) | 29.8 | - |
| w/o CHARM | 24.6 | -5.2 |
| w/o STAR | 27.0 | -2.8 |
| w/o BRIDGE | 28.4 | -1.4 |
| w/o MAVEN | 28.0 | -1.8 |
| w/o VTOs | 23.4 | -6.4 |
| SFT-only | 21.6 | -8.2 |

*Note: Component ablations retrain without the specified module; contributions are non-additive due to interactions.*

## 🤝 Human Evaluation

We conducted human evaluation with 3 expert annotators on 200 test samples per dataset:

| System | Rec. Quality | Exp. Quality | Overall | κ |
|--------|-------------|--------------|---------|---|
| GPT-4 | 3.42 | 3.51 | 3.45 | 0.74 |
| UniCRS | 2.87 | 2.93 | 2.89 | 0.72 |
| **HARPO** | **3.97** | **4.01** | **4.00** | 0.76 |

*Scores on 1-5 scale; κ = Fleiss' kappa for inter-annotator agreement*

## ⚠️ Limitations

1. **Computational Cost**: Full training requires significant GPU resources (2x A100 recommended)
2. **Data Contamination Risk**: ReDial (2018) and INSPIRED (2020) may have contamination with LLM training data
3. **Evaluation Circularity**: CHARM-based metrics should be interpreted with human evaluation as validation
4. **Missing Baselines**: Comparison focuses on methods through 2023; evaluation against 2024 LLM-based CRS methods would strengthen positioning

## 📚 Citation

```bibtex
@inproceedings{anonymous2025harpo,
  title={HARPO: Optimizing Conversational Recommendation for User-Aligned Quality via Hierarchical Preference Learning},
  author={Anonymous},
  booktitle={Proceedings of ACL 2025},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DeepSeek for the R1-Distill-Qwen-7B model
- ReDial, INSPIRED, and MUSE dataset creators
- HuggingFace for the Transformers library