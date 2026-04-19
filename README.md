# CoAct: Co-Active LLM Preference Learning with Human-AI Synergy

Implementation for the **ACL 2026** paper
**"CoAct: Co-Active LLM Preference Learning with Human-AI Synergy"**.

## Abstract

Learning from preference-based feedback has become an effective approach for
aligning LLMs across diverse tasks. However, high-quality human-annotated
preference data remains expensive and scarce. Existing methods address this
challenge through either self-rewarding, which scales by using purely
AI-generated labels but risks unreliability, or active learning, which ensures
quality through oracle annotation but cannot fully leverage unlabeled data. In
this paper, we present **CoAct**, a novel framework that synergistically combines
self-rewarding and active learning through strategic human-AI collaboration.
CoAct leverages self-consistency to identify both reliable self-labeled data and
samples requiring oracle verification. Additionally, oracle feedback guides the
model to generate new instructions within its solvable capability. 


## Installation

```bash
conda env create -f environment.yml
conda activate coact

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && git checkout 5ed62a29c5bfcb0eee00bb4d920bb68ca0c4514c && pip install -e .
```

Copy `.env.example` to `.env` and fill in `HF_TOKEN`, `LLAMAFACTORY_DIR`,
`AZURE_API_KEY`, `AZURE_ENDPOINT`. Source it before running any command:

```bash
set -a; source .env; set +a
```

## Dataset preparation

Run the creator scripts to build train/dev/test splits and per-iteration
active-learning splits into `${LLAMAFACTORY_DIR}/data/`, then register the
resulting dataset names in `${LLAMAFACTORY_DIR}/data/dataset_info.json`:

```bash
python data_prep/create_gsm8k.py
python data_prep/create_math.py
python data_prep/create_webinstruct.py
```

## Running

```bash
# iteration 0 with the default Llama-3-8B backbone
bash run_iteration.sh 0 --dataset gsm8k

# subsequent iterations resume from the previous LoRA adapter
bash run_iteration.sh 1 --dataset gsm8k

# switch backbones via env vars 
BASE_MODEL=Qwen/Qwen3-4B MODEL_SIZE=qwen3-4b bash run_iteration.sh 0 --dataset math
```

## Repository layout

```
CoAct/
├── run_iteration.sh      # main 
├── scripts/              # generation, self-consistency, KNN, oracle, ICL, DPO
├── configs/              # DPO training configs
├── data_prep/            # dataset creator scripts
└── README.md
```

## Citation

```bibtex
@inproceedings{coact_2026,
  title     = {CoAct: Co-Active LLM Preference Learning with Human-AI Synergy},
  author    = {Xu, Ruiyao and Parmar, Mihir and Yang, Tiankai and Hu, Zhengyu and Zhao, Yue and Ding, Kaize},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```


