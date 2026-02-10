This is a reproduction code of previous work ***LLM-based Conversational AI Therapist for Daily Functioning Screening and Psychotherapeutic Intervention via Everyday Smart Devices*** at <https://doi.org/10.1145/3712299>

```bibtex
@article{10.1145/3712299,
author = {Nie, Jingping and Shao, Hanya (Vera) and Fan, Yuang and Shao, Qijia and You, Haoxuan and Preindl, Matthias and Jiang, Xiaofan},
title = {LLM-based Conversational AI Therapist for Daily Functioning Screening and Psychotherapeutic Intervention via Everyday Smart Devices},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3712299},
doi = {10.1145/3712299},
abstract = {Despite the global mental health crisis, access to screenings, professionals, and treatments remains high. In collaboration with licensed psychotherapists, we propose a Conversational AI Therapist with psychotherapeutic Interventions (CaiTI), a platform that leverages large language models (LLM)s and smart devices to enable better mental health self-care. CaiTI can screen the day-to-day functioning using natural and psychotherapeutic conversations. CaiTI leverages reinforcement learning to provide personalized conversation flow. CaiTI can accurately understand and interpret user responses. When the user needs further attention during the conversation, CaiTI can provide conversational psychotherapeutic interventions, including cognitive behavioral therapy (CBT) and motivational interviewing (MI). Leveraging the datasets prepared by the licensed psychotherapists, we experiment and microbenchmark various LLMs’ performance in tasks along CaiTI's conversation flow and discuss their strengths and weaknesses. With the psychotherapists, we implement CaiTI and conduct 14-day and 24-week studies. The study results, validated by therapists, demonstrate that CaiTI can converse with users naturally, accurately understand and interpret user responses, and provide psychotherapeutic interventions appropriately and effectively. We showcase the potential of CaiTI LLMs to assist the mental therapy diagnosis and treatment and improve day-to-day functioning screening and precautionary psychotherapeutic intervention systems.},
note = {Just Accepted},
journal = {ACM Trans. Comput. Healthcare},
month = jan,
keywords = {Large Language Models (LLMs), Foundation Models, AI therapist, Psychotherapy, Everyday Smart Devices, Cognitive Behavioral Therapy, Motivational Interviewing}
}
```

## Overview

This repository reproduces core components of a conversational AI therapist pipeline that performs day-to-day functioning screening and delivers psychotherapeutic interventions. The system integrates:

- Personalized dialogue flow via reinforcement learning (RL) Q-tables
- Response analysis (segmentation, dimension mapping, scoring)
- Reflection & validation (RV) logic when critical signals arise
- Cognitive Behavioral Therapy (CBT) intervention modules
- Simple application server and CLI-like runner

The project aims to facilitate research reproducibility, ablations, and integration tests around the end-to-end pipeline.

## Project Structure

```text
root/
├── LLM_therapist_Application.py                # Main application entry
├── LLM_therapist_Application_server.py         # Main application background entry
├── config.yaml                                 # Config for hyper-parameters
├── environment.yml                             # Conda env (baseline)
├── environment_upgradable.yml                  # Conda env (upgradable path)
├── src/
│   ├── CBT.py                                  # CBT engine
│   ├── handler_rl.py                           # RL handler / policy
│   ├── questioner.py                           # Conversation question logic
│   ├── reflection_validation.py                # RV logic
│   ├── response_analyzer.py                    # Response analysis pipeline
│   └── utils/                                  # IO, logging, config helpers
├── data/                                       # Data & results (do not modify)
│   ├── q_tables/
│   └── results/
```

## Requirements & Setup

Recommend using Conda. The baseline environment is in `environment.yml`.

```bash
# Create and activate env
conda env create -f environment.yml
conda activate llm_therapist

# (Optional) Upgradable pacakges for more compatibility
conda env update -f environment_upgradable.yml --prune
```

OPENAI API key is needed for running this project. Configure as environment variables
```bash
export OPENAI_API_KEY='<your-api-key>'
```

## Configuration

Hyper-parameters and frequently tuned research settings live in `config.yaml`. Avoid putting one-off constants or file paths there. Typical items include:

- RL policy and exploration settings
- Analyzer thresholds and dimension weights
- CBT module toggles and limits

Use `src/utils/config_loader.py` to load configuration at runtime.

## Data & Artifacts

- All data files reside under `data/`. Please do not modify existing contents.
- RL Q-tables are under `data/q_tables/`.
- Generated notes and reports are under `data/results/`.

If you need to run experiments, write new outputs under `data/results/` or another new subfolder instead of altering existing inputs.

## Run

The main Python entry points are located at the project root. Running from the root directory is required for imports to resolve correctly.

```bash
# Start the application (interactive / CLI-like)
python LLM_therapist_Application.py
```
```bash
# Start the simple FLASK server
python LLM_therapist_Application_server.py

# Health Check
curl -s http://127.0.0.1:8899/health

# Start Session（Return the first Question）
curl -sX POST 'http://127.0.0.1:8899/gpt' \
  -H 'Content-Type: application/json' \
  -d '{"user_input":"start","subject_ID":"8901"}'

# Continue（Send User's answer to App）
curl -sX POST 'http://127.0.0.1:8899/gpt' \
  -H 'Content-Type: application/json' \
  -d '{"user_input":"I feel anxious recently","subject_ID":"8901"}'
```

Notes:

- If you rely on GPUs, ensure `CUDA_VISIBLE_DEVICES=1,2,3` is set before execution.
- Logs will print status to terminal; see `src/utils/log_util.py` for configuration.


## Pipeline at a Glance

The end-to-end conversation pipeline follows this sequence:

1. Ask a question (prompting)
2. Segment user response
3. Map segments to dimensions and assign scores
4. For unmapped segments, attempt a single restatement
5. When a segment receives score = 2, enter Reflection & Validation (RV)
6. After completing a dimension, proceed to the next
7. After all dimensions are processed, or user prompts `stop`, enter CBT

Core modules:

- `src/response_analyzer.py`: segmentation, mapping, scoring
- `src/reflection_validation.py`: RV logic
- `src/CBT.py`: CBT intervention steps
- `src/handler_rl.py`: RL piteration and policy updates 
- `src/questioner.py`: question generation

## Logging
The system uses logs for key runtime events to aid development and diagnosis. You can adjust levels and formats in `src/utils/log_util.py`.

## Reproducibility Tips

- Fix random seeds where applicable if you aim for deterministic comparisons.
- Keep environment pinned via `environment.yml` and document any additional packages you install.
- Do not overwrite files under `data/`; write new artifacts to timestamped files in `data/results/`.

## Citation

If you use this repository in academic work, please cite the original paper above at <https://doi.org/10.1145/3712299>. The BibTeX entry appears at the top of this README for convenience.

## License

If not otherwise specified in the repository, this code is provided for research and educational purposes without warranty. Please check with the repository owner before commercial use.



