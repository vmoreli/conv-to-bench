# conv-to-bench

A toolkit to build and evaluate benchmarks derived from implicit user feedback in conversational datasets. The repository provides data extraction pipelines, LangGraph-based workflows to generate checklist requirements from conversations, and an LLM-driven evaluator that scores model responses against those checklists.

**Key ideas**
- Extract instructions and feedback from large conversational corpora (LMSYS, WildChat).
- Convert feedback (positive/negative turns) into explicit checklist requirements.
- Use LLMs as automated judges to evaluate solutions against the generated checklists.

**Features**
- Data loading and filtering for coding-related conversations: [src/data/loader.py](src/data/loader.py)
- LangGraph workflows for dataset generation and evaluation: [src/graphs/workflows.py](src/graphs/workflows.py)
- Nodes that call LLMs to extract instructions, identify feedback and create checklists: [src/graphs/nodes](src/graphs/nodes)
- LLM client wrapper supporting Gemini and OpenAI: [src/llm/client.py](src/llm/client.py)
- Orchestrating scripts for dataset generation and evaluation: [scripts](scripts)

Project layout
- [config.py](config.py) — central configuration (paths, model lists, pricing)
- [requirements.txt](requirements.txt) — Python dependencies
- [scripts/02_generate_dataset.py](scripts/02_generate_dataset.py) — run the dataset extraction pipeline
- [scripts/03_run_evaluation.py](scripts/03_run_evaluation.py) — evaluate model responses using checklist-based judgments
- [src/] — implementation of graphs, nodes, prompts, schemas and utilities

Quickstart
1. Install dependencies (preferably inside a virtualenv):

```bash
pip install -r requirements.txt
```

2. Provide credentials / environment variables for your LLM of choice (Gemini or OpenAI) in a `.env` file at the repository root. The repository uses `python-dotenv` and `src/llm/client.py` will load these values.

3. Configure paths and models in [config.py](config.py) as needed. Note: the active LLM in code is also selectable in [src/llm/client.py](src/llm/client.py) via the `SELECTED_MODEL` constant.

4. Generate the extracted dataset (runs the LangGraph dataset pipeline):

```bash
python scripts/02_generate_dataset.py --output-dir output/traces --workers 4
```

5. Run evaluations using the generated responses and checklist files:

```bash
python scripts/03_run_evaluation.py --responses-file path/to/extracted_dataset.json --checklist-file path/to/checklists.json --output-dir output/evals --models "gemini-2.5-flash" --workers 6 --runs 1
```

Notes on configuration and behavior
- Dataset loading and filtering: see [src/data/loader.py](src/data/loader.py). The loader caches filtered datasets to `data/` paths defined in `config.py`.
- LangGraph workflows: graph entrypoints are `dataset_app` and `evaluator_app` in [src/graphs/workflows.py](src/graphs/workflows.py).
- LLM calls: centralized in [src/llm/client.py](src/llm/client.py). The client supports both Google Gemini and OpenAI APIs; ensure credentials and required SDKs are installed.
- Token & cost tracking: token accounting is aggregated in [src/utils/tracking.py](src/utils/tracking.py) and the client computes cost estimates using `PRICES` from `config.py`.

Development tips
- Run on a small debug subset first by adjusting the dataset loader or passing debug IDs to `load_datasets()`.
- Use the `--resume` and checkpointing behavior in `scripts/03_run_evaluation.py` to continue long runs.
- Call logging for LLM requests is written to the directory configured by `CONV_HIST_DIR` in `config.py`.

Contact
If you have questions about running the code or extending the pipelines, open an issue or contact the maintainers.

