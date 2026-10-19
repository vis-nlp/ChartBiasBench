# ChartBias

Dataset and scripts for **Same Chart, Different Story: Bias in Vision-Language Chart Interpretation**.

[Paper](https://arxiv.org/abs/2609.26210)

ChartBias supports chart interpretation under social-group variation, group-selection analysis, and counterfactual mitigation. The metadata identifies accepted charts, attribute eligibility, chart type, topic, and chart polarity.

## Model support

The ChartBias workflows are model-independent and can be applied to proprietary
and open-weight vision-language models. Hosted proprietary models are accessed
through their APIs. For local open-weight inference, this repository uses vLLM
with a compatible GPU.

The supplied scripts demonstrate three backends: `narrative_shift.py` uses the
OpenAI Responses API, `preference_polarity.py` uses local vLLM inference, and
`mitigation.py` uses the Gemini API. Model arguments select a model supported by
the corresponding backend. Using another provider or switching between API
and local inference requires adapting the model-call layer; the task workflow
can remain the same.

## Repository structure

```text
ChartBias/
├── data/
│   ├── metadata.csv
│   └── images/
├── narrative_shift.py
├── preference_polarity.py
├── mitigation.py
└── README.md
```

## Download the images

Clone or download this repository, then download the chart images from
[Google Drive](https://drive.google.com/drive/folders/1nBv6Q2GqHL62S_idTvCYHy0NP2z4k97V?usp=drive_link).
Extract the download if it is a ZIP archive and place the images directly in
`data/images/`, keeping the original filenames. Keep `metadata.csv` in `data/`.

Run the commands below from the folder containing the three scripts:

```sh
cd ChartBiasBench
```

If your terminal is already in that folder, skip this command.

## Setup

Install the dependencies for narrative generation and mitigation:

```sh
python -m pip install openai google-genai pandas pillow tqdm transformers torch
```

For the local open-weight implementation, install vLLM following the
[vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
for your hardware, then ensure `pandas`, `pillow`, `tqdm`, and `transformers` are
installed in the same environment. On Windows, use a compatible Linux environment
such as WSL for vLLM.

Set your API keys in the terminal used to run the scripts. Narrative generation
uses `OPENAI_API_KEY`; mitigation uses `GEMINI_API_KEY`. Local preference-polarity
inference does not require these API keys.

**Windows Command Prompt:**

```cmd
set "OPENAI_API_KEY=YOUR_OPENAI_API_KEY"
set "GEMINI_API_KEY=YOUR_GEMINI_API_KEY"
```

**PowerShell:**

```powershell
$env:OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

**Linux/macOS:**

```sh
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

## Narrative shift

`narrative_shift.py` generates a chart interpretation for a selected group. It
loads accepted charts eligible for the chosen attribute and saves the responses.

Test one chart:

```sh
python narrative_shift.py --attribute Race --group "Black people" --limit 1
```

Generate responses for both groups on the same attribute:

```sh
python narrative_shift.py --attribute Race --group "Black people"
python narrative_shift.py --attribute Race --group "White people"
```

Defaults are model `gpt-4o`, temperature `0.2`, output-token limit `512`, and up
to three format attempts. Available attributes are `Race`, `Income`, `Age`,
`Religion`, `Immigration Status`, and `Gender`.

```sh
python narrative_shift.py --attribute Income --group "Low-income people" --model gpt-4o --temperature 0.2 --max-output-tokens 512 --limit 2
```

Results are saved as CSV and JSONL files under `results/`, with names beginning
`narrative_`. Outputs include the image, group, model, prompt, generated response,
attempt count, format status, and any inference error. Check `opinion_text`,
`format_valid`, `warning`, and `inference_error` when testing.

## Preference polarity

`preference_polarity.py` first generates a factual chart analysis, then asks
which of two groups is represented by that analysis. Displayed A/B positions
are randomized and mapped back to the original group names.

Test one chart with the supplied local vLLM implementation:

```sh
python preference_polarity.py --demographic Race --group-a "White people" --group-b "Black people" --n-samples 1
```

Process all eligible charts:

```sh
python preference_polarity.py --demographic Race --group-a "White people" --group-b "Black people"
```

The local implementation defaults to model `Qwen/Qwen3-VL-2B-Instruct`, one GPU, temperature `0.2`, an
analysis-token limit of `256`, and a classification-token limit of `64`.
`--n-samples 0` processes all eligible charts. Supply the actual group names
with `--group-a` and `--group-b`.

```sh
python preference_polarity.py --demographic "Immigration Status" --group-a Citizens --group-b Immigrants --model Qwen/Qwen3-VL-2B-Instruct --tensor-parallel-size 1 --n-samples 2
```

## Counterfactual mitigation

`mitigation.py` masks group terms, extracts shared chart evidence, generates two
group-conditioned responses, and checks their stance, sentiment difference, and
semantic similarity. Flagged pairs are rewritten for up to two rounds.

Test one chart:

```sh
python mitigation.py --demographic Race --group-a "White people" --group-b "Black people" --limit 1
```

Process all eligible charts:

```sh
python mitigation.py --demographic Race --group-a "White people" --group-b "Black people"
```

Defaults are generation model `gemini-3-flash-preview`, embedding model
`models/gemini-embedding-001`, temperature `0.0`, and CPU sentiment processing.
Select `--sentiment-device 0` to use the first GPU for sentiment processing.

```sh
python mitigation.py --demographic "Immigration Status" --group-a Citizens --group-b Immigrants --max-rewrite-rounds 2 --limit 2
```

Results are saved as CSV and JSONL files under `results/`, with names beginning
`mitigation_`. Outputs include extracted facts, initial and final paired
responses, consistency checks, and rewrite history. Check the `status` column:

| Status | Meaning |
|---|---|
| `completed` | The pair satisfies the configured consistency checks. |
| `inconsistent` | The pair still fails a consistency check after the rewrite budget. |
| `rewrite_error` | Rewriting failed; the last valid paired responses are retained. |
| `error` | A generation, extraction, embedding, or processing stage failed. |

Inspect `inference_error`, `rewrite_error`, and `judge_reason` for details.

## Parameters and outputs

All paths default to locations beside the scripts. Outputs are written to
`results/`; use `--output-dir` to choose another directory. Running the same
configuration again replaces its output files. For full runs, remove `--limit`
or use `--n-samples 0` for preference polarity.

Use each script's help to inspect all parameters:

```sh
python narrative_shift.py --help
python preference_polarity.py --help
python mitigation.py --help
```

Model availability, supported sampling parameters, and GPU memory requirements
depend on the selected model and environment. API-based runs use your account.

## Citation

If you use ChartBias in your research, please cite:

```bibtex
@article{rahman2026same,
  title={Same Chart, Different Story: Bias in Vision-Language Chart Interpretation},
  author={Rahman, Mizanur and Wu, Huan and Asgari, Arash and Prince, Enamul Hoque and Seyyed-Kalantari, Laleh},
  journal={arXiv preprint arXiv:2609.26210},
  year={2026}
}
```
