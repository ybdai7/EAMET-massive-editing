# EAMET-massive-editing
Official code implementation of "EAMET: ROBUST MASSIVE MODEL EDITING
VIA EMBEDDING ALIGNMENT OPTIMIZATION"

## Environment Setup

### Requirements
At least one NVIDIA GPU with 80GB

- python==3.9.21
- torch==2.0.0
- tokenizers==0.21.0
- torchaudio==2.0.0
- torchvision==0.15.0
- transformers==4.49.0
- datasets==1.18.0
- nltk==3.6.5
- numpy==1.22.4
- pandas==2.2.3
- scipy==1.13.1
- scikit-learn==1.6.1
- matplotlib==3.5.1

### Usage

#### Configuration
The main configuration is done through `general.sh`. Here are the key parameters you can customize:

1. **Algorithm Selection** (`alg_name`):
   - EAMET (default)
   - MEMIT
   - PMET
   - ROME
   - FT
   - MEND
   - ALPHAEDIT

2. **Model Selection** (`model_name`):
   - NousResearch/Llama-2-7b-hf (default)
   - meta-llama/Llama-3.1-8B
   - NousResearch/Llama-2-13b-hf
   - tiiuae/falcon-7b
   - deepseek-ai/deepseek-llm-7b-base
   - Qwen/Qwen2.5-7B
   - google/gemma-7b-it
   - microsoft/phi-1_5

3. **Dataset Selection** (`ds_name`):
   - counterfact (default)
   - zsre
   - wikirecent

4. **Hyperparameters**:
   - Choose appropriate `hparams_fname` based on your model.

5. **GLEU Benchmark Evaluation**:
   To evaluate the edited models using the GLEU benchmark, modify the evaluation command in `general.sh`:
   ```diff
   - python -m experiments.evaluate \
   + python -m experiments.evaluate_gleu \
   ```
   The GLEU benchmark provides additional metrics for assessing the general
   ability of edited models.

#### Running Experiments
1. Configure your parameters in `general.sh`
2. Run the script:
   ```bash
   bash general.sh
   ```

#### Additional Options
- Set `dataset_size_limit` to control the number of editing tasks (default: 10000)
- Use `--use_cache` flag to cache KV pairs if needed
- Adjust `assigned_prefix_len` for evaluation (default: 5)

### Output
Results will be saved in the specified output directory with your chosen
`./results/out_name`.