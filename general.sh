#!/bin/bash
module purge                     # clear environment modules inherited from submission
# alg_name could be 
# - EAMET 
# - MEMIT
# - PMET
# - ROME
# - FT
# - MEND
# - ALPHAEDIT
export alg_name=EAMET

# choose your job name to be displayed in the results folder
export job_name=your_job_name_to_be_displayed

# choose your model from the following list
# model_name:
# - "meta-llama/Llama-3.1-8B",
# - "NousResearch/Llama-2-13b-hf",
# - "NousResearch/Llama-2-7b-hf",
# - "tiiuae/falcon-7b",
# - "deepseek-ai/deepseek-llm-7b-base",
# - "Qwen/Qwen2.5-7B",
# - "google/gemma-7b-it",
# - "microsoft/phi-1_5",

export model_name=NousResearch/Llama-2-7b-hf

# choose your hparams file from the following list
# hparams_fname:
# - "LLAMA3-8B.json",
# - "Qwen-7B.json",
# - "LLAMA2-13B.json",
# - "LLAMA2-7B.json",
# - "Deepseek-7B.json",
# - "Falcon-7B.json",
export hparams_fname=LLAMA2-7B.json

# choose dataset from the following list
# ds_name:
# - "counterfact",
# - "zsre",
# - "wikirecent"
export ds_name=counterfact
export dir_name=counterfact

# choose your output dir name
export out_name="llama2-counterfact" 

# choose your assigned prefix length during evaluation
# we choose 5 for default 10000 editing tasks
export assigned_prefix_len="5"

# choose your cache id if need to cache all KV pairs
export cache_id="0"

# v_lr for PMET, MEMIT, ALPHAEDIT, EAMET is 0.5
export v_lr="0.5"
export dataset_size_limit="10000"

echo "-----------------------------------------"
echo "Running evals for $dataset_size_limit ..."
echo " "

# change experiments.evaluate to experiments.evaluate_gleu if you want to
# evaluate on GLEU benchmark 
python3 -m experiments.evaluate \
    --job_name $job_name \
    --alg_name $alg_name \
    --model_name $model_name \
    --hparams_fname $hparams_fname \
    --ds_name $ds_name \
    --dir_name $dir_name \
    --out_name $out_name \
    --dataset_size_limit $dataset_size_limit \
    --assigned_prefix_len $assigned_prefix_len \
    --cache_id $cache_id \
    --edit_layer $edit_layer \
    --lambda_cov $lambda_cov \
    --v_lr $v_lr\ 

    # uncomment if you want to cache all KV pairs
    # --use_cache \
    # uncomment if you want to cache all KV pairs

    ### uncomment if needed
    # --model_path $model_path\ 
    ### uncomment if needed

echo " "
echo "Finishing evals for $dataset_size_limit ..."
echo "-----------------------------------------"
