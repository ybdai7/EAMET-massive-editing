import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class MENDQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"
        self.model_name_config = kwargs.get("model_name_config", None)

        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        for ind, record in enumerate(raw):
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"

            if "mistral" in str(self.model_name_config).lower() \
                or "qwen" in str(self.model_name_config).lower() \
                or "deepseek" in str(self.model_name_config).lower():
                ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            elif "llama" in str(type(tok)) or \
                "llama-3.1" in str(self.model_name_config).lower():
                ans_toks = tok(" " + record["loc_ans"])["input_ids"][1:]
            else:
                ans_toks = tok(" " + record["loc_ans"])["input_ids"]

            data.append(
                {
                    "case_id": ind,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts":[
                        {
                            "prompt": record["loc"] + "?" + self._correct_decode(tok, ans_toks[:i], ind, type="prompt"),
                            "target": self._correct_decode(tok, ans_toks[i], ind, type="target"),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        if kwargs.get("repeat_subjects", None) != None:
            subject_dict = {}
            need_check_id = []
            for idx, item in enumerate(data):
                subject = item["requested_rewrite"]["subject"]
                if subject in subject_dict:
                    subject_dict[subject].append(idx)
                else:
                    subject_dict[subject] = [idx]
            
            duplicate_subjects = {subj: ids for subj, ids in subject_dict.items()}
            for subject, ids in duplicate_subjects.items():
                if len(ids) == kwargs.get("repeat_subjects", 1):
                    need_check_id.extend(ids)

            select_data = [data[i] for i in need_check_id]
            data = select_data
        
        if kwargs.get("randomize_editing_sequence", False):
            random.seed(kwargs.get("shuffle_seed", 0))
            random.shuffle(data)

        # if kwargs.get("repeat_subjects", None) != None:
        #     subject_dict = {}
        #     need_check_id = []
        #     need_check_id_1 = []
        #     for idx, item in enumerate(data):
        #         subject = item["requested_rewrite"]["subject"]
        #         if subject in subject_dict:
        #             subject_dict[subject].append(idx)
        #         else:
        #             subject_dict[subject] = [idx]
            
        #     duplicate_subjects = {subj: ids for subj, ids in subject_dict.items()}
        #     for subject, ids in duplicate_subjects.items():
        #         if len(ids) == kwargs.get("repeat_subjects", 1):
        #             need_check_id.extend(ids)
        #         if len(ids) == 1:
        #             need_check_id_1.extend(ids)

        #     select_data = [data[i] for i in need_check_id]
        #     select_data_1 = [data[i] for i in need_check_id_1]
        #     data = select_data_1[:size//2]+ select_data[:size//2]

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
    
    def _correct_decode(self, tok, target_tokens, ind, type):
        """
        Correctly decode the target tokens for llama-3.1 models.
        """
        return tok.decode(target_tokens)