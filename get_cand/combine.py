
import os
import json
import pickle
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from scipy.stats import kendalltau, spearmanr
from sentence_transformers.util import cos_sim
from bert_score import score
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch


os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar()

class Calculate:
    def __init__(self, root_path, task_n, level_n, llm_dict, llm_name):
        self.task_n = task_n
        self.level_n = level_n
        self.llm_dict = llm_dict
        self.llm_name = llm_name
        self.llm_b_name = llm_dict[llm_name]
        self.root_path = root_path
        self.cur_path = root_path + "/baseline"
        self.model = SentenceTransformer(r"G:\python_code\pre_model\all-mpnet-base-v2")
        # self.model2 = SentenceTransformer(r"G:\python_code\pre_model\all-MiniLM-L6-v2")
        pass

    def formate_check(self, v_t):
        assert v_t in ["test", "train"]
        err_num = 0
        for i in range(3, 9):
            file_name = f"{self.task_n}_{v_t}_{self.llm_b_name}_{i}.json"
            file_path = self.cur_path + f"/get_cand/result/{file_name}"
            file = json.load(open(file_path, "r", encoding="utf-8"))
            for ele in file:
                src_ind = ele["src_ind"]
                sentence = ele["sentence"]
                llm_out = ele[self.llm_name]
                try:
                    llm_out_json = json.loads(llm_out)
                    dict_len = len(llm_out_json)
                    for j in range(dict_len):
                        ss = llm_out_json[f"level_{j+1}"]
                except Exception as e:
                    err_num += 1
                    print(file_name)
                    print(e)
                    print(f"src_ind={src_ind}")
                    print(sentence)
                    pass
        if err_num > 0:
            raise Exception(f"error number = {err_num}")

    def generate_v2(self, v_t):
        assert v_t in ["test", "train"]
        ret_list = {}
        for layer_num in range(3, 9):
            result_name = f"{self.task_n}_{v_t}_dsv4f_{layer_num}.json"
            result_file_path = self.cur_path + f"/get_cand/result/{result_name}"
            result_file = json.load(open(result_file_path, "r", encoding="utf-8"))
            print(f"current layer:{layer_num}")
            for ind, ele in enumerate(result_file):
                if ind % 10 == 0:
                    print(ind, end=" ")
                src_ind = ele["src_ind"]
                sentence = ele["sentence"]
                output = json.loads(ele["deepseek-v4-flash"])
                sent_list = [ee for ee in output.values()]

                embeddings = self.model.encode(
                    [sentence]+sent_list,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

                # list[0] 作为参考
                ref_embedding = embeddings[0]

                # list[1:] 与 list[0] 的余弦相似度
                similarities = torch.matmul(embeddings[1:], ref_embedding)
                sim_score_list = similarities.to("cpu").detach().tolist()

                if src_ind not in ret_list.keys():
                    ret_list[src_ind] = {"sentence": sentence, "sent_list": sent_list,
                                         "sbert_sim": sim_score_list}
                else:
                    assert ret_list[src_ind]["sentence"] == sentence
                    ret_list[src_ind]["sent_list"].extend(sent_list)
                    ret_list[src_ind]["sbert_sim"].extend(sim_score_list)

        save_name = f"{self.task_n}_{v_t}_dsv4f_all.json"
        save_path = self.cur_path + f"/get_cand/result/{save_name}"
        json.dump(ret_list, open(save_path, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    root_path = r"G:\python_code\StyleTrans"
    task_n, level_n = "task8", "level-0"
    v_t = "train"
    llm_dict = {"deepseek-v4-flash": "dsv4f",
                "qwen3.5-flash": "qw35f",
                "gemini3.5-flash": "gm35f",
                "GPT5.6-Luna": "gpt56l"}
    api_keys_dict = {"deepseek-v4-flash": "sk-6003d",
                     "qwen3.5-flash": "sk-f5ded7",
                     "gemini3.5-flash": "sk-or-v1-e05",
                     "GPT5.6-Luna": "sk-or-v1-e05b731"}
    llm_name = "deepseek-v4-flash"
    llm_b_name = llm_dict[llm_name]

    ccc = Calculate(root_path, task_n, level_n, llm_dict, llm_name)
    ccc.formate_check(v_t)
    ccc.generate_v2(v_t)

