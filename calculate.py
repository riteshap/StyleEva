
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
        self.model = SentenceTransformer(r"G:\python_code\pre_model\all-MiniLM-L6-v2")
        pass

    def formate_check(self):
        err_num = 0
        for i in range(3, 13):
            file_name = f"{self.task_n}_test_{self.llm_b_name}_{i}.json"
            file_path = self.cur_path + f"/result/{file_name}"
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

    def calculate(self):
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )
        # 3. SBERT cosine similarity
        sbert_model = SentenceTransformer(r"G:\python_code\pre_model\all-mpnet-base-v2")

        for i in range(5, 10):
            print(f"i={i}")
            file_name = f"{self.task_n}_test_src_{self.level_n}_{self.llm_b_name}_{i}.json"
            file_path = self.cur_path + f"/result/{file_name}"
            file = json.load(open(file_path, "r", encoding="utf-8"))
            rouge_1_list = []
            rouge_2_list = []
            rouge_L_list = []
            bertscore_list = []
            sbert_list = []
            for target_list in [rouge_1_list, rouge_2_list, rouge_L_list, bertscore_list, sbert_list]:
                for s in range(i):
                    target_list.append([])

            ele_i = 0
            for ele in file:
                print(ele_i, end=" ")
                ele_i += 1
                src_ind = ele["src_ind"]
                sentence = ele["sentence"]
                llm_out = ele[self.llm_name]
                llm_out_json = json.loads(llm_out)
                dict_len = len(llm_out_json)

                emb2 = sbert_model.encode(sentence, convert_to_tensor=True)
                for j in range(dict_len):
                    sent2 = llm_out_json[f"level_{j+1}"]

                    rouge_scores = scorer.score(sentence, sent2)
                    rouge1 = rouge_scores["rouge1"].fmeasure
                    rouge2 = rouge_scores["rouge2"].fmeasure
                    rougeL = rouge_scores["rougeL"].fmeasure

                    # 2. BERTScore
                    P, R, F1 = score([sent2], [sentence], lang="en", verbose=False)
                    bertscore = F1.item()

                    # 3. SBERT cosine similarity
                    emb1 = sbert_model.encode(sent2, convert_to_tensor=True)
                    sbert_score = cos_sim(emb1, emb2).item()

                    rouge_1_list[j].append(rouge1)
                    rouge_2_list[j].append(rouge2)
                    rouge_L_list[j].append(rougeL)
                    bertscore_list[j].append(bertscore)
                    sbert_list[j].append(sbert_score)

            print("")
            for j in range(i):
                print(f"j={j}")
                for target_list in [rouge_1_list, rouge_2_list, rouge_L_list, bertscore_list, sbert_list]:
                    a = round(sum(target_list[j])/len(target_list[j]), 4)
                    b = round(np.std(target_list[j]), 4)
                    print(f"{a}   {b}", end="    ")
                print("\n")

    def generate(self):
        for i in range(3, 11):
            print(f"i={i}")
            text_file_name = f"{self.task_n}_test_dsv4f_{i}.json"
            text_file_path = self.cur_path + f"/result/{text_file_name}"
            text_file = json.load(open(text_file_path, "r", encoding="utf-8"))
            score_name = f"{self.task_n}_test_gpt56l_{i}.json"
            score_path = self.cur_path + f"/score/{score_name}"
            score_file = json.load(open(score_path, "r", encoding="utf-8"))

            ret_list = []
            for ind, ele in enumerate(text_file):
                if ind % 10 == 0:
                    print(ind, end=" ")
                src_ind = ele["src_ind"]
                sentence = ele["sentence"]
                output = json.loads(ele["deepseek-v4-flash"])
                target_sent = output[f"level_{i}"]
                sent_list = []
                score_list = []
                for ele2 in score_file:
                    if ele2["src_ind"] == int(src_ind):
                        true_dict = ele2["true_dict"]
                        score_out = json.loads(ele2["GPT5.6-Luna"])
                        for ind_key in score_out.keys():
                            level_name = true_dict[str(int(ind_key)-1)]
                            sent_list.append(output[level_name])
                            score_list.append(score_out[ind_key])
                        break
                sent_list = [sentence] + sent_list + [target_sent]
                score_list = [0] + score_list + [100]
                embeddings = self.model.encode(
                    sent_list,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

                ref_embedding = embeddings[0]

                similarities = torch.matmul(embeddings[1:], ref_embedding)

                ele_dict = {"src_ind": src_ind,
                            "sentence": sentence,
                            "sent_list": sent_list[1:],
                            "score": score_list[1:],
                            "sbert_sim": similarities.to("cpu").detach().tolist()}

                ret_list.append(ele_dict)

            print("")
            save_path = self.cur_path + f"/for_compare/{self.task_n}_test_gpt56l_compare_{i}.json"
            json.dump(ret_list, open(save_path, "w", encoding="utf-8"), indent=2)

    def generate_v2(self):
        ret_list = {}
        for layer_num in range(3, 9):
            result_name = f"{self.task_n}_test_dsv4f_{layer_num}.json"
            result_file_path = self.cur_path + f"/result/{result_name}"
            result_file = json.load(open(result_file_path, "r", encoding="utf-8"))
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

                ref_embedding = embeddings[0]

                similarities = torch.matmul(embeddings[1:], ref_embedding)
                sim_score_list = similarities.to("cpu").detach().tolist()

                if src_ind not in ret_list.keys():
                    ret_list[src_ind] = {"sentence": sentence, "sent_list": sent_list,
                                         "sbert_sim": sim_score_list}
                else:
                    assert ret_list[src_ind]["sentence"] == sentence
                    ret_list[src_ind]["sent_list"].extend(sent_list)
                    ret_list[src_ind]["sbert_sim"].extend(sim_score_list)

        save_name = f"{self.task_n}_test_dsv4f_all.json"
        save_path = self.cur_path + f"/result/{save_name}"
        json.dump(ret_list, open(save_path, "w", encoding="utf-8"), indent=2)

        pass


if __name__ == "__main__":
    root_path = r"G:\python_code\StyleTrans"
    task_n, level_n = "task1", "level-0"
    llm_dict = {"deepseek-v4-flash": "dsv4f",
                "qwen3.5-flash": "qw35f",
                "gemini3.5-flash": "gm35f",
                "GPT5.6-Luna": "gpt56l"}
    api_keys_dict = {"deepseek-v4-flash": "sk-6003",
                     "qwen3.5-flash": "sk-f5de",
                     "gemini3.5-flash": "sk-or-v1-e05",
                     "GPT5.6-Luna": "sk-or-v1-e05b7"}
    llm_name = "deepseek-v4-flash"
    llm_b_name = llm_dict[llm_name]

    ccc = Calculate(root_path, task_n, level_n, llm_dict, llm_name)
    ccc.generate_v2()
    # ccc.calculate()

