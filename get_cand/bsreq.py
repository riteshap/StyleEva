

import json
import os
import time
import random
import numpy as np
from openai import OpenAI

class DefectMining:
    def __init__(self, root_path, target_style, llm_name, llm_dict, task_n, level_n, api_keys
                 , top_k: int = 10):
        self.root_path = root_path
        self.cur_path = self.root_path + "/baseline"
        self.target_style = target_style
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.llm_b_name = self.llm_dict[self.llm_name]
        self.task_n = task_n
        self.level_n = level_n
        self.top_k = top_k
        # self.model = sbert

        self.api_keys = api_keys
        self.prefix_temp = open(self.cur_path + f"/get_cand/prefix").read()
        self.prompt_temp = open(self.cur_path + f"/get_cand/prompt").read()
        self.prompt_p2_temp = open(self.cur_path + f"/get_cand/prompt_p2").read()
        ss = "\n".join(["n. {REF_n}".replace("n", str(i + 1)) for i in range(self.top_k)])
        self.prompt_temp = self.prompt_temp.replace("{REF_N}", ss)

    def _llm_flash(self, prefix, que, api_key):
        if self.llm_name == "deepseek-v4-flash":
            base_url_name = "https://api.deepseek.com"
            model_name = "deepseek-v4-flash"
        elif self.llm_name == "qwen3.5-flash":
            base_url_name = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            model_name = "qwen3.5-flash"
        elif self.llm_name == "gemini3.5-flash":
            base_url_name = "https://openrouter.ai/api/v1"
            model_name = "google/gemini-3.5-flash"
        elif self.llm_name == "nemotron3-ultra":
            base_url_name = "https://openrouter.ai/api/v1"
            model_name = "openai/gpt-5.6-luna"
        else:
            raise Exception("error")

        client = OpenAI(api_key=api_key, base_url=base_url_name)
        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prefix},
                        {"role": "user", "content": "{}".format(que)},
                    ],
                    stream=False
                )
                ss = response.choices[0].message
                if len(ss.content) == 0:
                    raise Exception("Defect2Styletext:LLM empty")
                return ss.content
            except Exception as e:
                print(e)
                # print(prefix)
                print(que.split("Example")[0])
                time.sleep(3)
                if i >= 4:
                    print("error")
                    return ""

    def request_llm(self, layer_num, v_t, part=(None, None)):

        assert v_t in ["train", "test"]

        part_name = f"_{part[0]}-{part[1]}" if part[0] is not None else ""
        if part[0] is not None:
            assert 0 <= float(part[0]) <=1
            assert 0 <= float(part[1]) <=1

        defect_path = self.root_path + "/defect_mining"
        train_ref = json.load(open(defect_path + f"/weakness/{self.task_n}_train_ref.json", "r", encoding="utf-8"))

        def get_llm_output(src_file_name, n):
            cache_path = self.cur_path + "/get_cand/cache/" + src_file_name.replace(".json", f"{n}.json")
            if os.path.exists(cache_path):
                cache_file = json.load(open(cache_path, "r", encoding="utf-8"))
            else:
                cache_file = []

            src_file = json.load(open(defect_path + "/weakness/" + src_file_name, "r", encoding="utf-8"))

            total_num = len(src_file.keys())

            start_num, end_num = 0, 0
            if part_name != "":
                start_num, end_num = float(part[0]) * total_num, float(part[1]) * total_num
                print(f"total_num: {total_num}    part index：{start_num}-{end_num}")

            cur_num = -1
            ret_out_list = []
            start_time_ori = time.time()
            for ind in src_file.keys():
                cur_num += 1
                if part_name != "" and not start_num <= cur_num <= end_num:
                    continue

                if cur_num % 5 == 0:
                    print(f"{round(100 * cur_num / total_num, 2)}%, "
                          f"time: {round((time.time() - start_time_ori) / 60, 2)}min")

                ref_list = []
                for tr_ind in train_ref.keys():

                    if "valid" not in src_file_name and tr_ind == ind:
                        continue
                    ref_list.extend(train_ref[tr_ind])

                sent = src_file[ind][0]

                # simn_few_shot = self.select_sim_k(sent, ref_list)
                simn_few_shot = random.sample(ref_list, k=self.top_k)

                prefix_dict = {"TARGET_STYLE": self.target_style}
                prompt_dict = {"TARGET_STYLE": self.target_style, "SOURCE_TEXT": sent, "N":n}  # , "TARGET_STYLE": prefix_dict["TARGET_STYLE"]
                for i in range(self.top_k):
                    prefix_dict[f"REF_{i+1}"] = simn_few_shot[i]
                    prompt_dict[f"REF_{i+1}"] = simn_few_shot[i]

                prefix = self.prefix_temp.format(**prefix_dict)
                prompt = self.prompt_temp.format(**prompt_dict) + self.prompt_p2_temp

                cache_flag = False
                for ele in cache_file:
                    if ele[0] == sent:
                        print("命中")
                        llm_output = ele[1]
                        ret_out_list.append({"src_ind": ind, "sentence": sent, self.llm_name: llm_output})
                        cache_flag = True
                        break

                if not cache_flag:

                    llm_output = self._llm_flash(prefix, prompt, self.api_keys)
                    ret_out_list.append({"src_ind": ind, "sentence": sent, self.llm_name: llm_output})

                    cache_file.append([sent, llm_output])
                    if int(ind) % 5 == 0:
                        if part_name != "":
                            start_time, end_time = float(part[0]) * 100, float(part[1]) * 100
                            while not (start_time < int(time.time() % 100) < end_time):
                                time.sleep(1)
                        json.dump(cache_file, open(cache_path, "w", encoding="utf-8"), indent=2)

            return ret_out_list

        # for v_t in ["test"]:
        target_src = f"{self.task_n}_{v_t}_src_{self.level_n}.json"

        print(f"layer：N={layer_num}")
        save_name = f"{self.task_n}_{v_t}_{self.llm_b_name}_{layer_num}{part_name}.json"
        save_path = self.cur_path + f"/get_cand/result/{save_name}"
        print(f"save_name={save_name}")
        if not os.path.exists(save_path):
            out_list = get_llm_output(target_src, layer_num)
            json.dump(out_list, open(save_path, "w", encoding="utf-8"), indent=2)
        else:
            out_list = json.load(open(save_path, "r", encoding="utf-8"))
            print(f"baseline.request_llm,{save_path} exist")
            print(f"baseline.request_llm, {save_path}, type:dict, len:{len(out_list)}")


if __name__ == "__main__":

    task_n, level_n = "task8", "level-0"
    v_t = "train"
    llm_name = "deepseek-v4-flash"
    layer_num = 3

    llm_dict = {"deepseek-v4-flash": "dsv4f",
                "qwen3.5-flash": "qw35f",
                "gemini3.5-flash": "gm35f",
                "GPT5.6-Luna": "gpt56l"}
    api_keys_dict = {"deepseek-v4-flash": "sk-600",
                     "qwen3.5-flash": "sk-f5",
                     "gemini3.5-flash": "sk-or-v1-e",
                     "GPT5.6-Luna": "sk-or-v1-e05"}
    llm_b_name = llm_dict[llm_name]
    target_style = ["positive English style", "negative English style",
                    "formal (standard formal expression)  English style",
                    "informal (casual, colloquial, nonstandard expression) English style",
                    "impolite English style", "polite English style",
                    "original (Shakespearean original English style) English style",
                    "modern (modernized contemporary English style) English style",
                    "text detoxification English style",
                    "toxic and verbally aggressive English style", ][int(task_n[-1]) - 1]

    root_path = r"G:\python_code\StyleTrans"
    api_keys = api_keys_dict[llm_name]

    aa = DefectMining(root_path, target_style, llm_name, llm_dict, task_n, level_n, api_keys, top_k=10)
    aa.request_llm(layer_num, v_t)


