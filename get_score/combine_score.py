
import os
import json

task_n = "task8"
v_t = "train"
llm_b_name = "gpt56l"
times = "3"
root_paht = r"G:\python_code\StyleTrans"
score_path = root_paht + r"\baseline\get_score\score"
ret_list = []
src_ind_list = []
for file_name in os.listdir(score_path):
    if not f"{task_n}_{v_t}_{llm_b_name}_all_{times}" in file_name:
        continue
    file = json.load(open(f"{score_path}/{file_name}"))
    for ele in file:
        src_ind = ele["src_ind"]
        if src_ind not in src_ind_list:
            src_ind_list.append(src_ind)
            ret_list.append(ele)

save_name = f"{task_n}_{v_t}_{llm_b_name}_all_{times}.json"
save_path = f"{score_path}/{save_name}"
json.dump(ret_list, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
