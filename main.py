
import json
import os
import pickle

class Preprocessor:
    def __init__(self):
        pass

    def enter_next_round(self, root_path, task_n:str, level_n:str, llm_name:str, llm_dict:dict, mis_test_keys:list[str], mis_train_keys:list[str]):
        def get_add_up_data(root_path, mis_test_keys, mis_train_keys):
            add_test = []
            add_train = []
            for ri in range(4, 0, -1):
                r_n_name = "" if ri < 2 else f"_r-{ri}"
                file_name = f"{task_n}_test_{level_n}_{llm_dict[llm_name]}{r_n_name}.pkl"
                r_n_path = os.path.join(root_path + "/classifier/record_low", file_name)
                if not os.path.exists(r_n_path):
                    continue
                file = pickle.load(open(r_n_path, "rb"))
                checked_test_id = []

                for ele in file:
                    if ele[3] in mis_test_keys:
                        add_test.append(ele)
                        checked_test_id.append(ele[3])

                file_name = f"{task_n}_train_{level_n}_{llm_dict[llm_name]}{r_n_name}.pkl"
                r_n_path = os.path.join(root_path + "/classifier/record_low", file_name)
                if not os.path.exists(r_n_path):
                    continue
                file = pickle.load(open(r_n_path, "rb"))
                checked_train_id = []
                for ele in file:
                    if ele[3] in mis_train_keys:
                        add_train.append(ele)
                        checked_train_id.append(ele[3])

                mis_test_keys = [a for a in mis_test_keys if a not in checked_test_id]
                mis_train_keys = [a for a in mis_train_keys if a not in checked_train_id]

            if len(mis_test_keys) + len(mis_train_keys) != 0:
                raise Exception("error")

            return add_test, add_train

        add_test_data, add_train_data = get_add_up_data(root_path, mis_test_keys, mis_train_keys)

        up_path = root_path + "/classifier/record_up"
        test_name_key = f"{task_n}_test_{level_n}_{llm_dict[llm_name]}"
        train_name_key = f"{task_n}_train_{level_n}_{llm_dict[llm_name]}"
        up_test_data = []
        up_train_data = []
        for file in os.listdir(up_path):
            if test_name_key in file:
                cur_data = pickle.load(open(os.path.join(up_path, file), "rb"))
                up_test_data.extend(cur_data)
            if train_name_key in file:
                cur_data = pickle.load(open(os.path.join(up_path, file), "rb"))
                up_train_data.extend(cur_data)

        up_test_data.extend(add_test_data)
        up_train_data.extend(add_train_data)

        new_level = ""
        for i in range(7):
            if str(i) in level_n:
                new_level = level_n.replace(str(i), str(i + 1))
                break

        # 转存为下一轮的负样本
        test_src = f"{task_n}_test_src_{new_level}_{llm_dict[llm_name]}_emb.pkl"
        train_src = f"{task_n}_train_src_{new_level}_{llm_dict[llm_name]}_emb.pkl"
        pickle.dump(up_test_data, open(os.path.join("./classifier/pkl", test_src), "wb"))
        pickle.dump(up_train_data, open(os.path.join("./classifier/pkl", train_src), "wb"))

        def out_new_json(up_pkl):
            tt_json = {}
            for ele in up_pkl:
                sent = ele[2]
                src_ind = ele[3]
                if src_ind not in tt_json.keys():
                    tt_json[src_ind] = [sent]
                else:
                    tt_json[src_ind].append(sent)
            return tt_json

        test_json = out_new_json(up_test_data)
        train_json = out_new_json(up_train_data)

        test_json_name = f"{task_n}_test_src_{new_level}_{llm_dict[llm_name]}.json"
        train_json_name = f"{task_n}_train_src_{new_level}_{llm_dict[llm_name]}.json"

        with open(os.path.join("./defect_mining", test_json_name), "w", encoding="utf-8") as f:
            json.dump(test_json, f, indent=2, ensure_ascii=False)
        with open(os.path.join("./defect_mining", train_json_name), "w", encoding="utf-8") as f:
            json.dump(train_json, f, indent=2, ensure_ascii=False)

    def rank_file(self, task_n):
        for root_path in ["./defect_mining", "./defect2styletext"]:
            for file_name in os.listdir(root_path):
                if ".json" not in file_name:
                    continue
                if task_n not in file_name:
                    continue

                print(file_name)
                file = open(root_path + "/" + file_name, "r", encoding="utf-8")
                file = json.load(file)

                if type(file) != list:
                    sorted_file = {}
                    keys_list = sorted([int(a) for a in file.keys()])
                    for k_name in keys_list:
                        k_name = str(k_name)
                        sorted_file[k_name] = file[k_name]
                    assert len(sorted_file) == len(file)
                    json.dump(sorted_file, open(root_path + "/" + file_name, "w", encoding="utf-8"), indent=2,
                              ensure_ascii=False)

                else:
                    sorted_file = []
                    keys_list = sorted(list(set([int(ele["src_ind"]) for ele in file])))
                    for k_name in keys_list:
                        k_name = str(k_name)
                        for ele in file:
                            if ele["src_ind"] == k_name:
                                sorted_file.append(ele)
                    assert len(sorted_file) == len(file)
                    json.dump(sorted_file, open(root_path + "/" + file_name, "w", encoding="utf-8"), indent=2,
                              ensure_ascii=False)

    def check_rank(self, task_n):
        err_path = []
        for root_path in ["./defect_mining", "./defect2styletext"]:
            for file_name in os.listdir(root_path):
                if ".json" not in file_name:
                    continue
                if task_n not in file_name:
                    continue
                if "cache" in file_name:
                    continue
                if "_r-" in file_name:
                    continue
                print(file_name)
                file = open(root_path + "/" + file_name, "r", encoding="utf-8")
                file = json.load(file)

                if type(file) != list:
                    keys_list = sorted(list(set([int(a) for a in file.keys()])))
                    for ind, k_name in enumerate(keys_list):
                        if k_name != keys_list[-1] and k_name != keys_list[ind + 1] - 1:
                            err_path.append(root_path + "/" + file_name)
                else:
                    keys_list = sorted(list(set([int(ele["src_ind"]) for ele in file])))
                    for ind, k_name in enumerate(keys_list):
                        if k_name != keys_list[-1] and k_name != keys_list[ind + 1] - 1:
                            err_path.append(root_path + "/" + file_name)

        if len(err_path) > 0:
            raise Exception(f"{err_path}")

    def main(self):
        from defect_mining.start_round import ori2round1, ori2round1_v2
        from classifier.classification import (Turn2Emb, TrainTest, StyleTextClassifier)
        from defect_mining.process1 import DefectMining
        from defect_mining.defect_clean import DefectCleaner
        from defect2styletext.process1 import Defect2Styletext
        from defect2styletext.style_clean import StyleCleaner

        task_n, level_n = "task7", "level-1"
        llm_name = "nemotron3-ultra"
        acc_threshold = 0.75
        retention_rate = 0.5

        emb_model_path = r"G:\python_code\pre_model\Llama-3.2-3B-Instruct"
        sbert_path = r"G:\python_code\pre_model\all-mpnet-base-v2"

        llm_dict = {"deepseek-v4-flash": "dsv4f",
                    "qwen3.5-flash": "qw35f",
                    "gemini2.5-flash": "gm25f",
                    "llama3.3-70b": "la3370",
                    "nemotron3-ultra": "ne3u"}
        api_keys_dict = {"deepseek-v4-flash": "sk-600",
                         "qwen3.5-flash": "sk-f5de",
                         "gemini2.5-flash": "sk-or-v1-e",
                         "llama3.3-70b": "sk-or-v1-e",
                         "nemotron3-ultra": "sk-or-v1-e"}
        llm_b_name = llm_dict[llm_name]
        target_style = ["test",
                        "positive English style", "negative English style",
                        "formal English style", "informal English style",
                        "neutral English style", "toxic English style"][int(task_n[-1]) - 1]
        root_path = os.getcwd()
        api_keys = api_keys_dict[llm_name]
        if level_n == "level-0":
            if task_n in ["task1", "task4", "task5", "task6", "task7"]:
                train_data_path = f"./data/{task_n}_train.json"
                test_data_path = f"./data/{task_n}_test.json"
                ori2round1(root_path, train_data_path)
                ori2round1(root_path, test_data_path)
            elif task_n in ["task2", "task3"]:
                ori2round1_v2(root_path)
            else:
                raise Exception("no such task")

        te = Turn2Emb(root_path, llm_name, llm_dict, emb_model_path)
        te.src_ref_turn2emb(task_n, level_n)

        tt = TrainTest(root_path, llm_b_name)
        best_acc = tt.train_main(task_n, level_n)
        if best_acc < acc_threshold:
            return None

        top_k = 5
        dm = DefectMining(root_path, target_style, llm_name, llm_dict, task_n, level_n, api_keys, sbert_path, top_k)
        dm.request_llm()

        dc = DefectCleaner(root_path, llm_name, llm_dict)
        dc.formal_check(task_n)

        d2s = Defect2Styletext(root_path, target_style, llm_name, llm_dict, task_n, level_n, api_keys, sbert_path, top_k)
        d2s.request_llm()

        sc = StyleCleaner(root_path, llm_name, llm_dict)
        sc.formal_check(task_n)

        stc = StyleTextClassifier(root_path, task_n, level_n, llm_b_name)

        for r_n in range(2, 5):
            te.style_text_turn2emb(task_n, level_n, r_n=r_n-1)

            stc.round_sep(r_n=r_n-1)
            mis_test_keys, mis_train_keys, _, _ = stc.check_mis_ind()
            if (len(mis_test_keys) + len(mis_train_keys)) == 0:
                break

            d2s.add_request(r_n=r_n)
            sc.formal_check(task_n)
            if r_n == 4:
                te.style_text_turn2emb(task_n, level_n, r_n=r_n)
                stc.round_sep(r_n=r_n)

        mis_test_keys, mis_train_keys, percent_mis_test, percent_mis_train = stc.check_mis_ind()
        json.dump(mis_test_keys, open(f"./mis_key/{task_n}_{level_n}_{llm_dict[llm_name]}_test.json", "w"))
        json.dump(mis_train_keys, open(f"./mis_key/{task_n}_{level_n}_{llm_dict[llm_name]}_train.json", "w"))
        if min(percent_mis_test, percent_mis_train) < retention_rate:
            return None
        self.enter_next_round(root_path, task_n, level_n, llm_name, llm_dict, mis_test_keys, mis_train_keys)
        self.rank_file(task_n)
        self.check_rank(task_n)
        return None


if __name__ == "__main__":
    prep = Preprocessor()
    prep.main()




