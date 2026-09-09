
from sentence_transformers import SentenceTransformer, util
import torch
import json
from itertools import combinations, product
from sklearn.metrics import precision_recall_fscore_support, f1_score


class SbertModel:
    def __init__(self, root_path, task_n):
        self.task_n = task_n
        self.root_path = root_path
        self.cur_path = root_path + "/baseline"
        self.model = SentenceTransformer(r"G:\python_code\pre_model\all-mpnet-base-v2")

    def _get_stable_relation(self, score_lists, idx1, idx2):

        scores_1 = [scores[idx1] for scores in score_lists]
        scores_2 = [scores[idx2] for scores in score_lists]

        if all(s1 > s2 for s1, s2 in zip(scores_1, scores_2)):
            return 1

        if all(s1 < s2 for s1, s2 in zip(scores_1, scores_2)):
            return -1

        return 0

    def _evaluate_sbert_pairs(self, embeddings, source_embedding, rank_list, score_lists=None, same_threshold=0.85):

        level_map = {}
        for level_idx, group in enumerate(rank_list):
            if not isinstance(group, list):
                group = [group]

            for sample_idx in group:
                level_map[sample_idx] = level_idx

        source_sims = [
            util.cos_sim(emb, source_embedding).item()
            for emb in embeddings
        ]

        y_true = []
        y_pred = []
        records = []

        sample_indices = sorted(level_map.keys())
        for i, j in combinations(sample_indices, 2):

            # ===== Gold label =====
            level_i = level_map[i]
            level_j = level_map[j]

            if level_i == level_j:
                gold = "same"
            else:
                stable_relation = self._get_stable_relation(
                    score_lists,
                    level_i,
                    level_j
                )

                # unresolved cross-level pair
                if stable_relation == 0:
                    continue

                if stable_relation == 1:
                    gold = "higher"
                else:
                    gold = "lower"

            # ===== SBERT prediction =====
            pair_sim = util.cos_sim(embeddings[i], embeddings[j]).item()

            if pair_sim >= same_threshold:
                pred = "same"

            else:
                pred = "higher" if source_sims[i] < source_sims[j] else "lower"

            y_true.append(gold)
            y_pred.append(pred)

            records.append({"pair": (i, j), "gold": gold, "pred": pred,
                "pair_similarity": pair_sim,
                "source_sim_i": source_sims[i],
                "source_sim_j": source_sims[j]
            })

        return y_true, y_pred, records

    def _turn2coarse_rank(self, coarse_rank_dict):
        ret_list = []
        for i in range(20):
            cur_list = []
            for ele in coarse_rank_dict:
                label = int(ele["label"])
                if label == i:
                    cur_list.extend(ele["samples"])
            if len(cur_list) > 0:
                ret_list.append(cur_list)

        return ret_list

    def p_r_f1(self, all_true, all_pred):
        labels = ["same", "higher", "lower"]

        precision, recall, f1, support = precision_recall_fscore_support(
            all_true,
            all_pred,
            labels=labels,
            zero_division=0
        )

        ret_dict = {}
        for label, p, r, f, n in zip(labels, precision, recall, f1, support):
            ret_dict[label] = {"precision": p, "recall": r, "f1": f, "n":n}

        # Macro-F1
        macro_f1 = f1_score(all_true,all_pred, labels=labels, average="macro", zero_division=0)
        ret_dict["macro_f1"] = macro_f1

        return ret_dict

    def print_fun(self, result):
        labels = ["same", "higher", "lower"]
        same_score = f"{result["same"]["precision"]}-{result["same"]["recall"]}-{result["same"]["f1"]}"
        higher_score = f"{result["higher"]["precision"]}-{result["higher"]["recall"]}-{result["higher"]["f1"]}"
        lower_score = f"{result["lower"]["precision"]}-{result["lower"]["recall"]}-{result["lower"]["f1"]}"
        macro_f1 = result["macro_f1"]
        print(f"{same_score}-{higher_score}-{lower_score}-{macro_f1}")

    def main(self):
        file_name = f"{self.task_n}_test.json"
        file_path = self.cur_path + f"/get_score/score/{file_name}"
        file = json.load(open(file_path, "r", encoding="utf-8"))
        threshold = 0.6
        for the in range(20):
            threshold = 0.6 + the * 0.02

            all_true_fine, all_pred_fine = [], []
            all_true_l9, all_pred_l9 = [], []
            all_true_l7, all_pred_l7 = [], []
            all_true_l5, all_pred_l5 = [], []
            all_true_l3, all_pred_l3 = [], []
            for src_ind in file:
                score_list = [json.loads(e) for e in file[src_ind]["score"]]
                sentence = file[src_ind]["sentence"]
                sent_list = file[src_ind]["sent_list"]
                fine_rank = json.loads(file[src_ind]["fine_rank"])
                l9_rank = self._turn2coarse_rank(json.loads(file[src_ind]["l9_rank"]))
                l7_rank = self._turn2coarse_rank(json.loads(file[src_ind]["l7_rank"]))
                l5_rank = self._turn2coarse_rank(json.loads(file[src_ind]["l5_rank"]))
                l3_rank = self._turn2coarse_rank(json.loads(file[src_ind]["l3_rank"]))

                embeddings = self.model.encode(
                    sent_list,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
                source_embedding = self.model.encode(
                    [sentence],
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )[0]

                y_true, y_pred, _ = self._evaluate_sbert_pairs(embeddings, source_embedding, fine_rank, score_list, threshold)
                all_true_fine.extend(y_true)
                all_pred_fine.extend(y_pred)
                y_true, y_pred, _ = self._evaluate_sbert_pairs(embeddings, source_embedding, l9_rank, threshold)
                all_true_l9.extend(y_true)
                all_pred_l9.extend(y_pred)
                y_true, y_pred, _ = self._evaluate_sbert_pairs(embeddings, source_embedding, l7_rank, threshold)
                all_true_l7.extend(y_true)
                all_pred_l7.extend(y_pred)
                y_true, y_pred, _ = self._evaluate_sbert_pairs(embeddings, source_embedding, l5_rank, threshold)
                all_true_l5.extend(y_true)
                all_pred_l5.extend(y_pred)
                y_true, y_pred, _ = self._evaluate_sbert_pairs(embeddings, source_embedding, l3_rank, threshold)
                all_true_l3.extend(y_true)
                all_pred_l3.extend(y_pred)

            all_result = self.p_r_f1(all_true_fine, all_pred_fine)
            l9_result = self.p_r_f1(all_true_l9, all_pred_l9)
            l7_result = self.p_r_f1(all_true_l7, all_pred_l7)
            l5_result = self.p_r_f1(all_true_l5, all_pred_l5)
            l3_result = self.p_r_f1(all_true_l3, all_pred_l3)


            print(f"{threshold}，fine, 9, 7, 5, 3")
            self.print_fun(all_result)
            self.print_fun(l9_result)
            self.print_fun(l7_result)
            self.print_fun(l5_result)
            self.print_fun(l3_result)


if __name__ == '__main__':
    task_n = "task8"
    root_path = r"G:\python_code\StyleTrans"
    smodel = SbertModel(root_path, task_n)
    smodel.main()

