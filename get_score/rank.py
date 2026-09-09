
import os
import json
import numpy as np

class Rank:
    def __init__(self, root_path, task_n):
        self.task_n = task_n
        self.root_path = root_path
        self.cur_path = root_path + "/baseline"

    def format_check(self, v_t):
        assert v_t in ["train", "test"]
        for file_name in os.listdir(self.cur_path + "/get_score/score"):
            if self.task_n not in file_name:
                continue
            if "all" not in file_name:
                continue

            file = json.load(open(self.cur_path + f"/get_score/score/{file_name}", "r", encoding="utf-8"))

            for ele in file:
                src_ind = ele["src_ind"]
                sent_list = ele["sent_list"]
                sbert_sim = ele["sbert_sim"]
                try:
                    score_output = json.loads(ele["score_output"])
                    assert len(sent_list) == len(score_output) == len(sbert_sim)
                    for ind in score_output.keys():
                        score = int(score_output[ind])

                except Exception as e:
                    print(e)
                    print(file_name)
                    print(src_ind)
                    print(ele["sentence"])
                pass

    def combine(self, v_t):
        assert v_t in ["train", "test"]
        save_name = f"{self.task_n}_{v_t}.json"
        save_path = self.cur_path + f"/get_score/score/{save_name}"
        if os.path.exists(save_path):
            return None

        ret_dict = {}
        for file_name in os.listdir(self.cur_path + "/get_score/score"):
            if self.task_n not in file_name:
                continue
            if "all" not in file_name:
                continue
            if v_t not in file_name:
                continue

            file = json.load(open(self.cur_path + f"/get_score/score/{file_name}", "r", encoding="utf-8"))

            for ele in file:
                src_ind = ele["src_ind"]
                sentence = ele["sentence"]
                sent_list = ele["sent_list"]
                sbert_sim = ele["sbert_sim"]
                score_output = json.loads(ele["score_output"])
                assert len(sent_list) == len(score_output) == len(sbert_sim)
                score_list = []
                for ind in range(len(sent_list)):  # 保证评分和候选文本列表长度一致
                    score_list.append(int(score_output[str(ind + 1)]))

                score_list = json.dumps(score_list, ensure_ascii=False)

                if src_ind not in ret_dict.keys():
                    ret_dict[src_ind] = {"sentence":sentence, "sent_list": sent_list, "sbert_sim": sbert_sim, "score": [score_list]}
                else:
                    ret_dict[src_ind]["score"].append(score_list)


        json.dump(ret_dict, open(save_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return None

    def _stable_order(self, score_lists):

        scores = np.array(score_lists)

        n = scores.shape[1]
        remaining = set(range(n))
        result = []

        while remaining:
            current_level = []

            for i in remaining:
                has_stable_predecessor = False

                for j in remaining:
                    if i == j:
                        continue

                    if np.all(scores[:, j] < scores[:, i]):
                        has_stable_predecessor = True
                        break

                if not has_stable_predecessor:
                    current_level.append(i)

            if len(current_level) == 1:
                result.append(current_level[0])
            else:
                result.append(current_level)

            remaining -= set(current_level)

        return result

    def _stable_order_V2(self, score_lists):
        scores = np.asarray(score_lists, dtype=float)
        n = scores.shape[1]

        stable = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(n):
                if i != j:
                    stable[i, j] = np.all(scores[:, i] < scores[:, j])

        mean_scores = scores.mean(axis=0)
        order = np.argsort(mean_scores)

        valid_cuts = []

        for k in range(1, n):
            left = order[:k]
            right = order[k:]

            valid = True

            for i in left:
                for j in right:
                    if not stable[i, j]:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                valid_cuts.append(k)

        boundaries = [0] + valid_cuts + [n]
        groups = []

        for a, b in zip(boundaries[:-1], boundaries[1:]):
            group = order[a:b].tolist()
            groups.append(group)

        return groups

    def _quantize_dense_levels(self, rank_list, score_lists, n):
        scores = np.array(score_lists, dtype=float)

        candidate_mean_scores = np.mean(scores, axis=0)

        target_scores = np.linspace(0, 100, n)

        quantized_levels = []
        sample_labels = {}

        for group in rank_list:

            if not isinstance(group, list):
                group = [group]

            level_score = np.mean([
                candidate_mean_scores[idx]
                for idx in group
            ])

            label = int(
                np.argmin(np.abs(target_scores - level_score))
            )

            quantized_levels.append({
                "label": label,
                "samples": group,
                "score": float(level_score)
            })

            for idx in group:
                sample_labels[idx] = label

        return quantized_levels, sample_labels

    def _merge_levels(self, groups, target_levels):
        L = len(groups)

        if L <= target_levels:
            return groups

        merged = [[] for _ in range(target_levels)]

        for i, group in enumerate(groups):
            idx = i * target_levels // L

            if isinstance(group, list):
                merged[idx].extend(group)
            else:
                merged[idx].append(group)

        return merged

    def _coarse_dist(self, l_dist, l_samples_label):
        l_labels = list(set([e for e in l_samples_label.values()]))
        if str(len(l_labels)) not in l_dist.keys():
            l_dist[str(len(l_labels))] = 1
        else:
            l_dist[str(len(l_labels))] += 1
        l_dist = {str(k): l_dist[str(k)] for k in sorted([int(e) for e in l_dist.keys()])}
        return l_dist


    def get_order(self, v_t):
        assert v_t in ["train", "test"]
        file_name = f"{self.task_n}_{v_t}.json"
        file_path = self.cur_path + f"/get_score/score/{file_name}"
        file = json.load(open(file_path, "r", encoding="utf-8"))

        l9_dist, l7_dist, l5_dist, l3_dist = {}, {}, {}, {}
        for src_ind in file.keys():
            if int(src_ind) % 10 == 0:
                print(src_ind, end= " ")
            score_text_list = file[src_ind]["score"]
            score_list = [json.loads(e) for e in score_text_list]
            rank_result = self._stable_order(score_list)

            l9_rank_result, l9_samples_label = self._quantize_dense_levels(rank_result, score_list, 9)
            l7_rank_result, l7_samples_label = self._quantize_dense_levels(rank_result, score_list, 7)
            l5_rank_result, l5_samples_label = self._quantize_dense_levels(rank_result, score_list, 5)
            l3_rank_result, l3_samples_label = self._quantize_dense_levels(rank_result, score_list, 3)

            file[src_ind]["fine_rank"] = json.dumps(rank_result)
            file[src_ind]["l9_rank"] = json.dumps(l9_rank_result)
            file[src_ind]["l9_rank_labels"] = json.dumps(l9_samples_label)
            file[src_ind]["l7_rank"] = json.dumps(l7_rank_result)
            file[src_ind]["l7_rank_labels"] = json.dumps(l7_samples_label)
            file[src_ind]["l5_rank"] = json.dumps(l5_rank_result)
            file[src_ind]["l5_rank_labels"] = json.dumps(l5_samples_label)
            file[src_ind]["l3_rank"] = json.dumps(l3_rank_result)
            file[src_ind]["l3_rank_labels"] = json.dumps(l3_samples_label)

            l9_dist = self._coarse_dist(l9_dist, l9_samples_label)
            l7_dist = self._coarse_dist(l7_dist, l7_samples_label)
            l5_dist = self._coarse_dist(l5_dist, l5_samples_label)
            l3_dist = self._coarse_dist(l3_dist, l3_samples_label)
        print("")
        print(f"l9_dist: {l9_dist}")
        print(f"l7_dist: {l7_dist}")
        print(f"l5_dist: {l5_dist}")
        print(f"l3_dist: {l3_dist}")


            # if len(rank_result) > 9:
            #     print(src_ind)
            #     # print(rank_result)

        json.dump(file, open(file_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    task_n = "task8"
    v_t = "test"
    root_path = r"G:\python_code\StyleTrans"
    rank_class = Rank(root_path, task_n)
    rank_class.format_check(v_t)
    # rank_class.combine(v_t)
    rank_class.get_order(v_t)

