
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class Statistic:
    def __init__(self, root_path, task_n):
        self.task_n = task_n
        self.root_path = root_path
        self.cur_path = root_path + "/baseline"
        pass

    def _coarse_dist(self, l_dist, l_samples_label):
        l_labels = list(set([e for e in l_samples_label.values()]))
        if str(len(l_labels)) not in l_dist.keys():
            l_dist[str(len(l_labels))] = 1
        else:
            l_dist[str(len(l_labels))] += 1
        l_dist = {str(k): l_dist[str(k)] for k in sorted([int(e) for e in l_dist.keys()])}
        return l_dist

    def convert_distribution(self, dist, use_percentage=True):

        values = []

        # 0~10
        for i in range(11):
            values.append(dist.get(str(i), 0))

        # >10
        over_10 = sum(
            count
            for level, count in dist.items()
            if int(level) > 10
        )
        values.append(over_10)

        if use_percentage:
            total = sum(dist.values())
            values = [
                value / total * 100
                for value in values
            ]

        return values

    def statis_1(self):
        file_name = f"{self.task_n}_train.json"
        file_path = self.cur_path + f"/get_score/score/{file_name}"
        file = json.load(open(file_path, "r", encoding="utf-8"))
        sent_list_num = []
        for src_ind in file.keys():
            ele = file[src_ind]
            sent_list = ele["sent_list"]
            sent_list_num.append(len(sent_list))

        print(f"{file_name}：{np.mean(sent_list_num)}, ：{np.std(sent_list_num)}")

    def dense_distributed(self):
        file_name = f"{self.task_n}_train.json"
        file_path = self.cur_path + f"/get_score/score/{file_name}"
        file = json.load(open(file_path, "r", encoding="utf-8"))

        fine_dist, l9_dist, l7_dist, l5_dist, l3_dist = {}, {}, {}, {}, {}
        for src_ind in file.keys():
            ele = file[src_ind]
            fine_rank_label = json.loads(ele["fine_rank"])
            if str(len(fine_rank_label)) not in fine_dist.keys():
                fine_dist[str(len(fine_rank_label))] = 1
            else:
                fine_dist[str(len(fine_rank_label))] += 1
            l9_rank_label = json.loads(ele["l9_rank_labels"])
            l7_rank_label = json.loads(ele["l7_rank_labels"])
            l5_rank_label = json.loads(ele["l5_rank_labels"])
            l3_rank_label = json.loads(ele["l3_rank_labels"])
            l9_dist = self._coarse_dist(l9_dist, l9_rank_label)
            l7_dist = self._coarse_dist(l7_dist, l7_rank_label)
            l5_dist = self._coarse_dist(l5_dist, l5_rank_label)
            l3_dist = self._coarse_dist(l3_dist, l3_rank_label)

        fine_dist = {str(k): fine_dist[str(k)] for k in sorted([int(e) for e in fine_dist.keys()])}
        print(f"fine_dist: {fine_dist}")
        print(f"l9_dist: {l9_dist}")
        print(f"l7_dist: {l7_dist}")
        print(f"l5_dist: {l5_dist}")
        print(f"l3_dist: {l3_dist}")

        data = {
            "Dense": self.convert_distribution(fine_dist),
            "9-level": self.convert_distribution(l9_dist),
            "7-level": self.convert_distribution(l7_dist),
            "5-level": self.convert_distribution(l5_dist),
            "3-level": self.convert_distribution(l3_dist),
        }

        categories = [
            "0", "1", "2", "3", "4", "5",
            "6", "7", "8", "9", "10", ">10"
        ]

        x = np.arange(len(categories))

        num_groups = len(data)
        width = 0.16

        fig, ax = plt.subplots(figsize=(11, 4.8))

        for idx, (name, values) in enumerate(data.items()):
            offset = (
                             idx - (num_groups - 1) / 2
                     ) * width

            ax.bar(
                x + offset,
                values,
                width,
                label=name
            )

        ax.set_xlabel("Number of Effective Stylistic Levels")
        ax.set_ylabel("Source Samples (%)")

        ax.set_xticks(x)
        ax.set_xticklabels(categories)

        ax.legend(
            ncol=5,
            frameon=False
        )

        ax.set_xlim(
            -0.6,
            len(categories) - 0.4
        )

        plt.tight_layout()
        plt.show()

    def main(self):


        pass


if __name__ == "__main__":
    root_path = r"G:\python_code\StyleTrans"
    task_n = "task1"
    statistic = Statistic(root_path, task_n)
    statistic.dense_distributed()