
import os
import json
import pickle
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset
from itertools import combinations
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support


def load_emb_model(emb_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(emb_model_path)
    model = AutoModel.from_pretrained(
        emb_model_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return tokenizer, model


class StylePairDataset(Dataset):

    def __init__(
        self,
        vec_list,
        hidden_dtype: torch.dtype = torch.float32,
    ):

        self.samples = []
        self.hidden_dtype = hidden_dtype

        for value in vec_list:
            hidden_states_1 = value[0]
            attention_mask_1 = value[1]
            text_1 = value[2]

            hidden_states_2 = value[3]
            attention_mask_2 = value[4]
            text_2 = value[5]

            label = value[6]
            sample_id = value[7]

            self.samples.append({
                "hidden_states_1": hidden_states_1,
                "attention_mask_1": attention_mask_1,
                "text_1": text_1,

                "hidden_states_2": hidden_states_2,
                "attention_mask_2": attention_mask_2,
                "text_2": text_2,

                "label": label,
                "sample_id": sample_id,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        # Text 1
        hidden_states_1 = item["hidden_states_1"]
        attention_mask_1 = item["attention_mask_1"]

        if not isinstance(hidden_states_1, torch.Tensor):
            hidden_states_1 = torch.tensor(
                hidden_states_1,
                dtype=self.hidden_dtype
            )
        else:
            hidden_states_1 = hidden_states_1.to(
                dtype=self.hidden_dtype
            )

        if not isinstance(attention_mask_1, torch.Tensor):
            attention_mask_1 = torch.tensor(
                attention_mask_1,
                dtype=torch.long
            )
        else:
            attention_mask_1 = attention_mask_1.long()

        # Text 2
        hidden_states_2 = item["hidden_states_2"]
        attention_mask_2 = item["attention_mask_2"]

        if not isinstance(hidden_states_2, torch.Tensor):
            hidden_states_2 = torch.tensor(
                hidden_states_2,
                dtype=self.hidden_dtype
            )
        else:
            hidden_states_2 = hidden_states_2.to(
                dtype=self.hidden_dtype
            )

        if not isinstance(attention_mask_2, torch.Tensor):
            attention_mask_2 = torch.tensor(
                attention_mask_2,
                dtype=torch.long
            )
        else:
            attention_mask_2 = attention_mask_2.long()

        # Output
        return {
            "hidden_states_1": hidden_states_1,
            "attention_mask_1": attention_mask_1,
            "text_1": item["text_1"],

            "hidden_states_2": hidden_states_2,
            "attention_mask_2": attention_mask_2,
            "text_2": item["text_2"],

            "label": torch.tensor(item["label"],dtype=torch.long),
            "sample_id": item["sample_id"],
        }


class StyleContrastiveModel(nn.Module):

    def __init__(
        self,
        hidden_size: int = 3072,
        max_len: int = 32,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        contrastive_margin: float = 1.0,
        direction_margin: float = 0.1,
        lambda_direction: float = 1.0,  # direction损失权重系数
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_len = max_len

        self.contrastive_margin = contrastive_margin
        self.direction_margin = direction_margin
        self.lambda_direction = lambda_direction

        # Shared encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def encode(self, hidden_states, attention_mask):
        padding_mask = attention_mask == 0
        encoded = self.transformer(
            hidden_states,
            src_key_padding_mask=padding_mask,
        )

        cls_vec = encoded[:, 0, :]
        return cls_vec

    def forward(self, hidden_states_1, attention_mask_1,
        hidden_states_2, attention_mask_2,
        labels: Optional[torch.Tensor] = None,
    ):
        device = next(self.parameters()).device

        hidden_states_1 = hidden_states_1.to(device)
        attention_mask_1 = attention_mask_1.to(device)

        hidden_states_2 = hidden_states_2.to(device)
        attention_mask_2 = attention_mask_2.to(device)

        # 1. Shared encoding
        z1_raw = self.encode(hidden_states_1, attention_mask_1)

        z2_raw = self.encode(hidden_states_2, attention_mask_2)

        z1 = F.normalize(z1_raw, p=2, dim=-1)
        z2 = F.normalize(z2_raw, p=2, dim=-1)
        distance = F.pairwise_distance(z1, z2, p=2)

        score1 = self.score_head(z1_raw).squeeze(-1)

        score2 = self.score_head(z2_raw).squeeze(-1)

        result = {
            "embedding_1": z1,
            "embedding_2": z2,
            "distance": distance,
            "score1": score1,
            "score2": score2,
        }

        if labels is None:
            return result

        labels = labels.to(device).long()

        # 4. Contrastive loss
        same_mask = labels == 0
        different_mask = labels != 0

        contrastive_losses = torch.zeros(labels.size(0), device=device)

        # Same pair: distance -> 0
        if same_mask.any():
            contrastive_losses[same_mask] = distance[same_mask] ** 2

        # Different pair: distance >= margin
        if different_mask.any():
            contrastive_losses[different_mask] = (
                torch.clamp(
                    self.contrastive_margin
                    - distance[different_mask],
                    min=0.0,
                ) ** 2
            )

        loss_contrastive = contrastive_losses.mean()

        # 5. Direction ranking loss
        if different_mask.any():
            diff_labels = labels[different_mask]
            diff_score1 = score1[different_mask]
            diff_score2 = score2[different_mask]

            # higher: score1 > score2
            # lower: score1 < score2
            ranking_target = torch.where(
                diff_labels == 1,
                torch.ones_like(diff_score1),
                -torch.ones_like(diff_score1),
            )

            loss_direction = F.margin_ranking_loss(
                diff_score1,
                diff_score2,
                ranking_target,
                margin=self.direction_margin,
                reduction="mean",
            )

        else:
            loss_direction = torch.tensor(0.0, device=device)

        # 6. Total loss
        loss = loss_contrastive + self.lambda_direction * loss_direction

        result.update({
            "loss": loss,
            "loss_contrastive": loss_contrastive,
            "loss_direction": loss_direction,
        })

        return result


class Turn2Emb:
    def __init__(self, root_path, tokenizer, emb_model):
        self.root_path = root_path
        self.cur_path = root_path + "/baseline"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model = emb_model
        self.model.eval()
        pass

    def _get_stable_relation(self, score_lists, idx1, idx2):

        scores_1 = [scores[idx1] for scores in score_lists]
        scores_2 = [scores[idx2] for scores in score_lists]

        if all(s1 > s2 for s1, s2 in zip(scores_1, scores_2)):
            return 1

        if all(s1 < s2 for s1, s2 in zip(scores_1, scores_2)):
            return -1

        return 0

    def encode_text_list_with_prefix(
            self,
            texts: list[str],
            label_ind: list[list[int]],
            src_ind: int,
            max_length: int = 32,
            prefix_token_id: int = 128000,
            tok_only: bool = False,
            score_lists:list[list[float]] = None,
            dense_flag: bool = False
    ):

        tokenizer = self.tokenizer
        model = self.model
        device = next(model.parameters()).device
        # 1. text index -> level index
        level_map = {}
        for level_idx, group in enumerate(label_ind):
            for text_idx in group:
                if text_idx in level_map:
                    raise ValueError(
                        f"text index {text_idx} appears in multiple levels."
                    )

                if text_idx < 0 or text_idx >= len(texts):
                    raise IndexError(
                        f"text index {text_idx} is out of range "
                        f"for texts with length {len(texts)}."
                    )

                level_map[text_idx] = level_idx

        missing_indices = [
            i for i in range(len(texts))
            if i not in level_map
        ]

        if missing_indices:
            raise ValueError(
                f"Some texts have no level label: {missing_indices}"
            )

        # 2. Tokenization
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length - 1,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        batch_size = input_ids.size(0)

        # 3. prefix token
        prefix_ids = torch.full(
            (batch_size, 1),
            prefix_token_id,
            dtype=input_ids.dtype,
        )

        prefix_mask = torch.ones(
            (batch_size, 1),
            dtype=attention_mask.dtype,
        )

        input_ids = torch.cat(
            [prefix_ids, input_ids],
            dim=1
        ).to(device)

        attention_mask = torch.cat(
            [prefix_mask, attention_mask],
            dim=1
        ).to(device)

        # 4. 获得hidden states
        model.eval()
        if not tok_only:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
            saved_data = outputs.last_hidden_state.detach().cpu()

        else:
            saved_data = input_ids.detach().cpu()

        attention_mask_cpu = attention_mask.detach().cpu()

        # 5. unordered pairs
        result_list = []
        for i, j in combinations(range(batch_size), 2):

            if random.random() < 0.5:
                idx1, idx2 = i, j
            else:
                idx1, idx2 = j, i

            level_1 = level_map[idx1]
            level_2 = level_map[idx2]

            if level_1 == level_2:
                pair_label = 0
            else:
                stable_relation = self._get_stable_relation(
                    score_lists,
                    idx1,
                    idx2
                )

                # unresolved cross-level pair
                if stable_relation == 0 and dense_flag:
                    continue

                if stable_relation == 1:
                    pair_label = 1  # Higher
                else:
                    pair_label = 2  # Lower

            result_list.append(
                (
                    saved_data[idx1], attention_mask_cpu[idx1], texts[idx1],
                    saved_data[idx2], attention_mask_cpu[idx2], texts[idx2],
                    pair_label,
                    src_ind
                )
            )

        return result_list

    def style_text_turn2tok(self, task_n:str):

        for v_t in ["test", "train"]:

            file_name = f"{task_n}_{v_t}.json"
            file_path = self.cur_path + f"/get_score/score/{file_name}"
            src_file = json.load(open(file_path, "r", encoding="utf-8"))

            emb_list = [[], [], [], []]
            emb_fine_list = []
            for src_ind in src_file.keys():
                if int(src_ind) % 100 == 0:
                    print(src_ind, end=" ")
                ele = src_file[src_ind]
                sentence = ele["sentence"]
                rewritten_text = ele["sent_list"]
                if len(rewritten_text) == 0:
                    continue

                for list_ind, layer in enumerate([3, 5, 7, 9]):

                    rank_list = []
                    l_label_dict = json.loads(ele[f"l{layer}_rank"])
                    for ly in range(0, 10):
                        cur_list = []
                        for l_e in l_label_dict:
                            if int(l_e["label"]) == ly:
                                cur_list.extend(l_e["samples"])
                        if len(cur_list) > 0:
                            rank_list.append(cur_list)

                    # return element list: last_hidden[i], attention_mask_cpu[i], batch_texts[i], src_ind
                    ind_list = self.encode_text_list_with_prefix(rewritten_text, rank_list, int(src_ind), tok_only=True)
                    emb_list[list_ind].extend(ind_list)

                score_list = [json.loads(e) for e in ele["score"]]
                fine_rank_list = json.loads(ele["fine_rank"])
                for level_idx, group in enumerate(fine_rank_list):
                    if not isinstance(group, list):
                        fine_rank_list[level_idx] = [group]

                ind_list = self.encode_text_list_with_prefix(rewritten_text, fine_rank_list, int(src_ind), tok_only=True,
                                                             score_lists=score_list, dense_flag=True)
                emb_fine_list.extend(ind_list)

            for list_ind, layer in enumerate([3, 5, 7, 9]):
                tok_name = f"{task_n}_{v_t}_{layer}.pkl"
                tok_path = self.cur_path + "/contrastive/tok/" + tok_name

                with open(tok_path, "wb") as file:
                    pickle.dump(emb_list[list_ind], file)

            tok_name = f"{task_n}_{v_t}_fine.pkl"
            tok_path = self.cur_path + "/contrastive/tok/" + tok_name

            with open(tok_path, "wb") as file:
                pickle.dump(emb_fine_list, file)


class TrainTest:
    def __init__(self, root_path, task_n, layer_num, emb_model, train_batch:int=32, tok_only=True):
        self.root_path = root_path
        self.task_n = task_n
        self.layer_n = layer_num
        self.cur_path = self.root_path + "/baseline"
        self.tok_only = tok_only
        self.train_batch = train_batch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emb_model = emb_model
        self.emb_model.eval()

    def _evaluate_p_r_f1(self, all_label, all_distance, all_score1, all_score2, eva_only=False):

        if not isinstance(all_label, torch.Tensor):
            all_label = torch.tensor(all_label, dtype=torch.long)

        if not isinstance(all_distance, torch.Tensor):
            all_distance = torch.tensor(
                all_distance,
                dtype=torch.float32
            )

        if not isinstance(all_score1, torch.Tensor):
            all_score1 = torch.tensor(
                all_score1,
                dtype=torch.float32
            )

        if not isinstance(all_score2, torch.Tensor):
            all_score2 = torch.tensor(
                all_score2,
                dtype=torch.float32
            )

        all_label = all_label.cpu()
        all_distance = all_distance.cpu()
        all_score1 = all_score1.cpu()
        all_score2 = all_score2.cpu()

        best_result = None
        best_macro_f1 = -1.0

        thresholds = np.arange(0.0,2.001,0.02)

        record_file = open(f"./{self.task_n}_record_file_{self.layer_n}.txt", "w", encoding="utf-8")
        for the in thresholds:
            preds = torch.zeros_like(all_label)
            # Same
            same_mask = all_distance < the
            preds[same_mask] = 0
            # Same
            different_mask = ~same_mask
            # text1 > text2
            higher_mask = different_mask & (all_score1 > all_score2)
            # text1 < text2
            lower_mask = different_mask & (all_score1 <= all_score2)

            preds[higher_mask] = 1
            preds[lower_mask] = 2

            # P / R / F1
            precision, recall, f1, support = (
                precision_recall_fscore_support(
                    all_label.numpy(),
                    preds.numpy(),
                    labels=[0, 1, 2],
                    zero_division=0
                )
            )

            macro_p = precision.mean()
            macro_r = recall.mean()
            macro_f1 = f1.mean()

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1

                best_result = {
                    "threshold": float(the),
                    "same": {
                        "precision": float(precision[0]),
                        "recall": float(recall[0]),
                        "f1": float(f1[0]),
                        "support": int(support[0]),
                    },
                    "higher": {
                        "precision": float(precision[1]),
                        "recall": float(recall[1]),
                        "f1": float(f1[1]),
                        "support": int(support[1]),
                    },
                    "lower": {
                        "precision": float(precision[2]),
                        "recall": float(recall[2]),
                        "f1": float(f1[2]),
                        "support": int(support[2]),
                    },
                    "macro": {
                        "precision": float(macro_p),
                        "recall": float(macro_r),
                        "f1": float(macro_f1),
                    }
                }
            if eva_only:
                same_line = f"{precision[0]}-{recall[0]}-{f1[0]}"
                higher_line = f"{precision[1]}-{recall[1]}-{f1[1]}"
                lower_line = f"{precision[2]}-{recall[2]}-{f1[2]}"
                record_file.write(f"threshold:{the}-{same_line}-{higher_line}-{lower_line}-{macro_f1}\n")
                pass

        record_file.close()

        return best_result

    def evaluate(self, model, test_loader, epoch, num_epochs, device, eva_only=False):
        model.eval()
        all_true = []
        all_distance = []
        all_score1 = []
        all_score2 = []
        test_num = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluate Epoch {epoch + 1}/{num_epochs}"):
                test_num += 1
                # if test_num >= 20:
                #     break
                # Text 1
                hidden_states_1 = batch["hidden_states_1"].to(device)
                attention_mask_1 = batch["attention_mask_1"].to(device)

                # Text 2
                hidden_states_2 = batch["hidden_states_2"].to(device)
                attention_mask_2 = batch["attention_mask_2"].to(device)

                if self.tok_only:
                    with torch.no_grad():
                        outputs_1 = self.emb_model(
                            input_ids=hidden_states_1.long(),
                            attention_mask=attention_mask_1,
                            output_hidden_states=False,
                        )
                        hidden_states_1 = outputs_1.last_hidden_state.to(torch.float32).to(device)

                        outputs_2 = self.emb_model(
                            input_ids=hidden_states_2.long(),
                            attention_mask=attention_mask_2,
                            output_hidden_states=False,
                        )
                        hidden_states_2 = outputs_2.last_hidden_state.to(torch.float32).to(device)

                else:
                    hidden_states_1 = hidden_states_1.to(torch.float32)
                    hidden_states_2 = hidden_states_2.to(torch.float32)

                # Labels 0 = same 1 = higher 2 = lower
                labels = batch["label"].to(device)

                # Forward
                outputs = model(
                    hidden_states_1=hidden_states_1,
                    attention_mask_1=attention_mask_1,
                    hidden_states_2=hidden_states_2,
                    attention_mask_2=attention_mask_2,
                    labels=labels
                )

                loss = outputs["loss"]
                loss_contrastive = outputs["loss_contrastive"]
                loss_direction = outputs["loss_direction"]

                distance = outputs["distance"]  # [B]
                score1 = outputs["score1"]  # [B]
                score2 = outputs["score2"]  # [B]
                labels = batch["label"].to(device)

                all_true.extend(labels.cpu().tolist())
                all_distance.extend(distance.cpu().tolist())
                all_score1.extend(score1.cpu().tolist())
                all_score2.extend(score2.cpu().tolist())

            best_result = self._evaluate_p_r_f1(all_true, all_distance, all_score1, all_score2, eva_only=eva_only)

            return best_result

    def train(
            self,
            train_list,
            test_list,
            pth_save_name: str,
            num_epochs: int = 15
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Dataset / DataLoader
        test_dataset = StylePairDataset(test_list)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        train_dataset = StylePairDataset(train_list)
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch, shuffle=True)

        # Model
        model = StyleContrastiveModel(
            hidden_size=3072, max_len=32, num_heads=8,
            contrastive_margin=1.0, direction_margin=0.0, lambda_direction=1.0,
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

        best_macro_f1 = 0.0
        best_epoch = 0
        best_result = {}
        # Training
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_contrastive_loss = 0.0
            total_direction_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                # Text 1
                hidden_states_1 = batch["hidden_states_1"].to(device)
                attention_mask_1 = batch["attention_mask_1"].to(device)

                # Text 2
                hidden_states_2 = batch["hidden_states_2"].to(device)
                attention_mask_2 = batch["attention_mask_2"].to(device)

                if self.tok_only:
                    with torch.no_grad():
                        outputs_1 = self.emb_model(
                            input_ids=hidden_states_1.long(),
                            attention_mask=attention_mask_1,
                            output_hidden_states=False,
                        )
                        hidden_states_1 = outputs_1.last_hidden_state.to(torch.float32).to(device)

                        outputs_2 = self.emb_model(
                            input_ids=hidden_states_2.long(),
                            attention_mask=attention_mask_2,
                            output_hidden_states=False,
                        )
                        hidden_states_2 = outputs_2.last_hidden_state.to(torch.float32).to(device)

                else:
                    hidden_states_1 = hidden_states_1.to(torch.float32)
                    hidden_states_2 = hidden_states_2.to(torch.float32)

                # Labels 0 = same 1 = higher 2 = lower
                labels = batch["label"].to(device)

                # Forward
                outputs = model(
                    hidden_states_1=hidden_states_1,
                    attention_mask_1=attention_mask_1,
                    hidden_states_2=hidden_states_2,
                    attention_mask_2=attention_mask_2,
                    labels=labels
                )

                loss = outputs["loss"]

                loss_contrastive = outputs["loss_contrastive"]
                loss_direction = outputs["loss_direction"]

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Statistics
                total_loss += loss.item()
                total_contrastive_loss += loss_contrastive.item()
                total_direction_loss += loss_direction.item()

            # Epoch statistics
            avg_loss = total_loss /len(train_loader)
            avg_contrastive_loss = total_contrastive_loss /len(train_loader)
            avg_direction_loss = total_direction_loss /len(train_loader)

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"loss={avg_loss:.4f} | "
                f"contrastive={avg_contrastive_loss:.4f} | "
                f"direction={avg_direction_loss:.4f}"
            )

            best_result = self.evaluate(model, test_loader, epoch, num_epochs, device)
            print(best_result)

            # if best_result["macro"]["f1"] >= best_macro_f1:
            #     best_macro_f1 = best_result["macro"]["f1"]
            #     best_epoch = epoch
            torch.save(model.state_dict(), self.cur_path + f"/contrastive/pth/{pth_save_name}")

        return model, best_epoch, best_result


    def train_main(self, num_epochs=10):

        pth_save_name = f"{self.task_n}_{self.layer_n}_best_classifier.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.exists(self.cur_path + f"/contrastive/pth/{pth_save_name}"):
        # if True:
            print(f"TrainTest.train_main: {pth_save_name} 已存在")
            test_path = self.cur_path + f"/contrastive/tok/{self.task_n}_test_{self.layer_n}.pkl"
            test_data = pickle.load(open(test_path, "rb"))
            test_dataset = StylePairDataset(test_data)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            model = StyleContrastiveModel(
                hidden_size=3072, max_len=32, num_heads=8,
                contrastive_margin=1.0, direction_margin=0.0, lambda_direction=1.0,
            ).to(device)
            model.load_state_dict(torch.load(self.cur_path + f"/contrastive/pth/{pth_save_name}"))
            result = self.evaluate(model, test_loader, 0, 0, device, eva_only=True)
            print(result)
            return result

        test_set, train_set = [], []
        for v_t in ["test", "train"]:
            data_path = self.cur_path + f"/contrastive/tok/{self.task_n}_{v_t}_{self.layer_n}.pkl"
            data = pickle.load(open(data_path, "rb"))
            if v_t == "test":
                test_set = data
            elif v_t == "train":
                train_set = data

        trained_model, best_epoch, best_result = self.train(train_set, test_set, pth_save_name, num_epochs=num_epochs)
        print(f"best_epoch={best_epoch}")
        return best_result


if __name__ == "__main__":
    task_n = "task1"
    root_path = r"/root/private_data/StyleTrans"
    # llm_b_name = "dsv4f"
    emb_model_path = r"/root/private_data/Llama-3.2-3B-Instruct"
    tokenizer, model = load_emb_model(emb_model_path)
    # te = Turn2Emb(root_path, tokenizer, model)
    # te.style_text_turn2tok(task_n)

    layer_num = 9
    train_batch = 128

    for layer_num in ["fine"]:
        print(f"{layer_num}")
        tt = TrainTest(root_path, task_n, layer_num, model, train_batch)
        tt.train_main(num_epochs=3)
        pass

