import os
import json
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from itertools import combinations
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

class StyleDataset(Dataset):

    def __init__(
            self,
            vec_list: list[tuple[torch.Tensor, torch.Tensor, list[str], list[int], int]],
            hidden_dtype: torch.dtype = torch.float32,
    ):
        self.samples = []
        self.hidden_dtype = hidden_dtype

        for value in vec_list:
            hidden_states, attention_mask, src_text, sample_label, sample_id = value
            self.samples.append({
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "text": src_text,
                "label": sample_label,
                "sample_id": sample_id
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        hidden_states = item["hidden_states"]
        attention_mask = item["attention_mask"]

        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = torch.tensor(hidden_states)

        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)

        return {
            "hidden_states": hidden_states,      # [seq_len, hidden_size]
            "attention_mask": attention_mask.long(),                   # [seq_len]
            "text": item["text"],
            "label": torch.tensor(item["label"], dtype=torch.long),    # []
            "sample_id": item["sample_id"]
        }


class StyleClassifier(nn.Module):

    def __init__(
        self,
        hidden_size: int = 3072,
        max_len: int = 32,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_len = max_len

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

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        labels: Optional[torch.Tensor] = None,
    ):
        device = next(self.parameters()).device

        padding_mask = attention_mask == 0

        encoded = self.transformer(
            hidden_states,
            src_key_padding_mask=padding_mask,
        )

        cls_vec = encoded[:, 0, :]

        logits = self.classifier(cls_vec)

        if labels is not None:
            labels = labels.to(device)
            loss = F.cross_entropy(logits, labels)
            return {
                "loss": loss,
                "logits": logits,
                "probs": torch.softmax(logits, dim=-1),
            }

        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
        }


class Turn2Emb:
    def __init__(self, root_path, tokenizer, emb_model):
        self.root_path = root_path
        self.cur_path = root_path + "/baseline"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model = emb_model
        self.model.eval()
        pass

    def encode_text_list_with_prefix(self, texts: list[str], label_ind: list, src_ind: int, max_length: int = 32,
                                     prefix_token_id: int = 128000, tok_only=False):

        assert len(texts) == len(label_ind)
        tokenizer = self.tokenizer
        model = self.model

        device = next(model.parameters()).device

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        result_list = []

        batch_texts = texts

        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length - 1,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        batch_size = input_ids.size(0)

        prefix_ids = torch.full(
            (batch_size, 1),
            prefix_token_id,
            dtype=input_ids.dtype,
        )

        prefix_mask = torch.ones(
            (batch_size, 1),
            dtype=attention_mask.dtype,
        )

        input_ids = torch.cat([prefix_ids, input_ids], dim=1).to(device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1).to(device)

        model.eval()
        if not tok_only:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )

            # [batch_size, max_length, hidden_size]
            saved_data = outputs.last_hidden_state.detach().cpu()
        else:
            saved_data = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()

        for i in range(batch_size):
            result_list.append((saved_data[i], attention_mask_cpu[i], batch_texts[i], label_ind[i], src_ind))

        return result_list

    def style_text_turn2tok(self, task_n:str):

        for v_t in ["test", "train"]:

            file_name = f"{task_n}_{v_t}.json"
            file_path = self.cur_path + f"/get_score/score/{file_name}"
            src_file = json.load(open(file_path, "r", encoding="utf-8"))

            emb_list = [[], [], [], []]
            for src_ind in src_file.keys():
                if int(src_ind) % 100 == 0:
                    print(src_ind, end=" ")
                ele = src_file[src_ind]
                sentence = ele["sentence"]
                rewritten_text = ele["sent_list"]
                if len(rewritten_text) == 0:
                    continue

                for list_ind, layer in enumerate([3, 5, 7, 9]):

                    l_label_dict = json.loads(ele[f"l{layer}_rank_labels"])
                    l_label_list = []
                    for i in range(len(l_label_dict)):
                        l_label_list.append(l_label_dict[str(i)])

                    # return element list: last_hidden[i], attention_mask_cpu[i], batch_texts[i], src_ind
                    ind_list = self.encode_text_list_with_prefix(rewritten_text, l_label_list, int(src_ind), tok_only=True)
                    emb_list[list_ind].extend(ind_list)

            for list_ind, layer in enumerate([3, 5, 7, 9]):
                tok_name = f"{task_n}_{v_t}_{layer}.pkl"
                tok_path = self.cur_path + "/multi_class/tok/" + tok_name

                with open(tok_path, "wb") as file:
                    pickle.dump(emb_list[list_ind], file)


class TrainTest:
    def __init__(self, root_path, emb_model, train_batch:int=32, tok_only=True):
        self.root_path = root_path
        self.cur_path = self.root_path + "/baseline"
        self.tok_only = tok_only
        self.train_batch = train_batch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emb_model = emb_model
        self.emb_model.eval()

    def _evaluate_pairwise_relations(self, gold_labels, pred_labels, sample_ids):


        relation_labels = ["same", "higher", "lower"]
        y_true = []
        y_pred = []

        grouped_indices = {}
        for idx, sample_id in enumerate(sample_ids):
            grouped_indices.setdefault(sample_id, []).append(idx)

        for indices in grouped_indices.values():
            for i, j in combinations(indices, 2):
                # Gold relation
                if gold_labels[i] == gold_labels[j]:
                    true_relation = "same"
                elif gold_labels[i] > gold_labels[j]:
                    true_relation = "higher"
                else:
                    true_relation = "lower"
                # Predicted relation
                if pred_labels[i] == pred_labels[j]:
                    pred_relation = "same"
                elif pred_labels[i] > pred_labels[j]:
                    pred_relation = "higher"
                else:
                    pred_relation = "lower"
                y_true.append(true_relation)
                y_pred.append(pred_relation)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=relation_labels,
            zero_division=0
        )

        result = {}
        for label, p, r, f, n in zip(relation_labels, precision, recall, f1, support):
            result[label] = {"precision": p, "recall": r, "f1": f, "support": n}

        result["macro"] = {"precision": precision.mean(), "recall": recall.mean(), "f1": f1.mean()}

        return result

    def train(self, train_list, test_list, layer_n:int, pth_save_name:str, num_epochs=15):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        test_dataset = StyleDataset(test_list)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        train_dataset = StyleDataset(train_list)
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch, shuffle=True)

        model = StyleClassifier(num_classes=layer_n, hidden_size=3072, max_len=32, num_heads=8).to(device)
        optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

        num_epochs = num_epochs
        best_macro_f1 = 0.0
        best_epoch = 0
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                hidden_states = batch["hidden_states"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                if self.tok_only:
                    outputs = self.emb_model(
                        input_ids=hidden_states,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                    )
                    # [batch_size, max_length, hidden_size]
                    hidden_states = outputs.last_hidden_state.to(torch.float32).to(self.device)
                else:
                    hidden_states = hidden_states.to(torch.float32)

                labels = batch["label"].to(device)

                outputs = model(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs["loss"]
                logits = outputs["logits"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_count += labels.size(0)

            avg_loss = total_loss / len(train_loader)
            acc = total_correct / total_count

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train loss={avg_loss:.4f} | acc={acc:.4f}"
            )

            model.eval()
            all_gold = []
            all_pred = []
            all_sample_ids = []

            for batch in test_loader:
                hidden_states = batch["hidden_states"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                if self.tok_only:
                    outputs = self.emb_model(
                        input_ids=hidden_states,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                    )
                    # [batch_size, max_length, hidden_size]
                    hidden_states = outputs.last_hidden_state.to(torch.float32).to(self.device)
                else:
                    hidden_states = hidden_states.to(torch.float32)

                labels = batch["label"].to(device)
                outputs = model(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"]
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)

                all_gold.extend(labels.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())

                all_sample_ids.extend(batch["sample_id"].cpu().tolist())

            relation_result = self._evaluate_pairwise_relations(all_gold, all_pred, all_sample_ids)

            torch.save(model.state_dict(), self.cur_path + f"/multi_class/pth/{pth_save_name}")

        print(f"best_macro_f1{best_macro_f1:.4f} | best_epoch：{best_epoch}")

        return model, best_epoch, best_macro_f1

    def evaluate(self, test_list, layer_n:int, pth_save_name:str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_dataset = StyleDataset(test_list)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        model = StyleClassifier(num_classes=layer_n, hidden_size=3072, max_len=32, num_heads=8).to(device)
        model.load_state_dict(torch.load(self.cur_path + f"/multi_class/pth/{pth_save_name}"))
        model.eval()

        all_gold = []
        all_pred = []
        all_sample_ids = []
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for batch in test_loader:
                hidden_states = batch["hidden_states"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                if self.tok_only:
                    outputs = self.emb_model(
                        input_ids=hidden_states,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                    )
                    # [batch_size, max_length, hidden_size]
                    hidden_states = outputs.last_hidden_state.to(torch.float32).to(self.device)
                else:
                    hidden_states = hidden_states.to(torch.float32)
                labels = batch["label"].to(device)
                outputs = model(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"]
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_count += labels.size(0)

                all_gold.extend(labels.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())

                all_sample_ids.extend(batch["sample_id"].cpu().tolist())

            acc = total_correct / total_count

        relation_result = self._evaluate_pairwise_relations(all_gold, all_pred, all_sample_ids)

        print(relation_result)

        print("evaluate acc:{}".format(acc))

        return relation_result

    def train_main(self, task_n:str, layer_n:int, num_epochs=15):

        pth_save_name = f"{task_n}_{layer_n}_best_classifier.pth"
        if os.path.exists(self.cur_path + f"/multi_class/pth/{pth_save_name}"):
            print(f"TrainTest.train_main: {pth_save_name} exist")
            test_path = self.cur_path + f"/multi_class/tok/{task_n}_test_{layer_n}.pkl"
            test_data = pickle.load(open(test_path, "rb"))
            relation_result = self.evaluate(test_data, layer_n, pth_save_name)
            return relation_result

        test_set, train_set = [], []
        for v_t in ["test", "train"]:
            data_path = self.cur_path + f"/multi_class/tok/{task_n}_{v_t}_{layer_n}.pkl"
            data = pickle.load(open(data_path, "rb"))
            if v_t == "test":
                test_set = data
            elif v_t == "train":
                train_set = data

        trained_model, best_epoch, best_macro_f1 = self.train(train_set, test_set, layer_n, pth_save_name, num_epochs=num_epochs)

        print(f"best_epoch={best_epoch}")

        return best_macro_f1


if __name__ == "__main__":
    task_n = "task8"
    root_path = r"G:\python_code\StyleTrans"
    # llm_b_name = "dsv4f"
    emb_model_path = r"G:\python_code\pre_model\Llama-3.2-3B-Instruct"
    tokenizer, model = load_emb_model(emb_model_path)
    # te = Turn2Emb(root_path, tokenizer, model)
    # te.style_text_turn2tok(task_n)

    layer_num = 9  # 粗粒度指定
    train_batch = 32
    
    tt = TrainTest(root_path, model, train_batch)
    tt.train_main(task_n, layer_num)


