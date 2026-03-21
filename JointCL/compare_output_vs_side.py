import os
import re
import torch
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModel

from models.bert_scl_prototype_graph import BERT_SCL_Proto_Graph

# ============================================================
# PATHS
# ============================================================
INPUT_CSV = "/mnt/c/Users/16248/Desktop/大学课程/541 Practical Code/summaries_output_V3__01_15_2025.csv"
CHECKPOINT_PATH = "./outputs/custom_bert-scl-prototype-graph_2026-03-17_23-58-36/state_dict/best_f1_model.bin"

OUTPUT_PIECE_CSV = "./jointcl_output_vs_side_piece_results.csv"
OUTPUT_ROW_CSV = "./jointcl_output_vs_side_row_results.csv"
OUTPUT_TXT = "./jointcl_output_vs_side_summary.txt"

# ============================================================
# SETTINGS
# ============================================================
PRETRAINED_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
SPLIT_MODE = "row"   # row / paragraph / sentence

# This checkpoint is 3-class.
# Most likely order is left / right / neutral, but if results look wrong,
# we may need to swap the mapping.
ID2LABEL = {
    0: "left",
    1: "right",
    2: "neutral",
}

# ============================================================
# HELPERS
# ============================================================
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()

def normalize_label(x):
    x = clean_text(x).lower()
    if x in {"left", "right", "neutral"}:
        return x
    return None

def split_into_sentences(text):
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]

def split_into_paragraphs(text):
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]

def split_output(text, mode="row"):
    text = clean_text(text)
    if not text:
        return []
    if mode == "row":
        return [text]
    elif mode == "paragraph":
        return split_into_paragraphs(text)
    elif mode == "sentence":
        return split_into_sentences(text)
    return [text]

def majority_vote(labels):
    if not labels:
        return None
    c = Counter(labels)
    max_count = max(c.values())
    winners = sorted([k for k, v in c.items() if v == max_count])
    return winners[0]

# ============================================================
# OPTION OBJECT
# ============================================================
class DummyOpt:
    def __init__(self):
        self.pretrained_bert_name = PRETRAINED_MODEL_NAME
        self.bert_dim = 768
        self.hidden_dim = 768
        self.dropout = 0.1
        self.polarities_dim = 3
        self.num_labels = 3
        self.max_seq_len = MAX_LEN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # required by prototype-graph model
        self.graph_num = 1
        self.layer_num = 2
        self.num_layers = 2
        self.top_k = 5
        self.temperature = 0.07
        self.proto_m = 0.99
        self.leakyrelu_alpha = 0.2
        self.use_cuda = torch.cuda.is_available()

        self.gnn_dims = "192,192"
        self.att_heads = "4,4"
        self.dp = 0.1
# ============================================================
# TOKENIZATION
# ============================================================
def prepare_inputs(tokenizer, target, text, max_len=128):
    target = clean_text(target)
    text = clean_text(text)

    encoded = tokenizer(
        target,
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))

    return input_ids, token_type_ids, attention_mask

# ============================================================
# MODEL LOADING
# ============================================================
def build_bert_backbone():
    print(f"[INFO] Loading backbone: {PRETRAINED_MODEL_NAME}")
    return AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)

def load_model():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"CSV not found: {INPUT_CSV}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    opt = DummyOpt()
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    bert = build_bert_backbone()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=opt.device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    print("[INFO] Checkpoint type:", type(checkpoint))
    print("[INFO] Total checkpoint keys:", len(state_dict))

    non_bert = [k for k in state_dict.keys() if not k.startswith("bert.")]
    print("[INFO] Non-bert checkpoint keys:", len(non_bert))
    for k in non_bert[:40]:
        print("   ", k)

    model = BERT_SCL_Proto_Graph(opt, bert)
    print("[INFO] Constructed BERT_SCL_Proto_Graph(opt, bert)")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[INFO] Loaded checkpoint into BERT_SCL_Proto_Graph")
    if missing:
        print(f"[INFO] Missing keys count: {len(missing)}")
        print("[INFO] Missing keys sample:", missing[:30])
    if unexpected:
        print(f"[INFO] Unexpected keys count: {len(unexpected)}")
        print("[INFO] Unexpected keys sample:", unexpected[:30])

    model = model.to(opt.device)
    model.eval()
    return model, tokenizer, opt

# ============================================================
# PREDICTION
# ============================================================
@torch.no_grad()
def predict_one(model, tokenizer, opt, target, text):
    input_ids, token_type_ids, attention_mask = prepare_inputs(
        tokenizer=tokenizer,
        target=target,
        text=text,
        max_len=opt.max_seq_len
    )

    input_ids = input_ids.to(opt.device)
    token_type_ids = token_type_ids.to(opt.device)
    attention_mask = attention_mask.to(opt.device)

    logits = None
    forward_errors = []

    attempts = [
        ("model([input_ids, token_type_ids])", lambda: model([input_ids, token_type_ids])),
        ("model(input_ids, token_type_ids)", lambda: model(input_ids, token_type_ids)),
        ("model(input_ids, token_type_ids, attention_mask)", lambda: model(input_ids, token_type_ids, attention_mask)),
        ("model([input_ids, token_type_ids, attention_mask])", lambda: model([input_ids, token_type_ids, attention_mask])),
        ("model({'input_ids':..., 'token_type_ids':..., 'attention_mask':...})",
         lambda: model({
             "input_ids": input_ids,
             "token_type_ids": token_type_ids,
             "attention_mask": attention_mask
         })),
    ]

    for label, fn in attempts:
        try:
            logits = fn()
            print(f"[INFO] Forward succeeded with {label}")
            break
        except Exception as e:
            forward_errors.append((label, repr(e)))

    if logits is None:
        print("[ERROR] All forward attempts failed.")
        for label, err in forward_errors:
            print(f"   {label} -> {err}")
        raise RuntimeError("Could not run model forward pass.")

    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    pred_id = int(torch.argmax(logits, dim=-1).item())
    return ID2LABEL.get(pred_id, str(pred_id))

# ============================================================
# MAIN
# ============================================================
def main():
    print("[INFO] Reading CSV...")
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["title", "side", "prompt", "output"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"[INFO] Original rows: {len(df)}")

    work_df = df[df["prompt"].astype(str).str.strip().str.lower() != "identify"].copy()
    work_df["gold_side_norm"] = work_df["side"].apply(normalize_label)
    work_df = work_df[work_df["gold_side_norm"].notna()].copy().reset_index(drop=True)

    print(f"[INFO] Rows after removing identify and invalid labels: {len(work_df)}")

    model, tokenizer, opt = load_model()

    piece_rows = []
    row_rows = []

    total_pieces = 0
    correct_pieces = 0
    total_rows = 0
    correct_rows_majority = 0
    correct_rows_all = 0

    for i, row in work_df.iterrows():
        title = clean_text(row["title"])
        prompt = clean_text(row["prompt"])
        gold_side = row["gold_side_norm"]
        output_text = clean_text(row["output"])

        if not output_text:
            continue

        target = title if title else "politics"
        pieces = split_output(output_text, mode=SPLIT_MODE)
        if not pieces:
            continue

        row_preds = []

        for j, piece in enumerate(pieces, start=1):
            pred = predict_one(model, tokenizer, opt, target, piece)
            match = int(pred == gold_side)

            total_pieces += 1
            correct_pieces += match
            row_preds.append(pred)

            piece_rows.append({
                "row_index": i,
                "title": title,
                "prompt": prompt,
                "gold_side": gold_side,
                "piece_index": j,
                "piece_text": piece,
                "predicted_stance": pred,
                "match": match,
            })

        majority_pred = majority_vote(row_preds)
        majority_match = int(majority_pred == gold_side) if majority_pred is not None else 0
        all_match = int(all(p == gold_side for p in row_preds)) if row_preds else 0
        piece_match_rate = float(sum(p == gold_side for p in row_preds) / len(row_preds)) if row_preds else 0.0

        total_rows += 1
        correct_rows_majority += majority_match
        correct_rows_all += all_match

        row_rows.append({
            "row_index": i,
            "title": title,
            "prompt": prompt,
            "gold_side": gold_side,
            "num_pieces": len(row_preds),
            "piece_predictions": " | ".join(row_preds),
            "majority_prediction": majority_pred,
            "majority_match": majority_match,
            "all_pieces_match": all_match,
            "piece_match_rate": round(piece_match_rate, 4),
            "output_text": output_text,
        })

        print(f"[{i+1}/{len(work_df)}] gold={gold_side} | majority={majority_pred} | majority_match={majority_match}")

    piece_df = pd.DataFrame(piece_rows)
    row_df = pd.DataFrame(row_rows)

    piece_df.to_csv(OUTPUT_PIECE_CSV, index=False, encoding="utf-8-sig")
    row_df.to_csv(OUTPUT_ROW_CSV, index=False, encoding="utf-8-sig")

    piece_accuracy = (correct_pieces / total_pieces) if total_pieces else 0.0
    row_majority_accuracy = (correct_rows_majority / total_rows) if total_rows else 0.0
    row_all_match_accuracy = (correct_rows_all / total_rows) if total_rows else 0.0

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("JointCL output-vs-side evaluation summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Input CSV: {INPUT_CSV}\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Split mode: {SPLIT_MODE}\n")
        f.write(f"Rows evaluated: {total_rows}\n")
        f.write(f"Pieces evaluated: {total_pieces}\n")
        f.write(f"Piece-level accuracy: {piece_accuracy:.4f}\n")
        f.write(f"Row-level majority accuracy: {row_majority_accuracy:.4f}\n")
        f.write(f"Row-level all-pieces-match accuracy: {row_all_match_accuracy:.4f}\n")

    print("\n[INFO] Done.")
    print(f"[INFO] Piece-level results saved to: {OUTPUT_PIECE_CSV}")
    print(f"[INFO] Row-level results saved to:   {OUTPUT_ROW_CSV}")
    print(f"[INFO] Summary saved to:             {OUTPUT_TXT}")
    print(f"[INFO] Piece-level accuracy:         {piece_accuracy:.4f}")
    print(f"[INFO] Row majority accuracy:        {row_majority_accuracy:.4f}")
    print(f"[INFO] Row all-match accuracy:       {row_all_match_accuracy:.4f}")

if __name__ == "__main__":
    main()
