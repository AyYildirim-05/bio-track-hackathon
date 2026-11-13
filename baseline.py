from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import peptides
import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModel, AutoTokenizer

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_SET = set(AMINO_ACIDS)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "acp_model.joblib"


def clean_sequence(seq: str) -> str:
    """Normalize a peptide sequence, stripping FASTA headers and whitespace."""
    lines = [
        line.strip()
        for line in str(seq).strip().splitlines()
        if line.strip() and not line.strip().startswith(">")
    ]
    cleaned = "".join(lines).replace(" ", "").upper()
    return "".join(c for c in cleaned if c in AMINO_ACID_SET)


def extract_physicochemical(seq: str) -> dict:
    clean_seq = clean_sequence(seq)
    length = len(clean_seq)

    if length == 0:
        feats = {f"aac_{aa}": 0.0 for aa in AMINO_ACIDS}
        feats.update(
            {
                "length": 0,
                "charge_pH7_4": 0.0,
                "hydrophobicity": 0.0,
                "hydrophobic_moment": 0.0,
                "isoelectric_point": 0.0,
                "aliphatic_index": 0.0,
                "boman_index": 0.0,
            }
        )
        return feats

    pep = peptides.Peptide(clean_seq)

    # 1. Amino Acid Composition (20 features)
    feats = {f"aac_{aa}": clean_seq.count(aa) / length for aa in AMINO_ACIDS}

    # 2. Biophysical properties key to anticancer peptide activity
    # Clamp window to sequence length so short sequences do not return None
    hm_window = min(11, length)
    hm_val = pep.hydrophobic_moment(window=hm_window)

    feats["length"] = length
    feats["charge_pH7_4"] = float(pep.charge(pH=7.4))
    feats["hydrophobicity"] = float(pep.hydrophobicity(scale="Eisenberg"))
    feats["hydrophobic_moment"] = float(hm_val) if hm_val is not None else 0.0
    feats["isoelectric_point"] = float(pep.isoelectric_point())
    feats["aliphatic_index"] = float(pep.aliphatic_index())
    feats["boman_index"] = float(pep.boman())
    return feats


@lru_cache(maxsize=2)
def _load_esm2_model(model_name: str = "facebook/esm2_t6_8M_UR50D"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def extract_esm2_embeddings(
    sequences: list[str], model_name: str = "facebook/esm2_t6_8M_UR50D"
) -> np.ndarray:
    if not sequences:
        return np.empty((0, 320), dtype=np.float32)

    tokenizer, model, device = _load_esm2_model(model_name)
    hidden_size = model.config.hidden_size

    embeddings = []
    for seq in sequences:
        clean_seq = clean_sequence(seq)
        if not clean_seq:
            embeddings.append(np.zeros(hidden_size, dtype=np.float32))
            continue

        inputs = tokenizer(
            clean_seq, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        outputs = model(**inputs)
        # Mean-pool token embeddings (excluding [CLS] and [EOS])
        token_emb = outputs.last_hidden_state[0, 1:-1, :]
        if token_emb.shape[0] == 0:
            mean_emb = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
        else:
            mean_emb = token_emb.mean(dim=0).cpu().numpy()
        embeddings.append(mean_emb)
    return np.vstack(embeddings)


def train_default_baseline(out_path: str | Path = DEFAULT_MODEL_PATH):
    data = pd.DataFrame(
        {
            "sequence": [
                "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin-like (ACP)
                "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin-2 (ACP)
                "FLPLIGRVLSGIL",  # Temporin-like (ACP)
                "AAADDDGGGSSSTTTNNNQQQ",  # Non-ACP
                "EEEEDDDDSSSSPPPPGGGG",  # Non-ACP
                "LLLLVVVVIIIIAAAAGGGG",  # Non-ACP
            ],
            "label": [1, 1, 1, 0, 0, 0],
        }
    )

    print("Extracting biophysical features...")
    bio_df = pd.DataFrame(
        [extract_physicochemical(s) for s in data["sequence"]]
    )

    print("Extracting ESM-2 embeddings...")
    esm_embs = extract_esm2_embeddings(data["sequence"].tolist())

    X = np.hstack([bio_df.values, esm_embs])
    y = data["label"].values

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, out_path)
    return clf, X, y


if __name__ == "__main__":
    clf, X, y = train_default_baseline(DEFAULT_MODEL_PATH)
    print(f"Feature matrix shape: {X.shape}")
    print(
        "Predicted ACP probabilities:",
        np.round(clf.predict_proba(X)[:, 1], 3),
    )
    print(f"Saved model to {DEFAULT_MODEL_PATH} — setup complete!")

