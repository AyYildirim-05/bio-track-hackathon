from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from baseline import (
    AMINO_ACID_SET,
    AMINO_ACIDS,
    DEFAULT_MODEL_PATH,
    extract_esm2_embeddings,
    extract_physicochemical,
    train_default_baseline,
)

st.set_page_config(
    page_title="PEPTIDE-SCAN 2000 // ACP PREDICTOR",
    layout="centered",
)

# Y2K CRT Monitor Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #050a05;
        color: #33ff33;
        font-family: 'Courier New', Courier, monospace;
    }
    h1, h2, h3, h4, p, label, .stMarkdown,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stCaptionContainer"] {
        color: #33ff33 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    .stTextInput input {
        background-color: #001400 !important;
        color: #33ff33 !important;
        border: 2px solid #33ff33 !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-size: 1.1rem !important;
    }
    .stButton button {
        background-color: #002b00 !important;
        color: #33ff33 !important;
        border: 2px solid #33ff33 !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(">> PEPTIDE-SCAN 2000 [v1.99]")
st.caption("ONCOLOGY BIOINFORMATICS WORKSTATION // 20-AMINO-ACID ALPHABET ASSAY")
st.markdown("---")


@st.cache_resource
def load_model():
    if not DEFAULT_MODEL_PATH.exists():
        clf, _, _ = train_default_baseline(DEFAULT_MODEL_PATH)
        return clf
    return joblib.load(DEFAULT_MODEL_PATH)


model = load_model()

default_seq = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
raw_input = st.text_input(
    "ENTER PEPTIDE SEQUENCE (CANONICAL 20 AMINO ACIDS):", value=default_seq
)
# Normalize whitespace and optional FASTA header lines
seq_lines = [
    line.strip()
    for line in raw_input.strip().splitlines()
    if line.strip() and not line.strip().startswith(">")
]
seq_input = "".join(seq_lines).replace(" ", "").upper()

if st.button("[ EXECUTE BIO-ASSAY ]"):
    invalid_chars = sorted({c for c in seq_input if c not in AMINO_ACID_SET})
    if not seq_input:
        st.error("SYNTAX ERROR: EMPTY PEPTIDE SEQUENCE.")
    elif invalid_chars:
        st.error(f"SYNTAX ERROR: INVALID RESIDUES DETECTED -> {invalid_chars}")
    else:
        with st.spinner("COMPUTING BIOPHYSICAL TENSORS & ESM-2 EMBEDDINGS..."):
            bio_feats = extract_physicochemical(seq_input)
            bio_df = pd.DataFrame([bio_feats])
            esm_emb = extract_esm2_embeddings([seq_input])
            X = np.hstack([bio_df.values, esm_emb])
            prob = float(model.predict_proba(X)[0, 1])

        st.markdown("### [ ASSAY DIAGNOSTIC OUTPUT ]")
        col1, col2, col3 = st.columns(3)
        col1.metric("ACP PROBABILITY", f"{prob * 100:.1f}%")
        col2.metric("NET CHARGE (pH 7.4)", f"{bio_feats['charge_pH7_4']:+.2f}")
        col3.metric("HYDROPHOBIC MOMENT", f"{bio_feats['hydrophobic_moment']:.3f}")

        if prob >= 0.5:
            st.success(
                "CLASSIFICATION: POSITIVE — CANDIDATE ANTICANCER PEPTIDE "
                f"(LEN={len(seq_input)})"
            )
        else:
            st.warning(
                "CLASSIFICATION: NEGATIVE — NON-ANTICANCER PEPTIDE "
                f"(LEN={len(seq_input)})"
            )

        st.markdown("#### RESIDUE COMPOSITION SPECTRUM")
        aac_data = pd.DataFrame(
            {
                "Amino Acid": AMINO_ACIDS,
                "Frequency": [bio_feats[f"aac_{aa}"] for aa in AMINO_ACIDS],
            }
        ).set_index("Amino Acid")
        st.bar_chart(aac_data)
