import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import itertools

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Drug Interaction GNN", layout="wide")


# -------------------------------
# GNN MODEL
# -------------------------------
class GNN(torch.nn.Module):

    def __init__(self, num_nodes):

        super().__init__()

        self.conv1 = GCNConv(num_nodes, 32)
        self.conv2 = GCNConv(32, 16)

        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, pair_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        d1 = x[pair_index[:, 0]]
        d2 = x[pair_index[:, 1]]

        pair = torch.cat([d1, d2], dim=1)

        out = torch.sigmoid(self.fc(pair))

        return out


# -------------------------------
# TRAIN MODEL
# -------------------------------
@st.cache_resource
def train_gnn(df):

    encoder = LabelEncoder()

    drugs = pd.concat([df["Drug1"], df["Drug2"]])
    encoder.fit(drugs)

    df["d1"] = encoder.transform(df["Drug1"])
    df["d2"] = encoder.transform(df["Drug2"])

    num_nodes = len(encoder.classes_)

    edge_index = torch.tensor(
        df[["d1", "d2"]].values.T,
        dtype=torch.long
    )

    x = torch.eye(num_nodes)

    pair_index = torch.tensor(
        df[["d1", "d2"]].values,
        dtype=torch.long
    )

    y = torch.tensor(df["Interaction"].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    model = GNN(num_nodes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):

        optimizer.zero_grad()

        pred = model(data.x, data.edge_index, pair_index).view(-1)

        loss = F.binary_cross_entropy(pred, y)

        loss.backward()

        optimizer.step()

    preds = (pred.detach().numpy() > 0.5).astype(int)

    accuracy = accuracy_score(y.numpy(), preds)

    return model, encoder, data, accuracy


# -------------------------------
# PREDICT INTERACTION
# -------------------------------
def predict_interaction(model, encoder, data, d1, d2):

    try:

        a = encoder.transform([d1])[0]
        b = encoder.transform([d2])[0]

    except:
        return None

    pair = torch.tensor([[a, b]])

    with torch.no_grad():

        pred = model(data.x, data.edge_index, pair)

    return pred.item()


# -------------------------------
# FIND SIDE EFFECT
# -------------------------------
def find_side_effect(df, d1, d2):

    match = df[
        ((df["Drug1"] == d1) & (df["Drug2"] == d2)) |
        ((df["Drug1"] == d2) & (df["Drug2"] == d1))
    ]

    # exact pair
    if not match.empty:

        severity = match.iloc[0]["Severity"]
        side = match.iloc[0]["SideEffect"]

        return severity, side

    # similar drug
    similar = df[
        (df["Drug1"] == d1) |
        (df["Drug2"] == d1) |
        (df["Drug1"] == d2) |
        (df["Drug2"] == d2)
    ]

    if not similar.empty:

        severity = similar.iloc[0]["Severity"]
        side = similar.iloc[0]["SideEffect"]

        return severity, side

    return "Unknown", "Potential drug interaction side effect"


# -------------------------------
# STREAMLIT APP
# -------------------------------
def main():

    st.title("💊 Drug–Drug Interaction & Side Effect Predictor (GNN)")

    st.sidebar.header("Upload Datasets")

    interaction_file = st.sidebar.file_uploader(
        "Upload Drug Interaction Dataset",
        type="csv"
    )

    patient_file = st.sidebar.file_uploader(
        "Upload Patient Dataset",
        type="csv"
    )

    if interaction_file is None:

        st.warning("Upload Drug Interaction Dataset")
        return

    # -------------------------------
    # LOAD DATASET
    # -------------------------------
    df = pd.read_csv(interaction_file)

    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "drug1": "Drug1",
        "drug2": "Drug2",
        "severity": "Severity",
        "description": "SideEffect"
    })

    # convert severity → interaction
    df["Interaction"] = 1

    st.subheader("Drug Interaction Dataset")

    st.dataframe(df)

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------
    model, encoder, data, accuracy = train_gnn(df)

    st.sidebar.success(f"GNN Model Accuracy: {accuracy*100:.2f}%")

    # -------------------------------
    # PATIENT DATA
    # -------------------------------
    if patient_file is None:

        st.info("Upload Patient Dataset")
        return

    patients = pd.read_csv(patient_file)

    st.subheader("Patient Dataset")

    st.dataframe(patients)

    st.subheader("Patient Drug Interaction Analysis")

    for _, row in patients.iterrows():

        st.markdown("---")

        name = row["Name"]

        drugs = [d.strip() for d in row["Drugs"].split(";")]

        st.write(f"### Patient: {name}")

        st.write("Drugs:", drugs)

        pairs = list(itertools.combinations(drugs, 2))

        results = []

        for d1, d2 in pairs:

            score = predict_interaction(model, encoder, data, d1, d2)

            if score is None:
                continue

            if score > 0.5:

                severity, side = find_side_effect(df, d1, d2)

                results.append({
                    "Drug1": d1,
                    "Drug2": d2,
                    "Interaction Probability": round(score, 3),
                    "Severity": severity,
                    "Possible Side Effect": side
                })

        if results:

            st.error("⚠ Drug Interaction Detected")

            st.table(pd.DataFrame(results))

        else:

            st.success("✅ No interaction predicted")


if __name__ == "__main__":
    main()