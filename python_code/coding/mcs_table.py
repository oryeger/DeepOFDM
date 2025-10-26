# mcs_table.py
import pandas as pd

# 5G NR MCS table (indices 0–27)
mcs_data = [
    {"Index": 0,  "Qm": 2, "Code Rate": 0.1172},
    {"Index": 1,  "Qm": 2, "Code Rate": 0.1885},
    {"Index": 2,  "Qm": 2, "Code Rate": 0.3008},
    {"Index": 3,  "Qm": 2, "Code Rate": 0.4385},
    {"Index": 4,  "Qm": 2, "Code Rate": 0.5879},
    {"Index": 5,  "Qm": 4, "Code Rate": 0.3692},
    {"Index": 6,  "Qm": 4, "Code Rate": 0.4238},
    {"Index": 7,  "Qm": 4, "Code Rate": 0.4785},
    {"Index": 8,  "Qm": 4, "Code Rate": 0.5401},
    {"Index": 9,  "Qm": 4, "Code Rate": 0.6016},
    {"Index": 10, "Qm": 4, "Code Rate": 0.6426},
    {"Index": 11, "Qm": 6, "Code Rate": 0.4551},
    {"Index": 12, "Qm": 6, "Code Rate": 0.5049},
    {"Index": 13, "Qm": 6, "Code Rate": 0.5537},
    {"Index": 14, "Qm": 6, "Code Rate": 0.6016},
    {"Index": 15, "Qm": 6, "Code Rate": 0.6504},
    {"Index": 16, "Qm": 6, "Code Rate": 0.7022},
    {"Index": 17, "Qm": 6, "Code Rate": 0.7539},
    {"Index": 18, "Qm": 6, "Code Rate": 0.8027},
    {"Index": 19, "Qm": 6, "Code Rate": 0.8525},
    {"Index": 20, "Qm": 8, "Code Rate": 0.6665},
    {"Index": 21, "Qm": 8, "Code Rate": 0.6943},
    {"Index": 22, "Qm": 8, "Code Rate": 0.7363},
    {"Index": 23, "Qm": 8, "Code Rate": 0.7783},
    {"Index": 24, "Qm": 8, "Code Rate": 0.8213},
    {"Index": 25, "Qm": 8, "Code Rate": 0.8643},
    {"Index": 26, "Qm": 8, "Code Rate": 0.8950},
    {"Index": 27, "Qm": 8, "Code Rate": 0.9258},
    {"Index": 28, "Qm": 4, "Code Rate": 0.8213},
    {"Index": 29, "Qm": 4, "Code Rate": 0.8950},
    {"Index": 30, "Qm": 4, "Code Rate": 0.9258},
    {"Index": 30, "Qm": 4, "Code Rate": 0.9258}
]

# Convert to DataFrame
mcs_df = pd.DataFrame(mcs_data)

# Function to get modulation and code rate by index
def get_mcs(index):
    row = mcs_df.loc[mcs_df['Index'] == index]
    if row.empty:
        raise ValueError(f"MCS index {index} not found (valid: 0-27)")
    return row.iloc[0]['Qm'], row.iloc[0]['Code Rate']
