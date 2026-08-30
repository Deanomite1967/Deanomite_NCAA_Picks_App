import streamlit as st
import pandas as pd
import os
from io import BytesIO

st.set_page_config(page_title="Historical NCAA Weeks", page_icon="📊", layout="wide")
st.title("📊 2026' Historical NCAA Results")

st.write("View past weekly NCAA results (actual scores, margins, and ATS outcomes).")

# -----------------------------------------
# Find available historical results files
# -----------------------------------------
historical_files = []
for wk in range(0, 16):   # NCAA weeks
    path = f"data/Week{wk}_Results.csv"
    if os.path.exists(path):
        historical_files.append((wk, path))

# If no history yet (Week 1 scenario)
if len(historical_files) < 0:
    st.info("No historical results available yet. Results will appear after Week 0 completes.")
    st.stop()

# -----------------------------------------
# Week selector
# -----------------------------------------
week_numbers = [wk for wk, _ in historical_files]

selected_week = st.selectbox(
    "Select a historical week",
    week_numbers,
    index=len(week_numbers) - 1  # default to most recent
)

file_path = f"data/Week{selected_week}_Results.csv"

# -----------------------------------------
# Load selected week
# -----------------------------------------
try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"Error loading Week {selected_week} results: {e}")
    st.stop()

# Ensure actual_margin exists
if "actual_margin" not in df.columns:
    df["actual_margin"] = df["TeamScore"] - df["OppScore"]

def ncaaf_ats(row):
    team = row["Team"].strip().lower()
    opp = row["Opp"].strip().lower()
    pick = row["recommended_pick"].strip().lower()
    spread = row["spread_value"]
    margin = row["actual_margin"]  # TeamScore - OppScore

    # Which side was picked?
    pick_team = pick.startswith(team)
    pick_opp = pick.startswith(opp)

    # For display: spread of picked side
    if pick_team:
        pick_spread = spread              # Team line (e.g., +8.5 or -31.5)
        ats_margin = margin + spread      # ATS from Team perspective
    elif pick_opp:
        pick_spread = -spread             # Opp line (e.g., -8.5 or +31.5)
        ats_margin = -margin - spread     # ATS from Opp perspective
    else:
        pick_spread = 0
        ats_margin = 0

    result = "Win" if ats_margin >= 0 else "Loss"
    return pd.Series([pick_spread, ats_margin, result])

df[["pick_spread", "ats_margin", "ATS_Result"]] = df.apply(ncaaf_ats, axis=1)
df["cover_flag"] = (df["ATS_Result"] == "Win").astype(int)




st.subheader(f"Historical Results — Week {selected_week}")

# -----------------------------------------
# Display clean table
# -----------------------------------------
clean_cols = [
    "Team", "Opp",
    "spread_value",
    "recommended_pick",
    "pick_spread",
    "TeamScore", "OppScore",
    "actual_margin",
    "ATS_Result"
]


available_cols = [c for c in clean_cols if c in df.columns]

st.dataframe(df[available_cols], use_container_width=True)

# -----------------------------------------
# Download historical week results (auto-sized)
# -----------------------------------------
from excel_utils import autosize_excel

export_df = df[available_cols]
buffer = autosize_excel(export_df)

st.download_button(
    label=f"Download Week {selected_week} NCAA Results",
    data=buffer,
    file_name=f"Week{selected_week}_NCAA_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
