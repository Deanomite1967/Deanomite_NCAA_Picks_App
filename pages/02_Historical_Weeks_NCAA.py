import streamlit as st
import pandas as pd
import os
from io import BytesIO

st.set_page_config(page_title="Historical NCAA Weeks", page_icon="📊", layout="wide")
st.title("📊 Historical NCAA Results")

st.write("View past weekly NCAA results (actual scores, margins, and ATS outcomes).")

# -----------------------------------------
# Find available historical results files
# -----------------------------------------
historical_files = []
for wk in range(1, 16):   # NCAA weeks
    path = f"data/Week{wk}_Results.csv"
    if os.path.exists(path):
        historical_files.append((wk, path))

# If no history yet (Week 1 scenario)
if len(historical_files) == 0:
    st.info("No historical results available yet. Results will appear after Week 1 completes.")
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

# Compute actual margin + ATS result if not already present
if "actual_margin" not in df.columns:
    df["actual_margin"] = df["TeamScore"] - df["OppScore"]

def ncaaf_ats_result(row):
    pick = row["recommended_pick"]
    team = row["Team"]
    margin = row["actual_margin"]
    spread = row["spread_value"]

    # Did the model pick Team or Opp?
    pick_team = pick.startswith(team)

    # Spread of the picked side
    pick_spread = spread if pick_team else -spread

    # Correct ATS logic
    return "Win" if (margin + pick_spread) > 0 else "Loss"

# Compute ATS result based on recommended pick
df["ATS_Result"] = df.apply(ncaaf_ats_result, axis=1)
df["cover_flag"] = (df["ATS_Result"] == "Win").astype(int)


st.subheader(f"Historical Results — Week {selected_week}")

# -----------------------------------------
# Display clean table
# -----------------------------------------
clean_cols = [
    "Team", "Opp",
    "spread_value",
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
