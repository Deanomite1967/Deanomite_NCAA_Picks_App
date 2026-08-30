import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NCAA Year Summary", page_icon="📈", layout="wide")
st.title("📈 2026' NCAA Season Summary")

st.write("Season‑to‑date ATS performance based on all available weekly NCAA results.")

# -----------------------------------------
# Load all available weekly results
# -----------------------------------------
all_weeks = []

for wk in range(1, 16):   # NCAA weeks
    path = f"data/Week{wk}_Results.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df["week"] = wk
            all_weeks.append(df)
        except:
            pass

# If no results exist yet
if len(all_weeks) == 0:
    st.info("No season results available yet. Summary will appear once Week 1 results are posted.")
    st.stop()

# Combine all weeks
season_df = pd.concat(all_weeks, ignore_index=True)

# -----------------------------------------
# Compute ATS metrics (correct formula)
# -----------------------------------------
season_df["actual_margin"] = season_df["TeamScore"] - season_df["OppScore"]

def compute_pick_side(row):
    pick = row["recommended_pick"].strip().lower()
    team = row["Team"].strip().lower()
    opp = row["Opp"].strip().lower()

    pick_team = pick.startswith(team)
    pick_opp = pick.startswith(opp)

    return pick_team, pick_opp

def compute_ats(row):
    margin = row["actual_margin"]
    spread = row["spread_value"]

    pick_team, pick_opp = compute_pick_side(row)

    # Spread of picked side (for display)
    pick_spread = spread if pick_team else -spread

    # Correct ATS margin
    if pick_team:
        ats_margin = margin + spread
    else:
        ats_margin = -margin - spread

    result = "Win" if ats_margin >= 0 else "Loss"

    return pd.Series([pick_spread, ats_margin, result])

def get_picked_side(row):
    pick = row["recommended_pick"].strip().lower()
    team = row["Team"].strip().lower()
    opp = row["Opp"].strip().lower()

    if pick.startswith(team):
        return row["Team"]
    elif pick.startswith(opp):
        return row["Opp"]
    else:
        return None  # should never happen

season_df["picked_side"] = season_df.apply(get_picked_side, axis=1)

season_df[["pick_spread", "ats_margin", "ATS_Result"]] = season_df.apply(compute_ats, axis=1)
season_df["cover_flag"] = (season_df["ATS_Result"] == "Win").astype(int)
season_df["ATS_Result"] = season_df["cover_flag"].map({1: "Win", 0: "Loss"})

total_games = len(season_df)
total_wins = season_df["cover_flag"].sum()
total_losses = total_games - total_wins
win_pct = round(total_wins / total_games * 100, 2)

# -----------------------------------------
# Display summary metrics
# -----------------------------------------
st.subheader("Season ATS Summary")

st.metric("Total Games", total_games)
st.metric("ATS Wins", total_wins)
st.metric("ATS Losses", total_losses)
st.metric("Win Percentage", f"{win_pct}%")

# -----------------------------------------
# Team‑level ATS record (correct pick-based)
# -----------------------------------------
st.subheader("Team‑Level ATS Performance")

team_summary = (
    season_df.groupby("picked_side")["cover_flag"]
    .agg(["count", "sum"])
    .rename(columns={"count": "Games", "sum": "ATS Wins"})
)

team_summary["ATS Losses"] = team_summary["Games"] - team_summary["ATS Wins"]
team_summary["Win %"] = (team_summary["ATS Wins"] / team_summary["Games"] * 100).round(2)

team_summary.index.name = "Team"

st.dataframe(team_summary, use_container_width=True)


# -----------------------------------------
# Week‑by‑week breakdown
# -----------------------------------------
st.subheader("Week‑by‑Week ATS Results")

week_summary = (
    season_df.groupby("week")["cover_flag"]
    .agg(["count", "sum"])
    .rename(columns={"count": "Games", "sum": "ATS Wins"})
)

week_summary["ATS Losses"] = week_summary["Games"] - week_summary["ATS Wins"]
week_summary["Win %"] = (week_summary["ATS Wins"] / week_summary["Games"] * 100).round(2)

st.dataframe(week_summary, use_container_width=True)

# -----------------------------------------
# Download Year Summary (auto-sized Excel)
# -----------------------------------------
from excel_utils import autosize_excel

export_df = season_df.copy()
buffer = autosize_excel(export_df)

st.download_button(
    label="Download Full NCAA Season Summary",
    data=buffer,
    file_name="NCAA_Season_Summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

team_buffer = autosize_excel(team_summary)

st.download_button(
    label="Download NCAA Team-Level ATS Summary",
    data=team_buffer,
    file_name="NCAA_Team_ATS_Summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

week_buffer = autosize_excel(week_summary)

st.download_button(
    label="Download NCAA Week-by-Week ATS Summary",
    data=week_buffer,
    file_name="NCAA_Week_ATS_Summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

