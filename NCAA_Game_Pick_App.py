import pandas as pd
import os
import streamlit as st

# Load your CSVs
off = pd.read_csv("data/2025_Team_Offense.csv")
defn = pd.read_csv("data/2025_Team_Defense.csv")
met = pd.read_csv("data/2025_Team_Metrics.csv")

# Standardize school names (critical)
for df in [off, defn, met]:
    df["School"] = df["School"].str.strip().str.lower()

# Merge into one master table
team = (
    off.merge(defn, on="School", suffixes=("_off", "_def"))
       .merge(met, on="School")
)

team.to_csv("data/team_master.csv", index=False)

# --- Load team-level NCAA data (already merged) ---
team_master_ncaaf = pd.read_csv("data/team_master.csv")
team_master_ncaaf["School"] = team_master_ncaaf["School"].str.strip().str.lower()


def load_week_spreads_ncaaf(week_number):
    path = f"data/Week{week_number}_Spreads.csv"
    df = pd.read_csv(path)

    df["Team"] = df["Team"].str.strip().str.lower()
    df["Opp"]  = df["Opp"].str.strip().str.lower()

    df["spread_value"] = pd.to_numeric(df["Spread"], errors="coerce")
    df["total_value"]  = pd.to_numeric(df["Total"], errors="coerce")
    return df


def merge_matchups_ncaaf(games, team_master):
    df = games.merge(
        team_master,
        left_on="Team",
        right_on="School",
        how="left"
    ).drop(columns=["School"])

    df = df.merge(
        team_master.add_suffix("_opp"),
        left_on="Opp",
        right_on="School_opp",
        how="left"
    ).drop(columns=["School_opp"])

    return df


def create_features_ncaaf(df):
    # SRS diffs
    df["tsrs_diff"] = df["TSRS"] - df["TSRS_opp"]
    df["osrs_diff"] = df["OSRS"] - df["OSRS_opp"]
    df["dsrs_diff"] = df["DSRS"] - df["DSRS_opp"]

    # Efficiency diffs
    df["pypa_diff"] = df["PYPA"] - df["PYPA_opp"]
    df["rypa_diff"] = df["RYPA"] - df["RYPA_opp"]

    # Offense points diff
    df["pts_off_diff"] = df["Pts_off"] - df["Pts_off_opp"]

    # Defense points diff
    df["pts_def_diff"] = df["Pts_def"] - df["Pts_def_opp"]

    # Optional weighting (same as before)
    df["tsrs_diff"] *= 1.0
    df["osrs_diff"] *= 1.5
    df["dsrs_diff"] *= 1.5
    df["pypa_diff"] *= 0.5
    df["rypa_diff"] *= 0.5
    df["pts_off_diff"] *= 0.75
    df["pts_def_diff"] *= 0.75

    return df


def add_recommended_pick_ncaaf(df):
    return add_recommended_pick(df)


def add_recommended_pick(df):
    picks = []

    for _, row in df.iterrows():
        team = row["Team"]          # AWAY team
        opp = row["Opp"]            # HOME team
        vegas = row["spread_value"]
        model = row["model_pred"]
        edge = row["edge"] if "edge" in row else (model - vegas)

        home_team = opp
        away_team = team

        home_is_fav = vegas > 0

        # Home‑bias rule
        if home_is_fav and abs(edge) <= 3.5:
            pick_side = home_team
            pick_spread = -vegas
            picks.append(f"{pick_side} {pick_spread:+.1f}")
            continue

        # Default logic
        if model < vegas:
            pick_side = team
            pick_spread = vegas
        else:
            pick_side = opp
            pick_spread = -vegas

        picks.append(f"{pick_side} {pick_spread:+.1f}")

    df["recommended_pick"] = picks
    return df


def confidence_tiers_ncaaf(df):
    labels = []

    for _, row in df.iterrows():
        spread = row["spread_value"]
        team = row["Team"]      # AWAY
        opp = row["Opp"]        # HOME
        edge = row["edge"]
        pick = row["recommended_pick"]

        # Favorite side (away perspective)
        if spread < 0:
            favorite = team
            underdog = opp
        elif spread > 0:
            favorite = opp
            underdog = team
        else:
            favorite = None
            underdog = None

        # Which side did we actually pick?
        pick_team = None
        if isinstance(pick, str):
            if pick.startswith(team):
                pick_team = team
            elif pick.startswith(opp):
                pick_team = opp

        if favorite is None or pick_team is None:
            pick_is_fav = None
        else:
            pick_is_fav = (pick_team == favorite)

        abs_edge = abs(edge)

        if abs_edge < 1:
            base = "No Model Edge"
        elif abs_edge < 3:
            base = "Lean"
        else:
            base = "Bet"

        if pick_is_fav is None or base == "No Model Edge":
            labels.append(base)
        else:
            labels.append(f"{base} Favorite" if pick_is_fav else f"{base} Underdog")

    df["confidence"] = labels
    return df


# ---------- NO TRAINING, JUST RAW METRICS → MARGIN ----------
def get_week_picks_ncaaf(week_number):
    spreads = load_week_spreads_ncaaf(week_number)
    games   = merge_matchups_ncaaf(spreads, team_master_ncaaf)

    # keep one direction
    games = games[games["Team"] < games["Opp"]].copy()

    feats = create_features_ncaaf(games)

    # Simple deterministic margin using your diffs
    margin = (
        feats["tsrs_diff"]
        + feats["osrs_diff"]
        + feats["dsrs_diff"]
        + feats["pypa_diff"]
        + feats["rypa_diff"]
        + feats["pts_off_diff"]
        - feats["pts_def_diff"]
    )

    # Weighting
    df["tsrs_diff"] *= 1.75
    df["osrs_diff"] *= 1.5
    df["dsrs_diff"] *= 1.5
    df["pypa_diff"] *= 0.5
    df["rypa_diff"] *= 0.5
    df["pts_off_diff"] *= 0.75
    df["pts_def_diff"] *= 0.75

    feats["model_pred"] = margin
    feats["edge"] = feats["model_pred"] - feats["spread_value"]

    feats = add_recommended_pick_ncaaf(feats)
    feats = confidence_tiers_ncaaf(feats)

    return feats


# Auto-detect available NCAA weeks based on spreads files
available_weeks = []
for wk in range(1, 16):  # NCAA weeks
    path = f"data/Week{wk}_Spreads.csv"
    if os.path.exists(path):
        available_weeks.append(wk)

if len(available_weeks) == 0:
    st.error("No NCAA spreads files found.")
    st.stop()

default_week = max(available_weeks)

st.set_page_config(page_title="NCAA Model Picks", page_icon="🏈", layout="wide")
st.title("🏈 Deanomites 2026' NCAA Weekly Picks")

current_week = default_week
st.sidebar.success(f"Current Week: {current_week}")

run_button  = st.sidebar.button("Run NCAA Model")

if run_button:
    results = get_week_picks_ncaaf(current_week)

    st.dataframe(
        results[[
            "Team", "Opp",
            "spread_value",
            "model_pred",
            "edge",
            "confidence",
            "recommended_pick"
        ]],
        use_container_width=True
    )

    export_df = results.copy()
    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download NCAA Picks as CSV",
        data=csv_data,
        file_name=f"Week{current_week}_NCAA_Picks.csv",
        mime="text/csv",
        key=f"download_picks_csv_week_{current_week}"
    )

st.markdown(
    """
    <a href="mailto:deanomite@gmail.com" style="text-decoration:none;">
        <button style="
            background-color:#4CAF50;
            color:white;
            padding:10px 20px;
            border:none;
            border-radius:5px;
            cursor:pointer;
            font-size:16px;">
            📧 Email Deanomite for Questions or Comments
        </button>
    </a>
    """,
    unsafe_allow_html=True
)
