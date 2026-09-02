import os
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# BUILD TEAM MASTER FROM OFF/DEF/MET FILES
# -------------------------------------------------------------------

off = pd.read_csv("data/2025_Team_Offense.csv")
defn = pd.read_csv("data/2025_Team_Defense.csv")
met = pd.read_csv("data/2025_Team_Metrics.csv")

for df in [off, defn, met]:
    df["School"] = df["School"].str.strip().str.lower()

# Rename offense columns
off = off.rename(columns={
    "Pts": "Pts_off",
    "Points": "Pts_off",
    "PPG": "Pts_off",
    "PYPA": "PYPA",
    "RYPA": "RYPA"
})

# Rename defense columns
defn = defn.rename(columns={
    "Pts": "Pts_def",
    "Points Allowed": "Pts_def",
    "PA": "Pts_def"
})

team = (
    met
    .merge(off, on="School", how="left")
    .merge(defn, on="School", how="left")
)

team.to_csv("data/team_master.csv", index=False)


team_master_ncaaf = pd.read_csv("data/team_master.csv")
team_master_ncaaf["School"] = team_master_ncaaf["School"].str.strip().str.lower()
st.write("TEAM MASTER COLUMNS:", team_master_ncaaf.columns.tolist())

# -------------------------------------------------------------------
# LOAD WEEK SPREADS
# -------------------------------------------------------------------

def load_week_spreads_ncaaf(week_number):
    path = f"data/Week{week_number}_Spreads.csv"
    df = pd.read_csv(path)

    df["Team"] = df["Team"].str.strip().str.lower()
    df["Opp"]  = df["Opp"].str.strip().str.lower()

    df["spread_value"] = pd.to_numeric(df["Spread"], errors="coerce")
    df["total_value"]  = pd.to_numeric(df["Total"], errors="coerce")
    return df

# -------------------------------------------------------------------
# MERGE MATCHUPS WITH TEAM MASTER
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# FEATURE CREATION
# -------------------------------------------------------------------

def create_features_ncaaf(df):
    df["tsrs_diff"]    = df["TSRS"]    - df["TSRS_opp"]
    df["osrs_diff"]    = df["OSRS"]    - df["OSRS_opp"]
    df["dsrs_diff"]    = df["DSRS"]    - df["DSRS_opp"]
    df["pypa_diff"]    = df["PYPA"]    - df["PYPA_opp"]
    df["rypa_diff"]    = df["RYPA"]    - df["RYPA_opp"]
    df["pts_off_diff"] = df["Pts_off"] - df["Pts_off_opp"]
    df["pts_def_diff"] = df["Pts_def"] - df["Pts_def_opp"]

    return df

# -------------------------------------------------------------------
# DETERMINISTIC MODEL SPREAD (NO TRAINING)
# -------------------------------------------------------------------

def predict_games_ncaaf(feats):
    feats["model_pred"] = (
          feats["tsrs_diff"]    * 0.25
        + feats["osrs_diff"]    * 0.20
        + feats["dsrs_diff"]    * 0.20
        + feats["pypa_diff"]    * 0.10
        + feats["rypa_diff"]    * 0.10
        + feats["pts_off_diff"] * 0.10
        - feats["pts_def_diff"] * 0.10
    )
    feats["edge"] = feats["model_pred"] - feats["spread_value"]
    return feats

# -------------------------------------------------------------------
# PICK LOGIC
# -------------------------------------------------------------------

def add_recommended_pick_ncaaf(df):
    return add_recommended_pick(df)

def add_recommended_pick(df):
    picks = []
    for _, row in df.iterrows():
        team = row["Team"]
        opp = row["Opp"]
        vegas = row["spread_value"]
        model = row["model_pred"]
        edge = row["edge"]

        home_team = opp
        home_is_fav = vegas > 0

        if home_is_fav and abs(edge) <= 3.5:
            pick_side = home_team
            pick_spread = -vegas
            picks.append(f"{pick_side} {pick_spread:+.1f}")
            continue

        if model < vegas:
            pick_side = team
            pick_spread = vegas
        else:
            pick_side = opp
            pick_spread = -vegas

        picks.append(f"{pick_side} {pick_spread:+.1f}")

    df["recommended_pick"] = picks
    return df

# -------------------------------------------------------------------
# CONFIDENCE TIERS
# -------------------------------------------------------------------

def confidence_tiers_ncaaf(df):
    labels = []
    for _, row in df.iterrows():
        spread = row["spread_value"]
        team = row["Team"]
        opp = row["Opp"]
        edge = row["edge"]
        pick = row["recommended_pick"]

        if spread < 0:
            favorite = team
        elif spread > 0:
            favorite = opp
        else:
            favorite = None

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

# -------------------------------------------------------------------
# RESULTS / TRAINING EXPORT (OPTIONAL)
# -------------------------------------------------------------------

def load_week_results_ncaaf(week_number):
    if week_number < 0:
        return None
    path = f"data/Week{week_number}_Results.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["actual_margin"] = df["TeamScore"] - df["OppScore"]
    df["cover_flag"] = (df["actual_margin"] > df["spread_value"]).astype(int)
    return df

def build_training_data_ncaaf(week_number):
    if week_number < 0:
        return None
    spreads = load_week_spreads_ncaaf(week_number)
    results = load_week_results_ncaaf(week_number)
    if results is None:
        return None
    df = spreads.merge(results, on=["Team", "Opp"], how="inner")
    return df

def build_season_training_ncaaf(up_to_week):
    frames = []
    for wk in range(0, up_to_week):
        df = build_training_data_ncaaf(wk)
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

# -------------------------------------------------------------------
# WEEK PICK FUNCTIONS
# -------------------------------------------------------------------

def get_week_picks_singleweek_ncaaf(week_number):
    spreads = load_week_spreads_ncaaf(week_number)
    games   = merge_matchups_ncaaf(spreads, team_master_ncaaf)
    games = games[games["Team"] < games["Opp"]].copy()
    feats = create_features_ncaaf(games)
    preds = predict_games_ncaaf(feats)
    preds = add_recommended_pick_ncaaf(preds)
    preds = confidence_tiers_ncaaf(preds)
    return preds

def get_week_picks_multiweek_ncaaf(week_number):
    return get_week_picks_singleweek_ncaaf(week_number)

def get_week_picks_ncaaf(week_number):
    return get_week_picks_singleweek_ncaaf(week_number)

# -------------------------------------------------------------------
# AUTO-DETECT AVAILABLE WEEKS
# -------------------------------------------------------------------

available_weeks = []
for wk in range(0, 16):
    path = f"data/Week{wk}_Spreads.csv"
    if os.path.exists(path):
        available_weeks.append(wk)

if not available_weeks:
    st.error("No NCAA spreads files found.")
    st.stop()

default_week = max(available_weeks)

# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------

st.set_page_config(page_title="NCAA Model Picks", page_icon="🏈", layout="wide")
st.title("🏈 Deanomites 2026' NCAA Weekly Picks")

current_week = 1
completed_week = 0
for wk in range(0, 19):
    if os.path.exists(f"data/Week{wk}_Spreads.csv"):
        current_week = wk
    if os.path.exists(f"data/Week{wk}_Results.csv"):
        completed_week = wk

week_number = current_week
st.sidebar.success(f"Current Week: {week_number}")

run_button = st.sidebar.button("Run NCAA Model")

if run_button:
    if week_number > 1:
        prev_week = week_number - 1
        training_df = build_training_data_ncaaf(prev_week)
        if training_df is not None:
            save_path = f"data/Week{prev_week}_Training.xlsx"
            training_df.to_excel(save_path, index=False)
            st.sidebar.success(f"NCAA training data saved for Week {prev_week}")
        else:
            st.sidebar.info(f"No NCAA training data available yet for Week {prev_week}")

    results = get_week_picks_ncaaf(week_number)

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
        file_name=f"Week{week_number}_NCAA_Picks.csv",
        mime="text/csv",
        key=f"download_picks_csv_week_{week_number}"
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
