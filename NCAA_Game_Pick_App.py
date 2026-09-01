import pandas as pd

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

# --- Load team-level NCAA data (your 3 CSVs already merged) ---
team_master_ncaaf = pd.read_csv("data/team_master.csv")
# columns like: School, Off/Def stats, OSRS, DSRS, TSRS, PYPA, RYPA, etc.

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

    # Optional weighting (similar to your NFL model)
    df["tsrs_diff"] *= 1.0
    df["osrs_diff"] *= 1.5
    df["dsrs_diff"] *= 1.5
    df["pypa_diff"] *= 0.5
    df["rypa_diff"] *= 0.5
    df["pts_off_diff"] *= 0.75
    df["pts_def_diff"] *= 0.75

    return df


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_model_ncaaf(df):
    feature_cols = [
        "tsrs_diff", "osrs_diff", "dsrs_diff",
        "pypa_diff", "rypa_diff",
        "pts_off_diff", "pts_def_diff"
    ]

    X = df[feature_cols]
    y = df["spread_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    return model, feature_cols


def predict_games_ncaaf(model, df, feature_cols):
    # Raw model prediction (predicted margin)
    raw_pred = model.predict(df[feature_cols])

    # Vegas margin = spread_value
    vegas_margin = df["spread_value"]

    # 70/30 blend (NFL-style)
    df["model_pred"] = (0.6 * raw_pred) + (0.4 * vegas_margin)

    # ATS edge using blended prediction
    df["edge"] = df["model_pred"] - df["spread_value"]

    return df

def add_recommended_pick_ncaaf(df):
    return add_recommended_pick(df)  # reuse

def add_recommended_pick(df):
    picks = []

    for _, row in df.iterrows():
        team = row["Team"]          # AWAY team
        opp = row["Opp"]            # HOME team (corrected)
        vegas = row["spread_value"]
        model = row["model_pred"]
        edge = row["edge"] if "edge" in row else (model - vegas)

        # ---------------------------------------------
        # HOME TEAM IS OPP  (corrected)
        # ---------------------------------------------
        home_team = opp
        away_team = team

        # ---------------------------------------------------------
        # Determine if HOME is favorite
        #
        # Spread is from AWAY perspective:
        #   vegas < 0  → AWAY favored
        #   vegas > 0  → HOME favored
        # ---------------------------------------------------------
        home_is_fav = vegas > 0

        # ---------------------------------------------------------
        # NEW RULE (Corrected):
        # If HOME TEAM (Opp) is favorite AND edge is within ±3,
        # ALWAYS pick the HOME TEAM (Opp)
        # ---------------------------------------------------------
        if home_is_fav and abs(edge) <= 3.5:
            pick_side = home_team
            pick_spread = -vegas      # flip spread to home perspective
            picks.append(f"{pick_side} {pick_spread:+.1f}")
            continue

        # ---------------------------------------------------------
        # DEFAULT MODEL LOGIC
        # ---------------------------------------------------------
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

        # -----------------------------
        # Favorite side (away perspective)
        # -----------------------------
        if spread < 0:
            favorite = team      # away favorite
            underdog = opp
        elif spread > 0:
            favorite = opp       # home favorite
            underdog = team
        else:
            favorite = None
            underdog = None

        # -----------------------------
        # Which side did we actually pick?
        # -----------------------------
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

        # -----------------------------
        # Edge magnitude → strength
        # -----------------------------
        abs_edge = abs(edge)

        if abs_edge < 1:
            base = "No Model Edge"
        elif abs_edge < 3:
            base = "Lean"
        else:
            base = "Bet"

        # -----------------------------
        # Final label
        # -----------------------------
        if pick_is_fav is None or base == "No Model Edge":
            labels.append(base)
        else:
            labels.append(f"{base} Favorite" if pick_is_fav else f"{base} Underdog")

    df["confidence"] = labels
    return df

def load_week_results_ncaaf(week_number):
    if week_number == 1:
        return None

    path = f"data/Week{week_number}_Results.csv"
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    df["actual_margin"] = df["TeamScore"] - df["OppScore"]
    df["cover_flag"] = (df["actual_margin"] > df["spread_value"]).astype(int)

    return df

def build_training_data_ncaaf(week_number):
    if week_number == 1:
        return None

    spreads = load_week_spreads_ncaaf(week_number)
    results = load_week_results_ncaaf(week_number)

    if results is None:
        return None

    df = spreads.merge(results, on=["Team", "Opp"], how="inner")
    return df

def build_season_training_ncaaf(up_to_week):
    frames = []

    for wk in range(1, up_to_week):
        df = build_training_data_ncaaf(wk)
        if df is not None:
            frames.append(df)

    if len(frames) == 0:
        return None

    return pd.concat(frames, ignore_index=True)

def train_multiweek_model_ncaaf(up_to_week):
    season_df = build_season_training_ncaaf(up_to_week)

    if season_df is None:
        print("No NCAA training data available yet.")
        return None, None

    # Merge team stats
    season_df = merge_matchups_ncaaf(season_df, team_master_ncaaf)

    # Remove duplicate reverse matchups
    season_df = season_df[season_df["Team"] < season_df["Opp"]].copy()

    # Build features
    season_df = create_features_ncaaf(season_df)

    # DEBUG: Print the actual values the model is training on
    debug_cols = [
        "Team", "Opp",
        "TSRS", "TSRS_opp",
        "OSRS", "OSRS_opp",
        "DSRS", "DSRS_opp",
        "Pts_off", "Pts_off_opp",
        "Pts_def", "Pts_def_opp",
        "tsrs_diff", "osrs_diff", "dsrs_diff",
        "pts_off_diff", "pts_def_diff",
        "actual_margin"
    ]

    st.write("=== TRAINING DATA DEBUG ===")
    st.dataframe(season_df[debug_cols].head(50))


    feature_cols = [
        "tsrs_diff", "osrs_diff", "dsrs_diff",
        "pypa_diff", "rypa_diff",
        "pts_off_diff", "pts_def_diff"
    ]

    X = season_df[feature_cols]
    y = season_df["actual_margin"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=800,
            max_depth=14,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model, feature_cols

def get_week_picks_singleweek_ncaaf(week_number):
    spreads = load_week_spreads_ncaaf(week_number)
    games   = merge_matchups_ncaaf(spreads, team_master_ncaaf)

    games = games[games["Team"] < games["Opp"]].copy()

    feats   = create_features_ncaaf(games)
    model, feature_cols = train_model_ncaaf(feats)
    preds   = predict_games_ncaaf(model, feats, feature_cols)
    preds   = add_recommended_pick_ncaaf(preds)
    preds   = confidence_tiers_ncaaf(preds)
    return preds

def get_week_picks_multiweek_ncaaf(week_number):
    model, feature_cols = train_multiweek_model_ncaaf(week_number)

    # If no past data → fallback
    if model is None:
        return get_week_picks_singleweek_ncaaf(week_number)

    spreads = load_week_spreads_ncaaf(week_number)
    games   = merge_matchups_ncaaf(spreads, team_master_ncaaf)

    games = games[games["Team"] < games["Opp"]].copy()

    feats = create_features_ncaaf(games)

    # DEBUG: Print the actual values used for prediction
    debug_cols = [
        "Team", "Opp",
        "TSRS", "TSRS_opp",
        "OSRS", "OSRS_opp",
        "DSRS", "DSRS_opp",
        "Pts_off", "Pts_off_opp",
        "Pts_def", "Pts_def_opp",
        "tsrs_diff", "osrs_diff", "dsrs_diff",
        "pts_off_diff", "pts_def_diff",
        "spread_value"
    ]

    st.write("=== PREDICTION DATA DEBUG ===")
    st.dataframe(feats[debug_cols].head(50))


    # Raw model prediction (predicted margin)
    raw_pred = model.predict(feats[feature_cols])

    # Vegas margin = spread_value
    vegas_margin = feats["spread_value"]

    # 70/30 blend (NFL-style)
    feats["model_pred"] = (0.6 * raw_pred) + (0.4 * vegas_margin)

    # ATS edge using blended prediction
    feats["edge"] = feats["model_pred"] - feats["spread_value"]


    feats = confidence_tiers_ncaaf(feats)
    feats = add_recommended_pick_ncaaf(feats)

    return feats

def get_week_picks_ncaaf(week_number):
    if week_number == 1:
        return get_week_picks_singleweek_ncaaf(week_number)
    else:
        return get_week_picks_multiweek_ncaaf(week_number)

# Auto-detect available NCAA weeks based on spreads files
available_weeks = []

import os
for wk in range(1, 16):  # NCAA weeks
    path = f"data/Week{wk}_Spreads.csv"
    if os.path.exists(path):
        available_weeks.append(wk)

# If no spreads exist yet
if len(available_weeks) == 0:
    st.error("No NCAA spreads files found.")
    st.stop()

# Default to the most recent week
default_week = max(available_weeks)


import streamlit as st

st.set_page_config(page_title="NCAA Model Picks", page_icon="🏈", layout="wide")
st.title("🏈 Deanomites 2026' NCAA Weekly Picks")

# -----------------------------------------
# Auto-detect latest completed week
# -----------------------------------------
current_week = 1
completed_week = 0

for wk in range(1, 19):
    if os.path.exists(f"data/Week{wk}_Spreads.csv"):
        current_week = wk
    if os.path.exists(f"data/Week{wk}_Results.csv"):
        completed_week = wk

week_number = current_week
st.sidebar.success(f"Current Week: {week_number}")

run_button  = st.sidebar.button("Run NCAA Model")

if run_button:

    # ---------------------------------------------------------
    # Build training data for PRIOR WEEK (not current week)
    # ---------------------------------------------------------
    if week_number > 1:
        prev_week = week_number - 1
        training_df = build_training_data_ncaaf(prev_week)

        if training_df is not None:
            save_path = f"data/Week{prev_week}_Training.xlsx"
            training_df.to_excel(save_path, index=False)
            st.sidebar.success(f"NCAA training data saved for Week {prev_week}")
        else:
            st.sidebar.info(f"No NCAA training data available yet for Week {prev_week}")

    # ---------------------------------------------------------
    # Run NCAA model (single-week or multi-week)
    # ---------------------------------------------------------
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

    # -----------------------------
    # Download CSV Button (NCAA)
    # -----------------------------
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

