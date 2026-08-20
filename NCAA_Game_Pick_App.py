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
    df["model_pred"] = (0.7 * raw_pred) + (0.3 * vegas_margin)

    # ATS edge using blended prediction
    df["edge"] = df["model_pred"] - df["spread_value"]

    return df


def confidence_tiers_ncaaf(df):
    labels = []

    for _, row in df.iterrows():
        edge = row["edge"]
        spread = row["spread_value"]
        team = row["Team"]
        opp = row["Opp"]

        # Who is the favorite?
        # If spread < 0, Team is favorite; if spread > 0, Opp is favorite
        if spread < 0:
            team_is_fav = True
        elif spread > 0:
            team_is_fav = False
        else:
            team_is_fav = None  # pick'em

        # Who did we pick? (recompute instead of parsing string)
        # Your pick logic: if model_pred < spread_value → pick Team, else Opp
        pick_team = row["model_pred"] < spread

        if pick_team:
            pick_is_fav = team_is_fav
        else:
            pick_is_fav = (not team_is_fav) if team_is_fav is not None else None

        # Edge magnitude → strength
        abs_edge = abs(edge)
        if abs_edge < 1:
            base = "No Play"
        elif abs_edge < 3:
            base = "Lean"
        else:
            base = "Bet"

        # If no favorite (true pick'em), just use base
        if pick_is_fav is None or base == "No Play":
            labels.append(base)
        else:
            if pick_is_fav:
                labels.append(f"{base} Favorite")
            else:
                labels.append(f"{base} Underdog")

    df["confidence"] = labels
    return df

def add_recommended_pick_ncaaf(df):
    return add_recommended_pick(df)  # reuse

def add_recommended_pick(df):
    picks = []

    for _, row in df.iterrows():
        team = row["Team"]
        opp = row["Opp"]
        vegas = row["spread_value"]
        model = row["model_pred"]

        # If model thinks TEAM should be stronger → pick TEAM
        if model < vegas:
            pick_side = team
            pick_spread = vegas
        else:
            pick_side = opp
            pick_spread = -vegas  # Opposite side of the spread

        picks.append(f"{pick_side} {pick_spread:+.1f}")

    df["recommended_pick"] = picks
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
    preds   = confidence_tiers_ncaaf(preds)
    preds   = add_recommended_pick_ncaaf(preds)

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

    # Raw model prediction (predicted margin)
    raw_pred = model.predict(feats[feature_cols])

    # Vegas margin = spread_value
    vegas_margin = feats["spread_value"]

    # 70/30 blend (NFL-style)
    feats["model_pred"] = (0.7 * raw_pred) + (0.3 * vegas_margin)

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
st.title("🏈 Deanomites NCAA Weekly Picks Engine")

latest_week = 1
for wk in range(1, 19):
    path = f"data/Week{wk}_Results.csv"
    if os.path.exists(path):
        latest_week = wk

week_number = latest_week
st.sidebar.success(f"Current Week: {week_number}")

run_button  = st.sidebar.button("Run NCAA Model")

if run_button:
    results = get_week_picks_ncaaf(week_number)
    
    training_df = build_training_data_ncaaf(week_number)
    if training_df is not None:
        training_df.to_excel(f"data/Week{week_number}_Training.xlsx", index=False)

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

    # Full export including all columns
    export_df = results.copy()

    from io import BytesIO
    from openpyxl import load_workbook

    buffer = BytesIO()

    # Write full dataframe to Excel
    export_df.to_excel(buffer, index=False, engine="openpyxl")

    buffer.seek(0)
    wb = load_workbook(buffer)
    ws = wb.active

    # Auto-size columns
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter

        for cell in col:
            try:
                cell_length = len(str(cell.value))
                if cell_length > max_length:
                    max_length = cell_length
            except:
                pass

        ws.column_dimensions[col_letter].width = max_length + 2

    # Save adjusted workbook
    buffer2 = BytesIO()
    wb.save(buffer2)
    buffer2.seek(0)

    st.download_button(
        label="Download NCAA Picks as Excel",
        data=buffer2,
        file_name=f"Week{week_number}_NCAA_Picks.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_picks_excel_week_{week_number}"
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

