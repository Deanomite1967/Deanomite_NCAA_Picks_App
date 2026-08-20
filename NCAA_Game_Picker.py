import pandas as pd

# Load your CSVs
off = pd.read_csv("C:/DraftKings/NCAA/2025_Team_Offense.csv")
defn = pd.read_csv("C:/DraftKings/NCAA/2025_Team_Defense.csv")
met = pd.read_csv("C:/DraftKings/NCAA/2025_Team_Metrics.csv")

# Standardize school names (critical)
for df in [off, defn, met]:
    df["School"] = df["School"].str.strip().str.lower()

# Merge into one master table
team = (
    off.merge(defn, on="School", suffixes=("_off", "_def"))
       .merge(met, on="School")
)

team.to_excel("c:/DraftKings/NCAA/Team_Master.xlsx", index=False)

games = pd.read_csv("C:/DraftKings/NCAA/Week1_Spreads.csv")

# Standardize names
games["Team"] = games["Team"].str.strip().str.lower()
games["Opp"]  = games["Opp"].str.strip().str.lower()


# Merge home team stats
games = games.merge(
    team.add_prefix("Team_"),
    left_on="Team",
    right_on="Team_School"
)

games = games.merge(
    team.add_prefix("Opp_"),
    left_on="Opp",
    right_on="Opp_School"
)

games.to_excel("c:/DraftKings/NCAA/Games_Master.xlsx", index=False)
