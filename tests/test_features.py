import pandas as pd
from features.possession_builder import build_possessions

def test_build_possessions_minimal():
    # minimal synthetic pbp: start -> made shot -> next possession start
    df = pd.DataFrame({
        "GAME_ID": ["X","X","X"],
        "EVENTNUM": [1,2,3],
        "PERIOD": [1,1,1],
        "PCTIMESTRING": ["12:00","11:45","11:30"],
        "HOMEDESCRIPTION": ["Start of Period","",""],
        "VISITORDESCRIPTION": ["","makes 2-pt shot","jump ball"],
        "NEUTRALDESCRIPTION": ["","",""],
    })
    poss = build_possessions(df)
    assert len(poss) >= 1
    assert poss.iloc[0]["START_EVENTNUM"] == 1
    assert poss.iloc[0]["END_EVENTNUM"] == 2
