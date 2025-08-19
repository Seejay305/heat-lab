import pandas as pd
from features.possession_builder import build_possessions

def test_build_possessions_stub():
    df = pd.DataFrame({"event": []})
    out = build_possessions(df)
    assert out is not None
