import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pro_gradu.classification import classification_features, make_classification_workflow
from pro_gradu.data import BASELINE_FEATURES


def test_classification_features_excludes_time_and_preserves_index():
    frame = pd.DataFrame({**{name: [1, 2] for name in BASELINE_FEATURES}, "time": [4, 5], "DEATH_EVENT": [0, 1]}, index=[3, 9])
    X, y = classification_features(frame)
    assert "time" not in X.columns
    assert len(X.columns) == 11
    assert X.index.tolist() == [3, 9]
    assert y.index.tolist() == [3, 9]


def test_selected_workflow_contains_scaler_and_selector():
    workflow = make_classification_workflow(LogisticRegression(), select_features=True)
    assert isinstance(workflow.named_steps["scale"], StandardScaler)
    assert isinstance(workflow.named_steps["select"], SelectKBest)
