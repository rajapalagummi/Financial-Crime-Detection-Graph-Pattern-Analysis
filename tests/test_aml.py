import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from src.graph_detector import build_graph, detect_fan_out, detect_fan_in, detect_cycles, run_graph_analysis
from src.feature_engineering import FEATURE_COLS


def make_pdf(edges):
    return pd.DataFrame(
        [{"from_account": a, "to_account": b, "amount_paid": 100.0,
          "payment_currency": "USD", "is_laundering": l} for a, b, l in edges]
    )


def test_fan_out_detected():
    G = build_graph(make_pdf([("A", t, 1 if t == "T1" else 0) for t in ["T1", "T2", "T3", "T4", "T5"]]))
    flagged = detect_fan_out(G, threshold=5)
    assert len(flagged) == 1
    assert flagged[0]["account"] == "A"
    assert flagged[0]["laundering_edges"] == 1


def test_fan_in_detected():
    G = build_graph(make_pdf([(s, "Z", 0) for s in ["S1", "S2", "S3", "S4", "S5"]]))
    flagged = detect_fan_in(G, threshold=5)
    assert len(flagged) == 1
    assert flagged[0]["account"] == "Z"


def test_below_threshold_not_flagged():
    G = build_graph(make_pdf([("A", "B", 0), ("A", "C", 0)]))
    assert detect_fan_out(G, threshold=5) == []
    assert detect_fan_in(G, threshold=5) == []


def test_cycle_detected():
    G = build_graph(make_pdf([("A", "B", 1), ("B", "C", 1), ("C", "A", 0)]))
    cycles = detect_cycles(G, max_length=10)
    assert len(cycles) == 1
    assert cycles[0]["length"] == 3
    assert cycles[0]["laundering_edges"] == 2


def test_run_graph_analysis_keys():
    result = run_graph_analysis(make_pdf([("A", "B", 0), ("B", "A", 1)]))
    for k in ["nodes", "edges", "fan_out_count", "fan_in_count", "cycle_count", "flagged_accounts"]:
        assert k in result
    assert result["cycle_count"] == 1


@pytest.fixture(scope="module")
def spark():
    from src.data_pipeline import get_spark
    s = get_spark()
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def test_features_built(spark):
    from datetime import datetime
    from src.feature_engineering import build_transaction_features
    rows = [
        (datetime(2022, 9, 1, 2, 0), "A", "B", 100.0, 90.0, "USD", "EUR", "Wire", 0),
        (datetime(2022, 9, 1, 12, 0), "A", "C", 50.0, 50.0, "USD", "USD", "Cash", 1),
        (datetime(2022, 9, 1, 23, 30), "B", "C", 20.0, 20.0, "USD", "USD", "Wire", 0),
    ]
    cols = ["timestamp", "from_account", "to_account", "amount_paid", "amount_received",
            "payment_currency", "receiving_currency", "payment_format", "is_laundering"]
    df = build_transaction_features(spark.createDataFrame(rows, cols)).toPandas()
    for c in FEATURE_COLS:
        assert c in df.columns
    assert df.loc[df["to_account"] == "B", "is_cross_currency"].iloc[0] == 1.0
    assert df.loc[df["to_account"] == "B", "is_night"].iloc[0] == 1.0
    assert df.loc[df["from_account"] == "A", "from_tx_count"].iloc[0] == 2
