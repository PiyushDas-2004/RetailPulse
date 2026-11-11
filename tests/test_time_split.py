from src.data.load import make_origins
from src.utils import load_config


def test_origins_are_time_ordered_and_non_overlapping():
    cfg = load_config()
    o = make_origins(cfg)
    h = cfg["problem"]["horizon"]
    for a, b in zip(o, o[1:]):
        assert a.test_end < b.test_start, "test windows must not overlap"
        assert a.train_end < b.train_end, "expanding window: training end moves forward"
    for x in o:
        assert x.train_end == x.test_start - 1, "train ends right before test starts (no gap, no overlap)"
        assert x.test_end - x.test_start + 1 == h


def test_holdout_is_last_and_untouched_by_tuning_and_cv():
    cfg = load_config()
    o = make_origins(cfg)
    assert o[-1].name == "holdout" and o[-1].test_end == cfg["data"]["n_full_weeks"] - 1
    kinds = [x.kind for x in o]
    # tuning origins come first, then CV, then holdout: tuning never sees CV or holdout weeks
    assert kinds == sorted(kinds, key=["tuning", "cv", "holdout"].index)
    last_tune = max(x.test_end for x in o if x.kind == "tuning")
    first_cv = min(x.test_start for x in o if x.kind == "cv")
    assert last_tune < first_cv
