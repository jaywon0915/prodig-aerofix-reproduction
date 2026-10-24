from prodig.config import ProgressiveConfig


def test_default_schedule_matches_paper_altitude_factors():
    cfg = ProgressiveConfig()
    assert cfg.initial_steps == 7000
    assert cfg.altitude_factors == (0.9, 0.7, 0.5, 0.3, 0.1)
    assert len(cfg.altitude_factors) == 5


def test_total_steps_includes_each_progressive_stage():
    cfg = ProgressiveConfig(initial_steps=7000, stage_steps=2000)
    assert cfg.total_steps == 17000
