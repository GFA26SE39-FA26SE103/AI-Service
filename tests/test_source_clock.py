import pytest
from app.source_clock import SourceClock

def test_recorded_clock_uses_video_time_not_inference_duration():
    clock=SourceClock("RECORDED")
    assert clock.observe(4900,25,100).elapsed_ms==4900
    assert clock.observe(5000,25,110).elapsed_ms==5000
    assert not clock.observe(5040,25,120).continuity_broken
    assert clock.observe(1000,25,121).continuity_broken

def test_missing_pts_uses_valid_fps_or_reports_error():
    clock=SourceClock("RECORDED")
    assert clock.observe(None,25,1).elapsed_ms==0
    assert clock.observe(None,25,20).elapsed_ms==40
    with pytest.raises(ValueError,match="AI_SOURCE_TIME_INVALID"):
        SourceClock("RECORDED").observe(None,float("nan"),1)

def test_live_gap_resets_continuity():
    clock=SourceClock("LIVE")
    assert clock.observe(None,None,100).elapsed_ms==0
    assert clock.observe(None,None,101).elapsed_ms==1000
    assert clock.observe(None,None,104).continuity_broken
