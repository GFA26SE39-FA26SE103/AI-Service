from uuid import uuid4
import pytest
from app.roi_measurements import RoiMeasurements,TrackedPerson,contains_footpoint
from tests.test_monitoring_contract import zone

def test_footpoint_includes_boundary_not_box_overlap():
    roi=zone().roi_polygon
    assert contains_footpoint(roi,(.2,.1,.8,1))
    assert not contains_footpoint(roi,(.2,.1,.8,1.1))

def test_counts_distinct_current_valid_tracks_and_continuous_queue():
    z=zone(); epoch=uuid4(); measurements=RoiMeasurements()
    inside=TrackedPerson(1,.9,(.1,.1,.3,.8),True)
    outside=TrackedPerson(2,.9,(1.1,.1,1.2,.8),True)
    invalid=[TrackedPerson(3,.9,(.3,.1,.2,.8),True),TrackedPerson(4,float("nan"),(.1,.1,.3,.8),True),TrackedPerson(5,.9,(.1,.1,.3,.8),False)]
    assert measurements.observe(z,[inside,inside,outside,*invalid],0,epoch).people_count==1
    assert measurements.observe(z,[inside],4900,epoch).queue_count==0
    assert measurements.observe(z,[inside],5000,epoch).queue_count==1
    assert measurements.observe(z,[],5040,epoch).people_count==0
    assert measurements.observe(z,[inside],6000,epoch).queue_count==0
    assert measurements.observe(z,[inside],11000,epoch).queue_count==1
    assert measurements.observe(z,[inside],11040,uuid4()).queue_count==0

@pytest.mark.parametrize("bbox",[(0,0,float("inf"),1),(0,0,0,1),(0,0,1,float("nan"))])
def test_invalid_box_never_counts(bbox):
    assert RoiMeasurements().observe(zone(),[TrackedPerson(1,.9,bbox,True)],0,uuid4()).people_count==0
