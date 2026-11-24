from uuid import uuid4
import pytest
from app.roi_measurements import RoiMeasurements,TrackedPerson,contains_footpoint
from tests.test_monitoring_contract import zone
from tests.test_monitoring_contract import checkout_mapping
from app.monitoring_schemas import CheckoutMappingInput
from app.roi_measurements import CheckoutRoiMeasurements

def test_checkout_queue_dwell_and_staff_presence_are_independent():
    mapping=CheckoutMappingInput(**checkout_mapping(staff_roi=[dict(x=0,y=0),dict(x=.4,y=0),dict(x=.4,y=1),dict(x=0,y=1)],
        lane_roi=[dict(x=.5,y=0),dict(x=1,y=0),dict(x=1,y=1),dict(x=.5,y=1)],
        service_roi=[dict(x=.9,y=0),dict(x=1,y=0),dict(x=1,y=1),dict(x=.9,y=1)]))
    epoch=uuid4(); measurements=CheckoutRoiMeasurements()
    lane=TrackedPerson(1,.9,(.6,.2,.8,.8),True)
    staff=TrackedPerson(2,.9,(.1,.2,.3,.8),True)
    first=measurements.observe(mapping,[lane,staff],0,epoch)
    assert first.person_present is True and first.queue_count==0
    assert first.lane_people_count==1
    assert measurements.observe(mapping,[lane],5000,epoch).queue_count==1
    assert measurements.observe(mapping,[lane],5000,epoch).max_wait_seconds==5
    served=TrackedPerson(1,.9,(.9,.2,1,.8),True)
    serving=measurements.observe(mapping,[served],5100,epoch)
    assert serving.max_wait_seconds==0 and serving.lane_people_count==0
    restarted=measurements.observe(mapping,[lane],5001,uuid4())
    assert restarted.queue_count==0 and restarted.lane_people_count==1
    other=CheckoutMappingInput(**checkout_mapping(is_queue_source=False))
    secondary=measurements.observe(other,[],0,epoch)
    assert secondary.queue_count is None and secondary.lane_people_count is None

def test_footpoint_includes_boundary_not_box_overlap():
    roi=zone().roi_polygon
    assert contains_footpoint(roi,(.2,.1,.8,1))
    assert not contains_footpoint(roi,(.2,.1,.8,1.1))

def test_zone_uses_visible_center_for_bottom_cropped_person():
    monitored_zone = zone(roi_polygon=[
        dict(x=.25, y=.25), dict(x=.75, y=.25), dict(x=.75, y=.75), dict(x=.25, y=.75),
    ])
    epoch = uuid4()
    measurements = RoiMeasurements()
    cropped = TrackedPerson(1, .9, (.4, .2, .6, .995), True)
    full_box = TrackedPerson(2, .9, (.4, .2, .6, .9), True)
    outside = TrackedPerson(3, .9, (.8, .2, .9, 1), True)
    assert measurements.observe(monitored_zone, [full_box, outside], 0, epoch).people_count == 0
    first = measurements.observe(monitored_zone, [cropped], 10, epoch)
    assert first.people_count == 1 and first.queue_count == 0
    assert measurements.observe(monitored_zone, [cropped], 5009, epoch).queue_count == 0
    assert measurements.observe(monitored_zone, [cropped], 5010, epoch).queue_count == 1

def test_multiple_rois_use_largest_individual_count_not_sum():
    left=[dict(x=0,y=0),dict(x=.45,y=0),dict(x=.45,y=1),dict(x=0,y=1)]
    right=[dict(x=.55,y=0),dict(x=1,y=0),dict(x=1,y=1),dict(x=.55,y=1)]
    monitored=zone(roi_polygon=left,additional_roi_polygons=[right])
    tracks=[TrackedPerson(1,.9,(.1,.2,.2,.8),True),TrackedPerson(2,.9,(.25,.2,.35,.8),True),
            TrackedPerson(3,.9,(.65,.2,.75,.8),True),TrackedPerson(4,.9,(.8,.2,.9,.8),True)]
    measurements=RoiMeasurements(); epoch=uuid4()
    assert measurements.observe(monitored,tracks,0,epoch).people_count==2
    result=measurements.observe(monitored,tracks,5000,epoch)
    assert result.people_count==2 and result.queue_count==2

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
