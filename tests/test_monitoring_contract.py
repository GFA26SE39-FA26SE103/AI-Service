from datetime import datetime,timezone
from uuid import UUID,uuid4
import pytest
from pydantic import ValidationError
from app.monitoring_schemas import MonitoringZoneInput,StartMonitoringRequest,ZoneMeasurement

def checkout_mapping(**changes):
    points=[dict(x=0,y=0),dict(x=1,y=0),dict(x=1,y=1),dict(x=0,y=1)]
    data=dict(mapping_id=uuid4(),counter_id=uuid4(),zone_id=uuid4(),config_id=uuid4(),
              config_version=datetime.now(timezone.utc),mapping_version=datetime.now(timezone.utc),
              lane_roi=points,staff_roi=points,is_queue_source=True,confidence=.5)
    return data|changes

def test_checkout_v2_accepts_checkout_only_and_rejects_invalid_sources():
    base=dict(stream_url="http://private.test/video",owner_id=uuid4(),configuration_fingerprint="f",protocol_version=2,zones=[])
    mapping=checkout_mapping()
    request=StartMonitoringRequest(**base,checkout_mappings=[mapping])
    assert request.checkout_mappings[0].mapping_id==mapping["mapping_id"]
    for changes in [dict(checkout_mappings=[]),dict(checkout_mappings=[mapping,mapping]),
                    dict(checkout_mappings=[checkout_mapping(lane_roi=[dict(x=0,y=0)])]),
                    dict(checkout_mappings=[checkout_mapping(mapping_version=datetime.now())]),
                    dict(checkout_mappings=[checkout_mapping(counter_id=UUID(int=0))])]:
        with pytest.raises(ValidationError): StartMonitoringRequest(**(base|changes))
    with pytest.raises(ValidationError): StartMonitoringRequest(**(base|dict(protocol_version=1,zones=[zone()],checkout_mappings=[mapping])))

def zone(**changes):
    data=dict(zone_id=uuid4(),config_id=uuid4(),config_version=datetime.now(timezone.utc),confidence=.5,queue_enabled=True,roi_polygon=[dict(x=0,y=0),dict(x=1,y=0),dict(x=1,y=1),dict(x=0,y=1)])
    return MonitoringZoneInput(**(data|changes))

@pytest.mark.parametrize("changes",[dict(confidence=float("nan")),dict(confidence=1.1),dict(roi_polygon=[dict(x=0,y=0)]),dict(roi_polygon=[dict(x=0,y=0),dict(x=1,y=1),dict(x=0,y=1),dict(x=1,y=0)])])
def test_rejects_invalid_zone_geometry_and_confidence(changes):
    with pytest.raises(ValidationError): zone(**changes)

def test_counts_are_finite_nonnegative_integers_and_queue_is_subset():
    z=zone()
    for people,queue in [(-1,0),(1,2),(1.5,0),(float("inf"),0),(True,0)]:
        with pytest.raises(ValidationError): ZoneMeasurement(zone_id=z.zone_id,config_id=z.config_id,config_version=z.config_version,people_count=people,queue_count=queue)

def test_rejects_protocol_duplicates_and_hides_private_request_values():
    z=zone()
    data=dict(stream_url="http://private.test/video",username="secret-user",password="secret-password",owner_id=uuid4(),configuration_fingerprint="fingerprint",zones=[z])
    request=StartMonitoringRequest(**data)
    assert "private.test" not in repr(request) and "secret-user" not in repr(request) and "secret-password" not in repr(request)
    assert str(request.owner_id) not in repr(request)
    for changes in [dict(protocol_version=2),dict(zones=[z,z]),dict(owner_id="invalid")]:
        with pytest.raises(ValidationError): StartMonitoringRequest(**(data|changes))
