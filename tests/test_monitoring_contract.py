from datetime import datetime,timezone
from uuid import uuid4
import pytest
from pydantic import ValidationError
from app.monitoring_schemas import MonitoringZoneInput,StartMonitoringRequest,ZoneMeasurement

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
