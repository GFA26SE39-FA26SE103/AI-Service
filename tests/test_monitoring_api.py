from uuid import uuid4
from datetime import datetime,timezone
from fastapi.testclient import TestClient
from app.main import create_app
from app.settings import Settings
from app.monitoring_schemas import MeasurementPage
from app.schemas import SessionState,SessionStatusResponse
from app.session_manager import SessionOwnershipError
from tests.test_monitoring_sessions import request

def test_private_monitoring_routes_validate_owner_protocol_and_redact_errors():
    camera=uuid4(); body=request(); session=uuid4()
    class TransportFake:
        async def start_monitoring(self,camera_id,submitted):
            assert submitted.owner_id==body.owner_id
            return SessionStatusResponse(camera_id=camera_id,session_id=session,state=SessionState.STARTING,updated_at=datetime.now(timezone.utc),purpose="MONITORING")
        async def measurements(self,camera_id,owner_id,after_session_id,after_sequence,limit):
            if owner_id!=body.owner_id: raise SessionOwnershipError("AI_SESSION_OWNER_MISMATCH")
            return MeasurementPage(session_id=session,oldest_sequence=1,newest_sequence=0,gap=False,batches=[],state=SessionState.STARTING)
        async def stop_monitoring(self,camera_id,owner_id):
            if owner_id!=body.owner_id: raise SessionOwnershipError("AI_SESSION_OWNER_MISMATCH")
            return SessionStatusResponse(camera_id=camera_id,state=SessionState.STOPPED,updated_at=datetime.now(timezone.utc))
    with TestClient(create_app(Settings(internal_service_key="test-key"),TransportFake())) as client:
        url=f"/monitoring/sessions/{camera}"
        headers={"X-AI-Service-Key":"test-key","X-AI-Monitoring-Owner":str(body.owner_id)}
        assert client.post(url+"/start",json=body.model_dump(mode="json")).status_code==401
        started=client.post(url+"/start",json=body.model_dump(mode="json"),headers=headers)
        assert started.status_code==200 and started.json()["purpose"]=="MONITORING"
        assert str(body.owner_id) not in started.text
        assert client.get(url+"/measurements",headers=headers).status_code==200
        bad=client.get(url+"/measurements",headers=headers|{"X-AI-Monitoring-Owner":str(uuid4())})
        assert bad.status_code==409 and bad.json()["code"]=="AI_SESSION_OWNER_MISMATCH"
        assert client.get(url+"/measurements?limit=65",headers=headers).status_code==422
        invalid=client.post(url+"/start",json=body.model_dump(mode="json")|dict(protocol_version=2,stream_url="http://secret-url/video",username="secret-user",password="secret-password"),headers=headers)
        assert invalid.status_code==422
        assert all(secret not in invalid.text for secret in ("secret-url","secret-user","secret-password"))
        assert client.delete(url,headers=headers).status_code==200
