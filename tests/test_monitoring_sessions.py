import asyncio
from uuid import uuid4
import pytest
from app.monitoring_schemas import StartMonitoringRequest,ZoneMeasurement
from app.monitoring_tracker import MonitoringFrameResult
from app.session_manager import SessionManager,SessionOwnershipError
from app.schemas import StartSessionRequest,SessionState
from tests.fakes import RecordingFactories,FakeTracker,frame
from tests.test_monitoring_contract import zone

class MonitoringFake(FakeTracker):
    def __init__(self,request): super().__init__(); self.zones=request.zones
    def process(self,value,elapsed_ms,continuity_id):
        return MonitoringFrameResult(value,[ZoneMeasurement(zone_id=z.zone_id,config_id=z.config_id,config_version=z.config_version,people_count=1,queue_count=0) for z in self.zones],"confidence:0.5")
    def reconfigure(self,zones): self.zones=zones
    def reset_tracking(self): pass

def request(**changes):
    return StartMonitoringRequest(**(dict(stream_url="http://camera.test/video",owner_id=uuid4(),configuration_fingerprint="one",zones=[zone()])|changes))

async def wait_for_frames(manager,camera,count=1):
    async with asyncio.timeout(3):
        while manager.status_now(camera).frame_sequence<count: await asyncio.sleep(.01)

@pytest.mark.asyncio
async def test_monitoring_preview_attaches_and_viewer_cannot_stop_owner():
    factories=RecordingFactories.with_sequences([(True,frame(1))]); made=[]
    def create(body): made.append(MonitoringFake(body)); return made[-1]
    manager=SessionManager(factories.reader_factory,factories.tracker_factory,monitoring_tracker_factory=create)
    camera=uuid4(); body=request()
    await manager.start_monitoring(camera,body); await wait_for_frames(manager,camera)
    before=manager.status_now(camera).session_id
    attached=await manager.start(camera,StartSessionRequest(stream_url="http://camera.test/video"))
    assert attached.purpose=="MONITORING" and attached.session_id==before and len(made)==1
    with pytest.raises(SessionOwnershipError,match="AI_SESSION_MONITORING_OWNED"): await manager.stop(camera)
    with pytest.raises(SessionOwnershipError,match="AI_SESSION_OWNER_MISMATCH"): await manager.measurements(camera,uuid4(),None,0,64)
    page=await manager.measurements(camera,body.owner_id,None,0,64)
    assert page.batches[0].zones[0].people_count==1
    assert str(body.owner_id) not in (await manager.status(camera)).model_dump_json()
    await manager.stop_monitoring(camera,body.owner_id)
    assert manager.status_now(camera).state==SessionState.STOPPED

@pytest.mark.asyncio
async def test_completed_session_not_replayed_and_reconfigure_does_not_reopen_source():
    factories=RecordingFactories.with_sequences([(True,frame(1)),(True,frame(2)),(False,None)])
    def reader(body):
        r=factories.reader_factory(body); r.fps=10; r.frame_interval=.1; r.position_ms=None; return r
    manager=SessionManager(reader,factories.tracker_factory,monitoring_tracker_factory=MonitoringFake)
    camera=uuid4(); body=request(source_type="RECORDED")
    await manager.start_monitoring(camera,body); await wait_for_frames(manager,camera)
    second=body.model_copy(update=dict(configuration_fingerprint="two",zones=[body.zones[0],zone()]))
    await manager.start_monitoring(camera,second)
    async with asyncio.timeout(3):
        while manager.status_now(camera).state not in (SessionState.COMPLETED,SessionState.ERROR): await asyncio.sleep(.01)
    assert manager.status_now(camera).state==SessionState.COMPLETED
    assert len(factories.readers)==1 and manager.status_now(camera).frame_sequence==2
    await manager.start(camera,StartSessionRequest(stream_url=body.stream_url))
    assert len(factories.readers)==1
    page=await manager.measurements(camera,body.owner_id,None,0,64)
    assert page.batches[-1].configuration_fingerprint=="two" and len(page.batches[-1].zones)==2
    assert page.batches[-1].sequence==2
    assert (await manager.measurements(camera,body.owner_id,page.session_id,2,64)).batches==[]
    await manager.stop_monitoring(camera,body.owner_id)

@pytest.mark.asyncio
async def test_viewer_polling_does_not_renew_monitoring_owner_lease():
    factories=RecordingFactories.with_sequences([(True,frame(1))])
    manager=SessionManager(factories.reader_factory,factories.tracker_factory,monitoring_tracker_factory=MonitoringFake,session_idle_timeout_seconds=.08)
    camera=uuid4(); body=request(); await manager.start_monitoring(camera,body)
    async with asyncio.timeout(3):
        while manager.status_now(camera).state!=SessionState.STOPPED:
            await manager.status(camera); await asyncio.sleep(.01)
    assert manager.active_worker_count==0

@pytest.mark.asyncio
async def test_reconfigure_at_eof_updates_version_without_replaying_or_serving_old_measurements():
    factories=RecordingFactories.with_sequences([(True,frame(1)),(False,None)])
    def reader(body):
        r=factories.reader_factory(body); r.fps=10; r.frame_interval=.01; r.position_ms=None; return r
    manager=SessionManager(reader,factories.tracker_factory,monitoring_tracker_factory=MonitoringFake)
    camera=uuid4(); body=request(source_type="RECORDED")
    await manager.start_monitoring(camera,body)
    async with asyncio.timeout(3):
        while manager.status_now(camera).state!=SessionState.COMPLETED: await asyncio.sleep(.01)
    second=body.model_copy(update=dict(configuration_fingerprint="new-at-eof",zones=[zone()]))
    status=await manager.start_monitoring(camera,second)
    assert status.state==SessionState.COMPLETED and status.configuration_fingerprint=="new-at-eof"
    assert status.annotation_context is None
    assert len(factories.readers)==1
    page=await manager.measurements(camera,body.owner_id,None,0,64)
    assert not page.batches
    await manager.stop_monitoring(camera,body.owner_id)

@pytest.mark.asyncio
async def test_reconnect_keeps_order_but_resets_tracking_continuity():
    factories=RecordingFactories.with_sequences([(True,frame(1)),(False,None)],[(True,frame(2))])
    made=[]
    class ResetRecording(MonitoringFake):
        resets=0
        def reset_tracking(self): self.resets+=1
    def tracker(body):
        item=ResetRecording(body); made.append(item); return item
    manager=SessionManager(factories.reader_factory,factories.tracker_factory,monitoring_tracker_factory=tracker,reconnect_delay_seconds=0)
    camera=uuid4(); body=request()
    try:
        await manager.start_monitoring(camera,body); await wait_for_frames(manager,camera,2)
        page=await manager.measurements(camera,body.owner_id,None,0,64)
        assert [b.sequence for b in page.batches[:2]]==[1,2]
        assert page.batches[0].continuity_id!=page.batches[1].continuity_id
        assert page.batches[1].source_elapsed_ms>=page.batches[0].source_elapsed_ms
        assert made[0].resets>=1 and len(factories.readers)==2
    finally:
        await manager.shutdown()
