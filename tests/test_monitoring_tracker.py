from uuid import uuid4
from types import SimpleNamespace
import numpy as np
from ultralytics.engine.results import Boxes
from app.monitoring_tracker import MonitoringFrameTracker
from app.monitoring_schemas import StartMonitoringRequest
from tests.test_monitoring_contract import zone

def test_one_detector_pass_filters_before_each_confidence_context():
    zones=[zone(confidence=.5),zone(confidence=.8),zone(confidence=.5)]
    calls=[]; contexts=[]
    class Detector:
        def predict(self,frame,**kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=Boxes(np.array([[10,10,30,80,.6,0],[40,10,60,80,.9,0]],dtype=np.float32),(100,100)))]
    class Context:
        def __init__(self): self.scores=[]; contexts.append(self)
        def update(self,boxes,img=None):
            self.scores=list(boxes.conf)
            return np.array([[*b.xyxy[0],i+1,float(b.conf[0]),0,i] for i,b in enumerate(boxes)],dtype=np.float32)
    request=StartMonitoringRequest(stream_url="http://camera.test/video",owner_id=uuid4(),configuration_fingerprint="f",zones=zones)
    tracker=MonitoringFrameTracker(request,detector=Detector(),context_factory=lambda _:Context())
    result=tracker.process(np.zeros((100,100,3),dtype=np.uint8),0,uuid4())
    assert len(calls)==1 and len(contexts)==2
    assert [m.people_count for m in result.measurements]==[2,1,2]
    assert sorted(len(c.scores) for c in contexts)==[1,2]
    assert result.annotation_context=="confidence:0.5"

def test_installed_bytetrack_accepts_numpy_boxes_without_gpu():
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    args=SimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    tracker=BYTETracker(args)
    result=tracker.update(Boxes(np.array([[10,10,30,80,.9,0]],dtype=np.float32),(100,100)).cpu().numpy(),np.zeros((100,100,3),dtype=np.uint8))
    assert result.shape==(1,8)
    assert result[0,4]>0
