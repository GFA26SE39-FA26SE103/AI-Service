from dataclasses import dataclass
from types import SimpleNamespace
from uuid import UUID
import math
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
from app.monitoring_schemas import MonitoringZoneInput,StartMonitoringRequest,ZoneMeasurement
from app.roi_measurements import RoiMeasurements,TrackedPerson

@dataclass
class MonitoringFrameResult:
    annotated_frame: np.ndarray
    measurements: list[ZoneMeasurement]
    annotation_context: str

class MonitoringFrameTracker:
    def __init__(self,request: StartMonitoringRequest,jpeg_quality: int=85,max_frame_dimension: int=1280,*,detector=None,context_factory=None):
        self._request=request
        self._model=detector if detector is not None else YOLO(request.model)
        self._jpeg_quality=jpeg_quality; self._max_frame_dimension=max_frame_dimension
        self._context_factory=context_factory or BYTETracker
        self._args=SimpleNamespace(**YAML.load(check_yaml(request.tracker)))
        if self._args.tracker_type!="bytetrack": raise ValueError("AI_TRACKER_UNSUPPORTED")
        self.reconfigure(request.zones)

    def reconfigure(self,zones: list[MonitoringZoneInput]):
        self._zones=zones
        self._contexts={confidence:self._context_factory(self._args) for confidence in sorted({z.confidence for z in zones})}
        self._measurements=RoiMeasurements()

    def reset_tracking(self): self.reconfigure(self._zones)

    def process(self,frame: np.ndarray,elapsed_ms: int,continuity_id: UUID) -> MonitoringFrameResult:
        height,width=frame.shape[:2]
        if max(height,width)>self._max_frame_dimension:
            scale=self._max_frame_dimension/max(height,width)
            frame=cv2.resize(frame,(round(width*scale),round(height*scale)),interpolation=cv2.INTER_AREA)
            height,width=frame.shape[:2]
        results=self._model.predict(frame,classes=[0],conf=min(self._contexts),device=self._request.device,half=self._request.half,verbose=False)
        boxes=results[0].boxes.cpu().numpy()
        # Drop invalid boxes before tracker association, not just before counting.
        data=boxes.data
        valid=np.isfinite(data).all(axis=1)&(data[:,2]>data[:,0])&(data[:,3]>data[:,1])&(data[:,4]>=0)&(data[:,4]<=1)&(data[:,5]==0)
        boxes=boxes[valid]
        tracks_by_conf={}
        for confidence,context in self._contexts.items():
            filtered=boxes[boxes.conf>=confidence]
            tracked=context.update(filtered,frame)
            tracks=[]
            for row in tracked:
                if len(row)<8 or not all(math.isfinite(float(x)) for x in row): continue
                # BYTETracker returns confirmed tracks updated by current detections.
                if int(row[6])!=0 or row[4]<=0 or int(row[4])!=row[4]: continue
                tracks.append(TrackedPerson(int(row[4]),float(row[5]),(float(row[0])/width,float(row[1])/height,float(row[2])/width,float(row[3])/height),True))
            tracks_by_conf[confidence]=tracks
        measurements=[self._measurements.observe(zone,tracks_by_conf[zone.confidence],elapsed_ms,continuity_id) for zone in self._zones]
        lowest=min(self._contexts); label=f"confidence:{lowest:g}"
        annotated=frame.copy()
        for track in tracks_by_conf[lowest]:
            x1,y1,x2,y2=track.bbox
            left,top,right,bottom=round(x1*width),round(y1*height),round(x2*width),round(y2*height)
            cv2.rectangle(annotated,(left,top),(right,bottom),(255,0,0),2)
            cv2.putText(annotated,f"id:{track.track_id} person {track.score:.2f}",(left,max(16,top-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),2)
        cv2.putText(annotated,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        return MonitoringFrameResult(annotated,measurements,label)

    def encode_jpeg(self,frame: np.ndarray) -> bytes:
        ok,encoded=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),self._jpeg_quality])
        if not ok: raise RuntimeError("AI_FRAME_ENCODE_FAILED")
        return encoded.tobytes()
