import numpy as np
from scipy.optimize import linear_sum_assignment
 
def iou(a, b):
    xa1,ya1,xa2,ya2=a; xb1,yb1,xb2,yb2=b
    xi1,yi1,xi2,yi2=max(xa1,xb1),max(ya1,yb1),min(xa2,xb2),min(ya2,yb2)
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    ua=(xa2-xa1)*(ya2-ya1)+(xb2-xb1)*(yb2-yb1)-inter
    return inter/ua if ua>0 else 0
 
class KalmanBoxTracker:
    count=0
    def __init__(self, bbox):
        KalmanBoxTracker.count+=1; self.id=KalmanBoxTracker.count
        x1,y1,x2,y2=bbox
        cx,cy,s,r=(x1+x2)/2,(y1+y2)/2,(x2-x1)*(y2-y1),(x2-x1)/(y2-y1+1e-6)
        self.state=np.array([cx,cy,s,r,0,0,0],float)
        self.hits=1; self.misses=0
    def predict(self):
        self.state[:3]+=self.state[4:7]
        return self._to_bbox()
    def update(self, bbox):
        x1,y1,x2,y2=bbox
        cx,cy=(x1+x2)/2,(y1+y2)/2; s=(x2-x1)*(y2-y1); r=(x2-x1)/(y2-y1+1e-6)
        self.state[:4]=[cx,cy,s,r]; self.hits+=1; self.misses=0
    def _to_bbox(self):
        cx,cy,s,r=self.state[:4]
        w=np.sqrt(abs(s*r)); h=abs(s/w) if w>0 else 0
        return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]
 
class SORT:
    def __init__(self, max_age=5, min_hits=2, iou_thr=0.3):
        self.max_age=max_age; self.min_hits=min_hits; self.iou_thr=iou_thr; self.trackers=[]
    def update(self, dets):
        preds=[t.predict() for t in self.trackers]
        matched,unmatched_d,unmatched_t=self._associate(dets,preds)
        for di,ti in matched: self.trackers[ti].update(dets[di])
        for di in unmatched_d: self.trackers.append(KalmanBoxTracker(dets[di]))
        alive=[t for t in self.trackers if t.misses<self.max_age]
        for ti in sorted(unmatched_t,reverse=True):
            if ti<len(self.trackers): self.trackers[ti].misses+=1
        self.trackers=alive
        return [(t._to_bbox(),t.id) for t in self.trackers if t.hits>=self.min_hits]
    def _associate(self, dets, preds):
        if not preds: return [],[list(range(len(dets)))],[]
        iou_mat=np.array([[iou(d,p) for p in preds] for d in dets])
        di,ti=linear_sum_assignment(-iou_mat)
        matched=[(d,t) for d,t in zip(di,ti) if iou_mat[d,t]>=self.iou_thr]
        matched_d={m[0] for m in matched}; matched_t={m[1] for m in matched}
        return matched,[d for d in range(len(dets)) if d not in matched_d],[t for t in range(len(preds)) if t not in matched_t]
 
sort=SORT()
for frame in range(5):
    dets=[[100+frame*5,100,150+frame*5,200],[300,200,400,300]]
    tracks=sort.update(dets)
    print(f"Frame {frame}: {[(f'ID {tid}',bbox) for bbox,tid in tracks]}")
