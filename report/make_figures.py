"""Generate vector figures from our own recorded experiments; no new model runs."""
from pathlib import Path
import json, subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'report/figures'; OUT.mkdir(exist_ok=True)
def start(name,title):
    c=canvas.Canvas(str(OUT/(name+'.pdf')),pagesize=(640,320))
    c.setFillColor(HexColor('#123047'));c.setFont('Helvetica-Bold',16);c.drawString(25,292,title)
    return c
def label(c,x,y,text,size=10):
    c.setFillColor(HexColor('#123047'));c.setFont('Helvetica',size);c.drawString(x,y,text)
c=start('architecture','Frozen selector: candidate comparison + reliability signals')
blocks=[(25,187,165,65,['Four candidate solutions','1024-d frozen embeddings','32 input-only features']),
        (238,187,165,65,['Training normalization','Shared 1056 -> 64 layer','ReLU + dropout (train only)']),
        (450,187,165,65,['Single-head attention','Q, K, V across candidates','No positional embedding']),
        (450,65,165,65,['Own state + context','128 -> 1 shared scorer','Four candidate scores']),
        (238,65,165,65,['Official parseability mask','Choose highest score','One index in {0,1,2,3}'])]
for x,y,w,h,lines in blocks:
    c.setFillColor(HexColor('#edf6fa'));c.setStrokeColor(HexColor('#2d7790'));c.roundRect(x,y,w,h,8,fill=1)
    for k,line in enumerate(lines):label(c,x+10,y+h-17-k*17,line,9)
for x1,y1,x2,y2 in [(190,220,238,220),(403,220,450,220),(532,187,532,130),(450,97,403,97)]:
    c.line(x1,y1,x2,y2)
label(c,25,35,'Training: maximize total probability of all correct candidates. Inference: no references.',10)
c.save()
old=json.loads((ROOT/'records/stability/results.json').read_text())
new=json.loads((ROOT/'records/reliability/results.json').read_text())
seeds=[7506,17,42,2026,3407]
a=[next(r['accuracy'] for r in old if r['seed']==s) for s in seeds]
b=[next(r['accuracy'] for r in new if r['name']==f'both_seed_{s}') for s in seeds]
c=start('seed_stability','Same five seeds: modest mean gain, greater variability')
x0,y0=65,63;w,h=520,190
for v in [91,91.5,92,92.5,93,93.5]:
    y=y0+(v-91)/2.5*h;c.setStrokeColor(HexColor('#d9e2e8'));c.line(x0,y,x0+w,y);label(c,23,y-3,str(v))
for values,color in [(a,'#64748b'),(b,'#087e8b')]:
    c.setStrokeColor(HexColor(color));c.setLineWidth(2)
    pts=[(x0+35+i*110,y0+(v-91)/2.5*h) for i,v in enumerate(values)]
    for p,q in zip(pts,pts[1:]):c.line(*p,*q)
    for x,y in pts:c.setFillColor(HexColor(color));c.circle(x,y,4,fill=1,stroke=0)
for i,s in enumerate(seeds):label(c,x0+23+i*110,43,str(s))
label(c,65,20,'Grey: original 91.88% +/- 0.30 pp     Teal: final features 92.36% +/- 0.54 pp',10)
label(c,65,269,'Validation accuracy (%) | axis starts at 91%; SD is across training seeds',10)
c.save()
rs=json.loads((ROOT/'records/ensembles/results.json').read_text())
c=start('accuracy_cost','More inference assets did not improve validation accuracy')
x0,y0=65,65;w,h=515,190
for v in [92,92.5,93,93.5]:
    y=y0+(v-92)/1.5*h;c.setStrokeColor(HexColor('#d9e2e8'));c.line(x0,y,x0+w,y);label(c,22,y-3,str(v))
for v in [0,1,2,3,4]:label(c,x0+v/4*w-3,45,str(v))
for r in rs:
    x=x0+r['checkpoint_bytes']/1024**2/4*w;y=y0+(r['accuracy']-92)/1.5*h
    c.setFillColor(HexColor('#087e8b'));c.circle(x,y,4,fill=1,stroke=0)
    dy=10 if r['name'] not in ['old_single','old_five'] else -17
    label(c,x+5,y+dy,r['name'],9)
label(c,65,20,'Total inference assets (MiB); all scores are validation, not test.',10)
label(c,65,269,'Validation accuracy (%) | axis starts at 92%',10)
c.save()
print(OUT)
