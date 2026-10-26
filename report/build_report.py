"""Generate a eight-page English report; reportlab is needed only for authoring."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT/'MODEL_MANIFEST.json').read_text())
timing = json.loads((ROOT/'evaluation/validation-timing.json').read_text())
size = manifest['total_inference_bytes']/1024**2
seconds = timing['total_seconds']
pages = json.loads((ROOT/'report/content_en.json').read_text(encoding='utf-8'))

body = ParagraphStyle('Body', fontName='Helvetica', fontSize=10, leading=14, spaceAfter=9, textColor=colors.HexColor('#263248'))
head = ParagraphStyle('Head', parent=body, fontName='Helvetica-Bold', fontSize=12.2, leading=17, spaceBefore=6, spaceAfter=7, textColor=colors.HexColor('#155e75'))
title = ParagraphStyle('Title', parent=head, fontSize=18, leading=24, spaceAfter=17)
cell = ParagraphStyle('Cell', parent=body, fontSize=8.8, leading=11.5, spaceAfter=0)
story = []; md = ['# DASE7506 MP2 Report\n']
for index, (name, entries) in enumerate(pages):
    if index: story.append(PageBreak())
    story.append(Paragraph(escape(name), title)); md += ['## '+name, '']
    for kind, value in entries:
        if kind in ('p','h'):
            story.append(Paragraph(escape(value).replace('\n','<br/>'), head if kind=='h' else body))
            md += [('### ' if kind=='h' else '')+value, '']
        elif kind == 'fig':
            story.append(Image(str(ROOT/'report/figures'/f'{value}.png'), width=495, height=247.5))
            story.append(Spacer(1,8))
            md += [f'![{value}](figures/{value}.png)', '']
        else:
            width = A4[0]-100
            widths = [width*.47]+[width*.53/(len(value[0])-1)]*(len(value[0])-1)
            table = Table([[Paragraph(escape(str(c)), cell) for c in row] for row in value], colWidths=widths, repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1eef2')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story += [table, Spacer(1,10)]
            md += ['| '+' | '.join(value[0])+' |', '| '+' | '.join(['---']*len(value[0]))+' |']
            md += ['| '+' | '.join(row)+' |' for row in value[1:]]; md += ['']
def footer(c, doc):
    c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#64748b'))
    c.drawString(50, 28, 'wuhaonan | 3036843288 | DASE7506 MP2 | Test: 381/467 (81.58%)')
    c.drawRightString(A4[0]-50,28,str(doc.page))
doc = SimpleDocTemplate(str(ROOT/'report/REPORT.pdf'), pagesize=A4, rightMargin=50,leftMargin=50,topMargin=43,bottomMargin=48)
doc.build(story, onFirstPage=footer,onLaterPages=footer)
(ROOT/'report/REPORT.md').write_text('\n'.join(md),encoding='utf-8')
from pypdf import PdfReader
reader = PdfReader(ROOT/'report/REPORT.pdf')
print('pages',len(reader.pages))
if len(reader.pages) != 8: raise ValueError('Unexpected pagination: inspect layout')
