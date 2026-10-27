"""
Figure 1 | Study design.  Nature style, double column 183 mm.
Hand-written SVG (a schematic, not a data plot) — named layer groups so it opens
cleanly in Illustrator / Inkscape / BioRender.

All type at final size: 8 pt panel tags, 7 pt headings, 6 pt labels, 5.5 pt detail.
Nature floor is 5 pt; nothing here goes below 5.5.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
import figstyle as FS
from panel_counts import counts as _counts, COLUMNS as SETTINGS

PT = 0.3528                      # 1 pt in mm
W, H = 183.0, 134.0
INK, MUTED, RULE = FS.INK, FS.MUTED, FS.RULE
SH  = FS.LEVEL_SHADES            # neutral, so the only colour in this figure is the family column
SHI = FS.LEVEL_INK
FAM = [(n, FS.FAMILY_COLORS[n]) for n in
       ["Marker-based", "Similarity-based", "Classical ML", "Deep learning",
        "Semi-supervised", "LLM", "Foundation model"]]
# Panel b is DERIVED, never typed: every cell is a count of tool configurations
# actually present in that arm's tidy file (see _shared/panel_counts.py).
MATRIX = _counts()
L9 = [(0,0,0,0),(0,1,1,1),(0,2,2,2),(1,0,1,2),(1,1,2,0),
      (1,2,0,1),(2,0,2,1),(2,1,0,2),(2,2,1,0)]
ROWS = ["Cells", "Class imbalance", "Cell types", "DE difficulty"]
LVL  = [["500","3,000","15,000"], ["balanced","mild (5:1)","severe (20:1)"],
        ["5","15","30"], ["easy (4.8×)","medium (2.4×)","hard (1.6×)"]]
# Nowakowski: 47 labelled WGCNA clusters; Zhao: 29 labelled "fine" types. The unlabelled
# cells (92 and 1,636) are not a type (V34 fix).
REAL = [("Darmanis","human, 9"),("Marques","mouse, 13"),("Nowakowski","human, 47"),
        ("Grün","human, 9"),("Tabula Muris","mouse, 30"),("He","human, 21"),
        ("Zhao","human, 29"),("Zheng","human, 10"),("MacParland","human, 8")]
STAGE = [("Synthetic", "9 scenarios × 3 seeds", "split reference or markers", "Cohen’s κ"),
         ("Real", "9 matched datasets", "split reference or markers", "Cohen’s κ"),
         ("Cross-platform", "CellBench · pancreas · PBMC", "other-platform reference", "Cohen’s κ"),
         ("Database &amp; LLM", "CellMarker 2.0 or LLMs", "no matched reference", "ontology score"),
         ("Foundation models", "5 pre-trained models", "full fine-tuning", "Cohen’s κ")]

o=[]; A=o.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">')
A(f'<style>text{{font-family:"Liberation Sans",Helvetica,Arial,sans-serif;fill:{INK}}}'
  f'.tag{{font-size:{8*PT:.3f}px;font-weight:700}}'
  f'.hd{{font-size:{7*PT:.3f}px;font-weight:700}}'
  f'.lab{{font-size:{6*PT:.3f}px}}'
  f'.cell{{font-size:{5.5*PT:.3f}px}}'
  f'.sml{{font-size:{5.5*PT:.3f}px;fill:{MUTED}}}</style>')
A(f'<rect width="{W}" height="{H}" fill="white"/>')

# ============================================================ a — datasets
# V33 (ZZ #18, #29): level values printed in the cells; measured kNN purity added for the
# synthetic scenarios and the matched real datasets (read from the tidy file, not typed).
import pandas as _pd
_tk = _pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared", "tidy_kappa.csv"))
_tk["sid"] = _tk.scenario.str.extract(r"^(S\d)")[0]
PUR = _tk.groupby(["arm", "sid"]).knn_purity.first()
A('<g id="a-datasets">')
A('<text class="tag" x="2" y="5.4">a</text>')
gx, gy, cw, ch, gp = 34.0, 8.6, 15.9, 5.0, 0.55
for j in range(9):
    A(f'<text class="sml" x="{gx+j*(cw+gp)+cw/2:.2f}" y="{gy-1.0:.2f}" text-anchor="middle">S{j+1}</text>')
for i, row in enumerate(ROWS):
    y = gy + i*(ch+gp)
    A(f'<text class="lab" x="{gx-1.8:.2f}" y="{y+ch*0.64:.2f}" text-anchor="end">{row}</text>')
    for j, sc in enumerate(L9):
        lv = sc[i]; x = gx + j*(cw+gp)
        A(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw}" height="{ch}" fill="{SH[lv]}"/>')
        A(f'<text class="cell" x="{x+cw/2:.2f}" y="{y+ch*0.64:.2f}" text-anchor="middle" '
          f'fill="{SHI[lv]}">{LVL[i][lv]}</text>')
# measured separability of each synthetic scenario (not a design factor)
py = gy + 4*(ch+gp) + 0.9
A(f'<text class="lab" x="{gx-1.8:.2f}" y="{py+ch*0.64:.2f}" text-anchor="end">kNN purity (measured)</text>')
for j in range(9):
    x = gx + j*(cw+gp)
    A(f'<rect x="{x:.2f}" y="{py:.2f}" width="{cw}" height="{ch}" fill="white" stroke="{RULE}" stroke-width="0.3"/>')
    A(f'<text class="cell" x="{x+cw/2:.2f}" y="{py+ch*0.64:.2f}" text-anchor="middle">'
      f'{PUR["synth", f"S{j+1}"]:.2f}</text>')
ry = py + ch + 1.6
A(f'<line x1="{gx:.2f}" y1="{ry-0.9:.2f}" x2="{gx+9*(cw+gp)-gp:.2f}" y2="{ry-0.9:.2f}" '
  f'stroke="{RULE}" stroke-width="0.4"/>')
A(f'<text class="lab" x="{gx-1.8:.2f}" y="{ry+2.9:.2f}" text-anchor="end">Matched real dataset</text>')
for j,(nm,sp) in enumerate(REAL):
    x = gx + j*(cw+gp)
    A(f'<text class="cell" x="{x+cw/2:.2f}" y="{ry+2.9:.2f}" text-anchor="middle">{nm}</text>')
    A(f'<text class="sml" x="{x+cw/2:.2f}" y="{ry+6.1:.2f}" text-anchor="middle">{sp}</text>')
    A(f'<text class="cell" x="{x+cw/2:.2f}" y="{ry+9.3:.2f}" text-anchor="middle">'
      f'{PUR["real", f"S{j+1}"]:.2f}</text>')
A(f'<text class="sml" x="{gx-1.8:.2f}" y="{ry+6.1:.2f}" text-anchor="end">species, cell types</text>')
A(f'<text class="sml" x="{gx-1.8:.2f}" y="{ry+9.3:.2f}" text-anchor="end">kNN purity</text>')
A('</g>')

# ============================================================ b — evaluation settings
# V33 (ZZ #18): drawn as a workflow. Shared preprocessing feeds five settings that run in
# parallel (no arrows between them); each box names its inputs and its scoring metric.
A('<g id="b-evaluation-settings">')
cy = ry + 15.6
A(f'<text class="tag" x="2" y="{cy:.2f}">b</text>')
bw, bg, bx0 = 31.4, 2.6, 8.0
bx1 = bx0 + 5*bw + 4*bg
A(f'<rect x="{bx0:.2f}" y="{cy+1.6:.2f}" width="{bx1-bx0:.2f}" height="4.5" fill="#f5f5f5" '
  f'stroke="{RULE}" stroke-width="0.4"/>')
A(f'<text class="lab" x="{bx0+2.0:.2f}" y="{cy+4.55:.2f}" font-weight="700">Shared preprocessing</text>')
A(f'<text class="lab" x="{bx0+27.5:.2f}" y="{cy+4.55:.2f}">(Seurat / Scanpy): quality filtering · '
  f'log-normalization · 2,000 highly variable genes · 50 principal components</text>')
for i,(name,sub1,sub2,metric) in enumerate(STAGE):
    x = bx0 + i*(bw+bg); xc = x + bw/2
    A(f'<line x1="{xc:.2f}" y1="{cy+6.1:.2f}" x2="{xc:.2f}" y2="{cy+7.6:.2f}" stroke="{MUTED}" stroke-width="0.4"/>')
    A(f'<rect x="{x:.2f}" y="{cy+7.6:.2f}" width="{bw}" height="12.4" fill="white" '
      f'stroke="{RULE}" stroke-width="0.4"/>')
    A(f'<text class="lab" x="{xc:.2f}" y="{cy+11.9:.2f}" text-anchor="middle" font-weight="700">{name}</text>')
    A(f'<text class="sml" x="{xc:.2f}" y="{cy+15.3:.2f}" text-anchor="middle">{sub1}</text>')
    A(f'<text class="sml" x="{xc:.2f}" y="{cy+18.2:.2f}" text-anchor="middle">{sub2}</text>')
    A(f'<line x1="{xc:.2f}" y1="{cy+20.0:.2f}" x2="{xc:.2f}" y2="{cy+21.4:.2f}" stroke="{MUTED}" stroke-width="0.4"/>')
    A(f'<rect x="{x+4:.2f}" y="{cy+21.4:.2f}" width="{bw-8:.2f}" height="4.0" fill="white" '
      f'stroke="{RULE}" stroke-width="0.4" rx="0.6"/>')
    A(f'<text class="sml" x="{xc:.2f}" y="{cy+24.1:.2f}" text-anchor="middle">{metric}</text>')
A('</g>')

# ============================================================ b — where each family is evaluated
A('<g id="c-evaluation-matrix">')
by = cy + 32.0
A(f'<text class="tag" x="2" y="{by:.2f}">c</text>')
tx, ty, rh = 8.0, by + 8.4, 4.05
BCOLS = [("Synthetic", 55.0), ("Real", 76.0), ("Cross-\nplatform", 97.0),
         ("Database\n&amp; LLM", 118.0), ("Fine-tuned\nFM", 139.0)]
TOTX = 165.5
A(f'<text class="sml" x="{tx:.2f}" y="{ty-3.0:.2f}" font-weight="700">Family</text>')
for name, cx in BCOLS:
    ln = name.split("\n")
    for k, s in enumerate(ln):
        A(f'<text class="sml" x="{cx:.2f}" y="{ty-5.6+(k+(2-len(ln)))*2.6:.2f}" text-anchor="middle">{s}</text>')
for k, s in enumerate(["Tools /", "Configurations"]):
    A(f'<text class="sml" x="{TOTX:.2f}" y="{ty-5.6+k*2.6:.2f}" text-anchor="middle" font-weight="700">{s}</text>')
# vertical rule setting off the roster total
A(f'<line x1="152.0" y1="{ty-7.4:.2f}" x2="152.0" y2="{ty+7*rh+4.6:.2f}" stroke="{RULE}" stroke-width="0.4"/>')
A(f'<line x1="{tx:.2f}" y1="{ty-1.6:.2f}" x2="179" y2="{ty-1.6:.2f}" stroke="{RULE}" stroke-width="0.4"/>')

def _cell(v):
    return ("&#8212;", MUTED) if not v else (str(v), INK)

for i,(fam,col) in enumerate(FAM):
    y = ty + i*rh
    A(f'<rect x="{tx:.2f}" y="{y+0.75:.2f}" width="2.6" height="2.6" fill="{col}"/>')
    A(f'<text class="lab" x="{tx+4.2:.2f}" y="{y+2.95:.2f}">{fam}</text>')
    row = MATRIX[fam]
    for name, cx in BCOLS:
        txt, fill = _cell(row.get(name.replace("\n"," ").replace("&amp;","/").strip()
                                  if False else {"Synthetic":"Synthetic","Real":"Real",
                                  "Cross-\nplatform":"Cross-platform",
                                  "Database\n&amp; LLM":"Database / LLM",
                                  "Fine-tuned\nFM":"Fine-tuned FM"}[name]))
        A(f'<text class="cell" x="{cx:.2f}" y="{y+2.95:.2f}" text-anchor="middle" fill="{fill}">{txt}</text>')
    A(f'<text class="cell" x="{TOTX:.2f}" y="{y+2.95:.2f}" text-anchor="middle" '
      f'font-weight="700">{row.get("Total","")}</text>')
    if i < len(FAM)-1:
        A(f'<line x1="{tx:.2f}" y1="{y+rh:.2f}" x2="179" y2="{y+rh:.2f}" stroke="#ececec" stroke-width="0.3"/>')

# total row
tyr = ty + 7*rh
A(f'<line x1="{tx:.2f}" y1="{tyr+0.5:.2f}" x2="179" y2="{tyr+0.5:.2f}" stroke="{RULE}" stroke-width="0.4"/>')
A(f'<text class="lab" x="{tx+4.2:.2f}" y="{tyr+4.0:.2f}" font-weight="700">Tools/Configurations</text>')
for name, cx in BCOLS:
    key = {"Synthetic":"Synthetic","Real":"Real","Cross-\nplatform":"Cross-platform",
           "Database\n&amp; LLM":"Database / LLM","Fine-tuned\nFM":"Fine-tuned FM"}[name]
    tot = sum(MATRIX[f].get(key,0) for f,_ in FAM)
    A(f'<text class="cell" x="{cx:.2f}" y="{tyr+4.0:.2f}" text-anchor="middle" font-weight="700">{tot}</text>')
A(f'<text class="cell" x="{TOTX:.2f}" y="{tyr+4.0:.2f}" text-anchor="middle" font-weight="700">'
  f'{sum(MATRIX[f].get("Total",0) for f,_ in FAM)}</text>')
A('</g>')

A('</svg>')


open("Figure1.svg","w").write("\n".join(o))
import cairosvg
from PIL import Image
cairosvg.svg2png(url="Figure1.svg", write_to="Figure1.png", scale=8)
cairosvg.svg2pdf(url="Figure1.svg", write_to="Figure1.pdf")
Image.open("Figure1.png").convert("L").save("Figure1_grayscale.png")
print("Figure1:", W, "x", H, "mm")
