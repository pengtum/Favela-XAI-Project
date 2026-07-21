# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project/Publications/2026_Favela_XAI_NC/2026_Favela_NatureCities/manuscript_Favela_XAI_morphology/cover_letter_NatureCities.docx"
FONT = "Times New Roman"

doc = Document()
st = doc.styles["Normal"]
st.font.name = FONT; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:ascii"), FONT); st.element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
st.paragraph_format.space_after = Pt(0); st.paragraph_format.line_spacing = 1.15
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)

def body(text=None, runs=None, after=10):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(after)
    if runs:
        for txt, ital in runs:
            r = p.add_run(txt); r.italic = ital
    else:
        p.add_run(text)

def line(text, after=2):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(after); p.add_run(text)

line("Dear Editors,", after=10)

body(runs=[
    ("We are pleased to submit our manuscript entitled “Building-scale morphological signatures of informal settlements” by Peng Luo, Fábio Duarte, Luiz Carlos Teixeira Coelho, and Carlo Ratti, for consideration as an Article in ", False),
    ("Nature Cities", True), (".", False),
])

body("More than one billion people live in informal settlements worldwide, yet defining and delimiting — quantitatively — which buildings belong within their boundaries remains an open challenge. Cities delimit informal settlements administratively, chiefly through land tenure; what the built fabric itself says about informality is far less understood. Existing approaches either compress an entire neighbourhood into a single index, or rely on opaque deep-learning features from satellite imagery that cannot say which aspect of a building marks it as informal. Because household surveys are scarce and costly in precisely the cities urbanising fastest, a fine-grained, interpretable characterisation recoverable from data that already exist everywhere — open building footprints — has been missing.")

body("Our manuscript addresses this gap using only open building footprints. For 188,735 buildings across ten favelas in Rio de Janeiro, we compute 25 morphometric indicators, train a per-site classifier to distinguish favela from formal buildings (mean accuracy 93.3%), and use explainable AI to recover, for every building, which footprint features drove its classification. We report three findings: (i) there is no single favela archetype — the dominant morphological feature differs from site to site; (ii) the morphological signal does not stop at the official favela boundary but grades across it in both directions — buildings on either side drift toward one another, with the dominant side varying by site — recasting a sharp legal line as a measurable morphological gradient; and (iii) a two-dimensional summary of each building’s signature reveals a single axis, consistent across all ten settlements, along which informal and formal buildings shift apart while still overlapping.")

body("Together, these results show that informal built form can be characterised building-by-building, and interpretably, from open data alone — offering a survey-free, transferable way to map morphological informality in any city where footprints exist, and a quantitative lens on how the balance between individual construction and collective regulation shapes urban form. They also carry a cautionary message: because the dominant features differ and even reverse between settlements, no single morphological detector transfers across places, and a settlement’s edge is better understood as a graded, two-way transition than a hard boundary.")

body(runs=[
    ("This work bridges urban morphology, informal-settlement studies, remote sensing, and explainable AI, and we believe it will be of strong interest to the interdisciplinary readership of ", False),
    ("Nature Cities", True), (".", False),
])

body("The manuscript is approximately 4,400 words long and contains five main figures.", after=12)

line("We suggest the following expert reviewers for this manuscript:", after=6)
for r in [
    "Ryan Wang, Northeastern University, USA, q.wang@northeastern.edu",
    "Monika Kuffer, University of Twente (ITC), Netherlands, m.kuffer@utwente.nl",
    "Luís M. A. Bettencourt, University of Chicago, USA, bettencourt@uchicago.edu",
    "Geoff Boeing, University of Southern California, USA, boeing@usc.edu",
    "Elsa Arcaute, University College London (CASA), UK, e.arcaute@ucl.ac.uk",
]:
    line(r, after=2)

line("", after=8)
line("The corresponding author for this manuscript is:", after=2)
line("Fábio Duarte")
line("Senseable City Laboratory, Massachusetts Institute of Technology")
line("fduarte@mit.edu", after=12)

body("We declare no competing interests. Thank you for considering our manuscript for publication. We look forward to your feedback.")

line("Sincerely,", after=2)
for a in ["Peng Luo", "Fábio Duarte", "Luiz Carlos Teixeira Coelho", "Carlo Ratti"]:
    line(a, after=2)

doc.save(OUT)
print("[saved]", OUT)
