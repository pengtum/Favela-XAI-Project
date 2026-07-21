import re
MAIN = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project/Publications/2026_Favela_XAI_NC/2026_Favela_NatureCities/manuscript_Favela_XAI_morphology/01_main_naturecities.tex"
t = open(MAIN).read()

edits = [
 # ITEM 3 — abstract opening question: de-grandiose
 ("Here we ask\nwhat informal-settlement morphology actually is, building by building,\nusing only open building footprints as input.",
  "Here we ask\nwhich footprint features distinguish informal from formal built form,\nbuilding by building, using only open building footprints as input."),

 # ITEM 2 — abstract closing: split + close the significance loop (any city)
 ("Together\nthese results recast informal-settlement morphology as at once\n\\emph{site-specific} and \\emph{spatially continuous}---turning the favela\nboundary from a sharp legal line into a quantifiable, feature-resolved\nmorphological gradient, and showing that informal built form can be\ncharacterised building-by-building, and interpretably, from open\nfootprints alone, though no single morphological template transfers across\nsettlements.}",
  "Together\nthese results recast informal-settlement morphology as at once\n\\emph{site-specific} and \\emph{spatially continuous}, turning the favela\nboundary from a sharp legal line into a quantifiable, feature-resolved\nmorphological gradient. Because it needs only open building footprints,\nthe approach offers a survey-free, interpretable way to characterise\ninformal built form building-by-building in any city where such data\nexist---though no single morphological template transfers across\nsettlements.}"),

 # ITEM 1 — intro pivot: lead with the discovery, not the technique/site
 ("Here we apply explainable AI (XAI) to open-access building\nfootprints to reveal such building-scale morphological signals\nacross ten Rio de Janeiro favelas (Figure~\\ref{fig:concept}).",
  "Here we show that informal built form can be characterised building by\nbuilding, and interpretably, using only open footprints---and that two\ncommon assumptions do not hold: that informal settlements share a single\nmorphological signature, and that a settlement's boundary is a sharp\nline. We demonstrate this across ten Rio de Janeiro favelas by applying\nexplainable AI (XAI) to open-access building footprints\n(Figure~\\ref{fig:concept})."),

 # ITEM 4 — intro paragraph 1: forward-point the lede so the discovery surfaces early
 ("interpretable enough to say \\emph{which} features distinguish informal\nform---remains scarce.",
  "interpretable enough to say \\emph{which} features distinguish informal\nform---remains scarce. We find that this built form is not one thing but\nmany---no single favela archetype holds even within a single city---and\nthat its morphological signature does not stop at the legal boundary but\ngrades continuously across it."),
]

for old, new in edits:
    c = t.count(old)
    assert c == 1, f"MATCH={c} for: {old[:60]!r}"
    t = t.replace(old, new)

s = re.sub(r'\\[{}]', '', t)
assert s.count('{') == s.count('}'), "brace imbalance!"
open(MAIN, "w").write(t)

# report abstract word count
a = t[t.find('begin{abstract}')+15:t.find('end{abstract}')]
a = re.sub(r'\\[a-zA-Z]+\{?|[{}\\~$]', ' ', a)
print(f"applied {len(edits)} framing edits; braces OK; abstract now ~{len(a.split())} words")
