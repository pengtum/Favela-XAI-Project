import re
MAIN = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project/Publications/2026_Favela_XAI_NC/2026_Favela_NatureCities/manuscript_Favela_XAI_morphology/01_main_naturecities.tex"
t = open(MAIN).read()

edits = [
 # A2 abstract: reconcile finding-1 vs finding-3 (self-contradiction)
 ("Third, a two-dimensional summary of each",
  "Third, although the dominant feature differs by site, a two-dimensional summary of each"),
 # I3 intro: first gap = resolution/interpretability, not coverage ("uneven across cities")
 ("Such a\ncharacterisation remains uneven across cities.",
  "Such a\ncharacterisation---at the scale of the individual building, and\ninterpretable enough to say \\emph{which} features distinguish informal\nform---remains scarce."),
 # I4 intro: "necessary" overclaims logical necessity
 ("make this analytic\ngranularity necessary.",
  "make this analytic\ngranularity especially informative."),
 # I5 intro: fold redundant slum-dynamics topic stub into the Mumbai example
 ("These approaches have also been extended to map slum dynamics\nover time. In Mumbai, for example, morphological slums proceed",
  "These approaches also capture slum dynamics over time: in Mumbai,\nfor example, morphological slums proceed"),
 # I6 intro: frame the gap as novelty, not mere difficulty
 ("What remains analytically more demanding is the recovery of that",
  "What existing work has not delivered is the recovery of that"),
 # I7 intro: de-jargon "opaque embeddings inside deep convolutional pipelines"
 ("or to opaque\nembeddings inside deep convolutional pipelines on satellite\nimagery;",
  "or to black-box\nfeatures learned inside deep-learning models on satellite\nimagery;"),
]

for old, new in edits:
    c = t.count(old)
    assert c == 1, f"MATCH={c} for: {old[:55]!r}"
    t = t.replace(old, new)

s = re.sub(r'\\[{}]', '', t)
assert s.count('{') == s.count('}'), "brace imbalance!"
open(MAIN, "w").write(t)
print(f"applied {len(edits)} safe edits; braces OK")
