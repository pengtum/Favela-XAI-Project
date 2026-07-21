import os, re, shutil

BASE = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project/Publications/2026_Favela_XAI_NC/2026_Favela_NatureCities/manuscript_Favela_XAI_morphology"
SUPP = os.path.join(BASE, "03_supplement_naturecities.tex")
BIB  = os.path.join(BASE, "02_references.bib")
APP  = os.path.join(BASE, "Appendix")
APPF = os.path.join(APP, "Figures")

supp = open(SUPP).read()

# collect dependencies referenced by the supplement
figs = sorted(set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}', supp)))
inputs = sorted(set(re.findall(r'\\input\{([^}]*)\}', supp)))
deps = figs + [i if i.endswith('.tex') else i + '.tex' for i in inputs]

os.makedirs(APPF, exist_ok=True)

# 1) copy + patch the supplement .tex (add a bibliography so it compiles standalone)
patched = supp
assert patched.rstrip().endswith("\\end{document}")
patched = patched.rstrip()[:-len("\\end{document}")] + (
    "\\bibliographystyle{elsarticle-num}\n"
    "\\bibliography{02_references}\n\n"
    "\\end{document}\n"
)
open(os.path.join(APP, "03_supplement_naturecities.tex"), "w").write(patched)

# 2) copy the bib
shutil.copy2(BIB, os.path.join(APP, "02_references.bib"))

# 3) copy every figure + \input table, preserving the Figures/ subpath
missing, copied = [], []
for d in deps:
    src = os.path.join(BASE, d)
    if not os.path.isfile(src):
        missing.append(d); continue
    dst = os.path.join(APP, d)              # d already starts with "Figures/"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst); copied.append(d)

print(f"Appendix folder: {APP}")
print(f"copied .tex (with bibliography added) + 02_references.bib")
print(f"copied {len(copied)} figure/table dependencies into Appendix/Figures/")
if missing:
    print("  MISSING (not copied):", missing)
else:
    print("  all dependencies present ✓")
# size
tot = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(APP) for f in fs)
print(f"total bundle size: {tot/1e6:.1f} MB")
