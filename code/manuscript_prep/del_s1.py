import re, io

BASE = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project/Publications/2026_Favela_XAI_NC/2026_Favela_NatureCities/manuscript_Favela_XAI_morphology"
MAIN = BASE + "/01_main_naturecities.tex"
SUPP = BASE + "/03_supplement_naturecities.tex"

def braces_ok(t):
    s = re.sub(r'\\[{}]', '', t)
    return s.count('{') == s.count('}')

# ---------- 1) SUPPLEMENT: delete the S1 figure block ----------
supp = open(SUPP).read()
nfig_before = len(re.findall(r'\\label\{fig:', supp))
FIGBLOCK = (
"\\begin{figure}[H]\n"
"  \\centering\n"
"  \\includegraphics[width=\\textwidth]{Figures/Figure_X_data_table.png}\n"
"  \\caption{Illustration of the 25 per-building morphological\n"
"    indicators (left) and their computation pipeline (right).}\n"
"  \\label{fig:feature_table}\n"
"\\end{figure}\n\n"
)
assert supp.count(FIGBLOCK) == 1, "FIGBLOCK not found exactly once!"
supp = supp.replace(FIGBLOCK, "")
nfig_after = len(re.findall(r'\\label\{fig:', supp))
print(f"[supp] figures {nfig_before} -> {nfig_after}  (deleted feature_table)")
assert nfig_after == nfig_before - 1

# ---------- 2) MAIN: remove the L809 'and visualised in Figure S1' clause ----------
main = open(MAIN).read()
CLAUSE_OLD = "Supplementary Section~S3 and visualised\nin Supplementary Figure~S1. All indicators"
CLAUSE_NEW = "Supplementary Section~S3. All indicators"
assert main.count(CLAUSE_OLD) == 1, "L809 clause not found exactly once!"
main = main.replace(CLAUSE_OLD, CLAUSE_NEW)
print("[main] removed 'and visualised in Supplementary Figure~S1' clause")

# ---------- 3) MAIN: decrement every remaining 'Figure(s)~SN (and SM)' by 1 ----------
# After clause removal, the lowest remaining figure ref is S2, so all -> N-1 (>=1).
changes = []
def dec(m):
    head, n1, tail, n2 = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    out = f"{head}S{n1-1}"
    if tail:
        out += f" and S{int(n2)-1}"
    changes.append((m.group(0), out))
    return out
main = re.sub(r'(Figures?~)S(\d+)( and S(\d+))?', dec, main)
for a, b in changes:
    print(f"    {a!r:40s} -> {b!r}")

# sanity: no S0, no remaining S1+ figure-ref pointing to deleted (min should be S1 now)
assert not re.search(r'Figures?~S0\b', main), "produced Figure~S0!"
print(f"[main] decremented {len(changes)} figure-reference site(s)")

# ---------- 4) brace check + write ----------
print("[braces] supp:", braces_ok(supp), " main:", braces_ok(main))
assert braces_ok(supp) and braces_ok(main)
open(SUPP, "w").write(supp)
open(MAIN, "w").write(main)
print("[written] both files saved")
