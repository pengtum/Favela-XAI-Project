import re
MAIN = "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/My Drive/Research/SCL_Project/Favela_Project/Publications/2026_Favela_XAI_NC/2026_Favela_NatureCities/manuscript_Favela_XAI_morphology/01_main_naturecities.tex"
lines = open(MAIN).read().split("\n")  # 0-indexed; file line N -> lines[N-1]

def region(a, b):  # inclusive file line numbers
    return "\n".join(lines[a-1:b])

def wc(text, drop_captions=True):
    t = text
    # strip inline/full-line comments (not \%)
    t = re.sub(r'(?<!\\)%.*', '', t)
    # drop float environments (captions/tables not counted as body text)
    if drop_captions:
        for env in ['figure','table','equation','align','tabular']:
            t = re.sub(r'\\begin\{'+env+r'\*?\}.*?\\end\{'+env+r'\*?\}', ' ', t, flags=re.DOTALL)
    # drop citations / refs / labels entirely
    t = re.sub(r'\\(citep|citet|cite|ref|autoref|eqref|label)\*?\{[^}]*\}', ' ', t)
    # drop inline math
    t = re.sub(r'\$[^$]*\$', ' ', t)
    # remove command names (keep braced text content), then braces
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t)
    t = t.replace('{',' ').replace('}',' ')
    # leftover escapes
    t = re.sub(r'\\[^a-zA-Z]', ' ', t)
    # count word tokens
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*", t)
    return len(words)

intro   = wc(region(126, 298))     # Introduction
results = wc(region(299, 590))     # Results
disc    = wc(region(591, 744))     # Discussion (before Declaration at 745)
methods = wc(region(750, 999))     # Methods (before \bibliographystyle at 1000)
abstract= wc(region(70, 94))

print(f"Abstract                     : {abstract:5d}")
print(f"Introduction                 : {intro:5d}")
print(f"Results                      : {results:5d}")
print(f"Discussion                   : {disc:5d}")
print(f"  -> Intro+Results+Discussion: {intro+results+disc:5d}   (main text, excl. Methods)")
print(f"Methods                      : {methods:5d}")
print(f"  -> incl. Methods           : {intro+results+disc+methods:5d}   (full body)")
print("(captions, refs, math, LaTeX commands excluded; ~approximate)")
