"""Step 9: SI Fig. S21, balanced accuracy of the held-out favela component.

Manuscript items
* SI Figure S21: balanced accuracy per site
  for four designs. Diamonds, random 80/20 split refitted in this pipeline (one
  stratified random split per site, ``appendix_S12_spatial_cv.component_figure_reference``);
  open squares, 250 m blocks with a 200 m buffer (five folds, three grid placements);
  circles, held-out favela component without (blue) and with (orange) the 200 m buffer.
  Per-site markers = mean over scored folds; bars = +/-1 s.d. over folds; dots =
  individual folds (``appendix_S12_spatial_cv.figure_S21.fold_dots``); grey column =
  mean of the per-site values over the scored sites with a site-bootstrap 95% interval.
* Caption numbers: groups per site (Catumbi 11, ..., 58 in total), the site not scored
  with the 200 m buffer, and the column values 0.904 (0.868-0.937), 0.641 (0.576-0.717)
  on the eight testable sites, 0.650 (0.577-0.726) and 0.628 (0.575-0.678)
  -> heldout_component_values.csv, caption_numbers.txt.

Scoring as in SI S12 'Scoring' (the second criterion per site for the component
design and per site x grid placement for the blocks); the single random split per
site is not a cross-validation design and is not scored.

Inputs  outputs/appendix/S12_spatial_cv/folds_all.parquet (step 04); component_groups/component_log.csv (step 02)
Outputs outputs/appendix/S12_spatial_cv/figures/Figure_R1_heldout_component.pdf (+ _preview.png)
        outputs/appendix/S12_spatial_cv/component/heldout_component_values.csv, caption_numbers.txt
Run     python appendix/S12_spatial_cv/09_figure_S21.py
"""
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from favela_xai import spatial_cv as scv  # noqa: E402

import s12_common as C  # noqa: E402

cfg, s, P = C.setup(__file__)
S12 = C.section(cfg)
FIG = S12["figure_S21"]
OUT = P.out / "component"
OUT.mkdir(exist_ok=True)
PRIM_L, PRIM_B = int(cfg.get("spatial_cv.primary.block_m")), int(cfg.get("spatial_cv.primary.buffer_m"))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5, "axes.linewidth": 0.6,
                     "legend.frameon": False})
SITES = list(cfg.sites)
# two-line site labels
DISP = {site: cfg.site_name(site) for site in SITES}
DISP.update({"faz_quem_quer": "Faz Quem\nQuer", "fazena_coqueiro": "Fazenda\nCoqueiro",
             "grotao_de_costa_barros": "Grotão de\nCosta Barros", "morro_da_guaiba": "Morro da\nGuaíba"})

F = C.read_folds(cfg)
TOT = C.favela_totals(cfg, P)
comp = F[F.design == "component"]
testable = sorted(set(comp.site))
lc = {}
for b in (0, PRIM_B):
    kept, cov = scv.apply_rules(comp[comp.buffer == b], TOT, s, unit=["site"])
    lc[b] = kept
g, gcov = scv.apply_rules(F[(F.design == f"sq{PRIM_L}") & (F.buffer == PRIM_B) & ~F.control], TOT, s)
rnd = F[F.design == "random_80_20_stratified"]


def pooled(df):
    sm = df.groupby("site").balanced_accuracy.mean().reindex(testable).dropna()
    m, lo, hi, n = C.boot_mean(sm.to_numpy(), s)
    return m, lo, hi, n


Pv = {"random": pooled(rnd), "grid": pooled(g), "comp0": pooled(lc[0]), "comp200": pooled(lc[PRIM_B])}
lab = {"random": "Random 80/20 split, refitted", "grid": f"{PRIM_L} m blocks, {PRIM_B} m buffer, three grid placements",
       "comp0": "Held-out favela component, 0 m buffer", "comp200": f"Held-out favela component, {PRIM_B} m buffer"}
src = {"random": rnd, "grid": g, "comp0": lc[0], "comp200": lc[PRIM_B]}
rows = []
for k, (m, lo, hi, n) in Pv.items():
    rows.append(dict(design=lab[k], site="", sites_scored=n, folds_scored=int(src[k].site.isin(testable).sum()),
                     balanced_accuracy=m, ci_lo=lo, ci_hi=hi))
for site in SITES:
    for k in Pv:
        v = src[k][src[k].site == site].balanced_accuracy.dropna()
        status = ("not testable (single component group)" if k.startswith("comp") and site not in testable
                  else "not scored" if not len(v) else "scored")
        rows.append(dict(design=lab[k], site=site, status=status, folds_scored=len(v),
                         balanced_accuracy=v.mean() if len(v) else np.nan, sd_folds=v.std(ddof=1) if len(v) > 1 else np.nan))
R = pd.DataFrame(rows)
R.to_csv(OUT / "heldout_component_values.csv", index=False)

# caption numbers: groups per site and the values of the grey column
grp = comp[comp.buffer == 0].groupby("site").fold.nunique()
lines = ["groups per testable site: " + ", ".join(f"{cfg.site_name(k)} {int(v)}" for k, v in grp.items())
         + f" ({int(grp.sum())} in total; manuscript 58)",
         "single group (not testable): " + ", ".join(cfg.site_name(x) for x in SITES if x not in testable)]
nsc = [cfg.site_name(x) for x in testable if x not in set(lc[PRIM_B].site)]
lines.append(f"not scored with the {PRIM_B} m buffer: {', '.join(nsc) or 'none'}")
for k, (m, lo, hi, n) in Pv.items():
    lines.append(f"{lab[k]}: {m:.3f} ({lo:.3f}-{hi:.3f}), {n} sites")
lines.append("(manuscript: 0.904 (0.868-0.937); 0.641 (0.576-0.717); 0.650 (0.577-0.726); 0.628 (0.575-0.678))")
(OUT / "caption_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

# ---------------- figure
C_R, C_G, C_0, C_2 = "#555555", "#8a8a8a", "#1F6FB4", "#E2700E"
C_NA = "#909090"
H = 2.70
NA_Y = 0.363
fig, ax = plt.subplots(figsize=(183 / 25.4, H))
rng = np.random.default_rng(int(FIG["dot_seed"]))
JIT = float(FIG["dot_jitter"])


def dots(x, v, col):
    """Individual folds (SI Fig. S21 caption: 'dots, individual folds')."""
    if FIG["fold_dots"] and len(v):
        ax.scatter(x + rng.uniform(-JIT, JIT, len(v)), v, s=4, color=col, alpha=0.45, lw=0, zorder=2)


missing = []
for i, site in enumerate(SITES):
    r = rnd[rnd.site == site].balanced_accuracy
    if len(r):
        ax.plot(i - 0.27, r.iloc[0], marker="D", ms=5.2, color=C_R, ls="none", zorder=3)
    gv = g[g.site == site].balanced_accuracy.dropna()
    if len(gv):
        dots(i - 0.09, gv, C_G)
        ax.errorbar(i - 0.09, gv.mean(), yerr=gv.std(ddof=1), marker="s", ms=5.0, mfc="white", mec=C_G, color=C_G,
                    lw=1.0, capsize=0, ls="none", zorder=3)
    else:
        missing.append(i - 0.09)
    for off, buf, col in ((0.09, 0, C_0), (0.27, PRIM_B, C_2)):
        v = lc[buf][lc[buf].site == site].balanced_accuracy.dropna() if site in testable else pd.Series(dtype=float)
        if not len(v):
            missing.append(i + off)
            continue
        dots(i + off, v, col)
        ax.errorbar(i + off, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else None, marker="o", ms=5.4, color=col,
                    mec="white", mew=0.4, lw=1.0, capsize=0, ls="none", zorder=4)
for x in missing:                           # design not run or not scored at this site
    ax.plot([x - 0.058, x + 0.058], [NA_Y, NA_Y], color=C_NA, lw=1.4, solid_capstyle="butt", zorder=3)
i = len(SITES)
for off, key, col, mk, open_ in ((-0.27, "random", C_R, "D", False), (-0.09, "grid", C_G, "s", True),
                                 (0.09, "comp0", C_0, "o", False), (0.27, "comp200", C_2, "o", False)):
    m, lo, hi, _ = Pv[key]
    ax.errorbar(i + off, m, yerr=[[m - lo], [hi - m]], marker=mk, ms=5.8, color=col, mfc="white" if open_ else col,
                mec=col, lw=1.2, capsize=2.0, ls="none", zorder=4)
ax.axvspan(i - 0.5, i + 0.5, color="0.94", zorder=0)
ax.axhline(0.5, color="0.6", lw=0.5, ls=":", zorder=0)
ax.set_ylim(0.35, 1.06)
ax.set_xlim(-0.55, len(SITES) + 0.55)
ax.set_ylabel("Balanced accuracy")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(width=0.6, length=2.5)
ax.set_xticks(np.arange(len(SITES) + 1))
ax.set_xticklabels([DISP[x] for x in SITES] + ["Mean of sites\n(95% CI)"], fontsize=6.5)
h = [Line2D([], [], marker="D", color=C_R, ls="none", ms=4.6, label="Random 80/20 split, refitted"),
     Line2D([], [], marker="s", color=C_G, mfc="white", ls="none", ms=4.6, label=f"{PRIM_L} m spatial blocks, {PRIM_B} m buffer"),
     Line2D([], [], marker="o", color=C_0, ls="none", ms=4.8, label="Held-out favela component, 0 m buffer"),
     Line2D([], [], marker="o", color=C_2, ls="none", ms=4.8, label=f"Held-out favela component, {PRIM_B} m buffer")]
fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.5, handletextpad=0.3,
           columnspacing=0.8)
fig.legend(handles=[Line2D([], [], color=C_NA, lw=1.4, label="Design not run or not scored at this site")],
           loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0 - 0.145 / H), fontsize=6.5, handletextpad=0.3,
           handlelength=1.6)
FWI = fig.get_figwidth()
ax.set_position([0.4864 / FWI, 0.4938 / H, 6.6011 / FWI, 1.8062 / H])
path = P.figures / "Figure_R1_heldout_component.pdf"
fig.savefig(path, dpi=600)
fig.savefig(path.with_name(path.stem + "_preview.png"), dpi=int(2000 / fig.get_figwidth()))
print("wrote", path)
