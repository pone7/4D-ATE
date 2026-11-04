#!/usr/bin/env python3
"""Write the paper's LaTeX table bodies from aggregate_results.py output.

Produces main_rows.tex (Table 4 body), full_<tier>.tex (appendix leaderboards,
mean +/- 95% CI) and ms_rows.tex (per-sample Multislice table).

Usage:
    python scripts/paper/make_latex_tables.py --agg paper_results.json --outdir tables/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--agg", required=True, help="JSON written by aggregate_results.py")
ap.add_argument("--outdir", default="tables")
args = ap.parse_args()
D = json.load(open(args.agg))
outdir = args.outdir
Path(outdir).mkdir(parents=True, exist_ok=True)
L={'tomopy_mlem':'MLEM','fourdgs':'4D-GS','isonet':'IsoNet','x2gaussian':'X$^2$-G','r2gaussian':'R$^2$-G','sirt':'SIRT',
'implicit_et':'Impl-ET','neat':'NeAT','fista_tv':'FISTA-TV','astra_cgls':'ASTRA-CGLS','astra_sirt':'ASTRA-SIRT','astra_sart':'ASTRA-SART',
'astra_ossart':'ASTRA-OS-SART','dart':'DART','genfire':'GENFIRE','fbp':'FBP','wbp':'WBP','tomopy_gridrec':'TomoPy-Gridrec',
'dif_gaussian':'DIF-Gaussian','instamap':'InstaMap','sax_nerf':'SAX-NeRF'}
F={'tomopy_mlem':'C','sirt':'C','astra_cgls':'C','astra_sirt':'C','astra_sart':'C','astra_ossart':'C','fbp':'C','wbp':'C','tomopy_gridrec':'C','genfire':'C',
'fista_tv':'R','dart':'R','isonet':'R','neat':'I','implicit_et':'I','sax_nerf':'I','instamap':'I','r2gaussian':'G','x2gaussian':'G','fourdgs':'G','dif_gaussian':'G'}
FAM={'C':'\\famC','R':'\\famR','I':'\\famI','G':'\\famG'}
def g(sub,m,k): x=D[sub][m].get(k); return x
def v(x,d=3): return '\\NA' if x is None else f"{x[0]:.{d}f}"
def vc(x,d=3): return '\\NA' if x is None else f"{x[0]:.{d}f}{{\\tiny$\\pm${x[1]:.{d}f}}}"
def best(sub,ms,k,hi=True):
    vals=[(g(sub,m,k)[0],m) for m in ms if g(sub,m,k)]
    return (max if hi else min)(vals)[1] if vals else None
def cell(sub,m,k,b,d=3,ci=False):
    s=(vc if ci else v)(g(sub,m,k),d)
    if b is not None and s!='\\NA' and v(g(sub,m,k),d)==v(g(sub,b,k),d):
        if ci: s=f"\\textbf{{{v(g(sub,m,k),d)}}}{{\\tiny$\\pm${g(sub,m,k)[1]:.{d}f}}}"
        else: s=f"\\textbf{{{s}}}"
    return s
# ---- main table rows
main={'mono':['tomopy_mlem','fourdgs','isonet','x2gaussian','r2gaussian','sirt','implicit_et','neat','fista_tv'],
'bi':['tomopy_mlem','x2gaussian','isonet','r2gaussian','implicit_et','sirt','neat','fista_tv'],
'4d':['x2gaussian','fourdgs'],'multislice':['fourdgs','x2gaussian','r2gaussian'],'real':['fourdgs','x2gaussian'],
'brownian_pt':['astra_cgls','fista_tv','neat','tomopy_mlem','implicit_et']}
# Matched species accuracy (Bi) and Brownian-Pt identity persistence are not
# stored in results.jsonl; they are copied from the paper's separate analyses.
SP={'tomopy_mlem':'0.932','isonet':'\\textbf{0.944}','x2gaussian':'0.904','implicit_et':'0.867','sirt':'0.921','r2gaussian':'0.852','neat':'0.831','fista_tv':'0.802'}
BPERS={'astra_cgls':'\\textbf{0.128}','fista_tv':'0.105','neat':'0.042','tomopy_mlem':'0.018','implicit_et':'0.000'}
NAME={'mono':'Mono','bi':'Bi','4d':'4D','multislice':'Multislice','real':'Real','brownian_pt':'Brownian-Pt'}
lines=[]
for sub,ms in main.items():
    rk='trajectory_rmsd' if sub=='4d' else 'rmsd'
    bs,br,bm=best(sub,ms,'vol_ssim'),best(sub,ms,'recall'),best(sub,ms,rk,False)
    bp=best(sub,ms,'atom_identity_persistence') if sub=='4d' else None
    for i,m in enumerate(ms):
        head=f"\\multirow{{{len(ms)}}}{{*}}{{{NAME[sub]}}}" if i==0 else ""
        sp=SP.get(m,'\\NA') if sub=='bi' else '\\NA'
        pers=cell(sub,m,'atom_identity_persistence',bp) if sub=='4d' else (BPERS.get(m,'\\NA') if sub=='brownian_pt' else '\\NA')
        lines.append(f"{head}\n& {L[m]} & {FAM[F[m]]} & {cell(sub,m,'vol_ssim',bs)} & {cell(sub,m,'recall',br)} & {cell(sub,m,rk,bm,2 if sub=='4d' else 3)} & {sp} & {pers} \\\\")
    lines.append("\\midrule")
open(outdir+'/main_rows.tex','w').write("\n".join(lines[:-1])+"\n")
# ---- appendix full tables (all methods, mean ± CI)
def full(sub,label,extra=False):
    ms=sorted(D[sub],key=lambda m:-(g(sub,m,'vol_ssim') or [0])[0])
    bs,br,bm=best(sub,ms,'vol_ssim'),best(sub,ms,'recall'),best(sub,ms,'rmsd',False)
    out=[]
    for m in ms:
        n=D[sub][m]['n']; na=(g(sub,m,'recall') or [0,0,0])[2]
        nstr=f"{n}" if na in (n,0) else f"{n} / {na}"
        row=f"{L[m]} & {FAM[F[m]]} & {nstr} & {cell(sub,m,'vol_ssim',bs,ci=True)} & {cell(sub,m,'recall',br,ci=True)} & {cell(sub,m,'precision',None,ci=True)} & {cell(sub,m,'rmsd',bm,ci=True)}"
        if extra: row+=f" & {vc(g(sub,m,'trajectory_rmsd'),2)} & {vc(g(sub,m,'atom_identity_persistence'))}"
        out.append(row+" \\\\")
    open(f"{outdir}/full_{label}.tex",'w').write("\n".join(out)+"\n")
for sub in ['mono','bi','real','brownian_pt']: full(sub,sub)
full('4d','4d',True)
# multislice per sample
ps=D['multislice_per_sample']; out=[]
for s in sorted(ps):
    for i,m in enumerate(['fourdgs','x2gaussian','r2gaussian']):
        x=ps[s].get(m)
        f=lambda y:'\\NA' if y is None or y!=y or y==float('inf') else f"{y:.3f}"
        out.append((f"\\multirow{{3}}{{*}}{{{s}}}" if i==0 else "")+f" & {L[m]} & {f(x[0])} & {f(x[1])} & {f(x[2])} \\\\")
    out.append("\\midrule")
open(outdir+'/ms_rows.tex','w').write("\n".join(out[:-1])+"\n")
print(f'wrote LaTeX tables to {outdir}/')
