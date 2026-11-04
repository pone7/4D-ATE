#!/usr/bin/env python3
"""Draw paper Figure 3 (three findings) from aggregate_results.py output.

Usage:
    python scripts/paper/fig_findings.py --agg paper_results.json --out findings.pdf
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--agg", required=True, help="JSON written by aggregate_results.py")
ap.add_argument("--out", default="findings.pdf")
args = ap.parse_args()
D = json.load(open(args.agg))
out = args.out
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'xtick.major.width':0.5,'ytick.major.width':0.5})
COL={'C':'#2a4a9e','R':'#3d7a2e','I':'#a3332b','G':'#7a3a8c'}
FAM={'tomopy_mlem':'C','sirt':'C','astra_cgls':'C','fista_tv':'R','isonet':'R','neat':'I','implicit_et':'I','r2gaussian':'G','x2gaussian':'G','fourdgs':'G'}
LAB={'tomopy_mlem':'MLEM','fourdgs':'4D-GS','isonet':'IsoNet','x2gaussian':'X²-G','r2gaussian':'R²-G','sirt':'SIRT','implicit_et':'Impl-ET','neat':'NeAT','fista_tv':'FISTA-TV','astra_cgls':'ASTRA-CGLS'}
g=lambda s,m,k:D[s][m][k]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.0),gridspec_kw={'width_ratios':[1.15,1.15,0.8],'wspace':0.38})
# Panel 1
ax=axs[0]; mono=['tomopy_mlem','fourdgs','isonet','x2gaussian','r2gaussian','sirt','implicit_et','neat','fista_tv']
off={'tomopy_mlem':(-5,-10),'fourdgs':(-6,4),'isonet':(-5,8),'sirt':(-5,-9),'x2gaussian':(6,3),'r2gaussian':(6,-5),'implicit_et':(-5,6),'neat':(-5,5),'fista_tv':(6,-5)}
for m in mono:
    x,y=g('mono',m,'vol_ssim')[0],g('mono',m,'recall')[0]*100; ye=g('mono',m,'recall')[1]*100
    ax.errorbar(x,y,yerr=ye,fmt='s' if FAM[m]=='G' else 'o',ms=4.5,color=COL[FAM[m]],mec='white',mew=0.5,elinewidth=0.6,capsize=1.5,zorder=4)
    dx,dy=off[m]; ax.annotate(LAB[m],(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=6,ha='left' if dx>0 else 'right',va='center')
ax.axvline(g('mono','tomopy_mlem','vol_ssim')[0],color='0.8',ls='--',lw=0.7,zorder=1); ax.axhline(g('mono','fista_tv','recall')[0]*100,color='0.8',ls='--',lw=0.7,zorder=1)
ax.text(0.03,0.42,'SSIM leader: MLEM\nRecall leader: FISTA-TV',transform=ax.transAxes,fontsize=6,va='bottom',bbox=dict(boxstyle='round,pad=0.25',fc='white',ec='0.8',lw=0.5))
ax.set_xlabel('Voxel SSIM ↑'); ax.set_ylabel('Atom recall (%) ↑'); ax.set_xlim(0.44,0.99); ax.set_ylim(-5,108)
ax.set_title('Finding 1: inversion on Mono',fontsize=7.5)
# Panel 2
ax=axs[1]; tiers=[('mono','Mono'),('bi','Bi'),('multislice','Multi'),('real','Real'),('brownian_pt','Brownian')]
MAIN={'mono':mono,'bi':['tomopy_mlem','x2gaussian','isonet','r2gaussian','implicit_et','sirt','neat','fista_tv'],'multislice':['fourdgs','x2gaussian','r2gaussian'],'real':['fourdgs','x2gaussian'],'brownian_pt':['astra_cgls','fista_tv','neat','tomopy_mlem','implicit_et']}
bs=[max(g(t,m,'vol_ssim')[0] for m in MAIN[t])*100 for t,_ in tiers]; br=[max(g(t,m,'recall')[0] for m in MAIN[t])*100 for t,_ in tiers]
xs=range(len(tiers))
ax.plot(xs,bs,'-o',color='#4a6fb5',ms=3.5,lw=1.2,label='Best SSIM (×100)'); ax.plot(xs,br,'-s',color='#c0392b',ms=3.5,lw=1.2,label='Best atom recall (%)')
for x,a,b in zip(xs,bs,br):
    ax.text(x,a-10 if x==0 else a+4,f'{a:.1f}',ha='center',fontsize=5.6,color='#4a6fb5'); ax.text(x,b+4 if (b>a or x>=2) else b-9,f'{b:.1f}',ha='center',fontsize=5.6,color='#c0392b')
ax.set_xticks(list(xs)); ax.set_xticklabels([l for _,l in tiers]); ax.set_ylim(-12,112); ax.set_ylabel('Score on common 0–100 scale')
ax.legend(fontsize=5.8,frameon=False,loc='center right'); ax.set_title('Finding 2: realism gap',fontsize=7.5)
# Panel 3
ax=axs[2]; ms=['x2gaussian','fourdgs']; w=0.36
for i,m in enumerate(ms):
    s=g('4d',m,'vol_ssim'); p=g('4d',m,'atom_identity_persistence')
    ax.bar(i-w/2,s[0]*100,w,yerr=s[1]*100,color='#8e7cc3',capsize=1.5,error_kw={'lw':0.6},label='Frame SSIM (×100)' if i==0 else None)
    ax.bar(i+w/2,p[0]*100,w,yerr=p[1]*100,color='#c0392b',capsize=1.5,error_kw={'lw':0.6},label='Atom ID persistence (%)' if i==0 else None)
    ax.text(i-w/2,s[0]*100+3,f'{s[0]*100:.1f}',ha='center',fontsize=5.6); ax.text(i+w/2,p[0]*100+(p[1]*100)+2,f'{p[0]*100:.1f}',ha='center',fontsize=5.6)
ax.set_xticks([0,1]); ax.set_xticklabels([LAB[m] for m in ms]); ax.set_ylim(0,138); ax.set_yticks([0,20,40,60,80,100]); ax.set_ylabel('Score (0–100)')
ax.legend(fontsize=5.6,frameon=False,loc='upper center',ncol=1); ax.set_title('Finding 3: 4D identity',fontsize=7.5)
fig.savefig(out,bbox_inches='tight',pad_inches=0.03); fig.savefig(out.replace('.pdf','.png'),dpi=200,bbox_inches='tight',pad_inches=0.03)
