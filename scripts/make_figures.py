"""Standalone publication figures using audited observations and embedded Times fonts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
P=Path(__file__).resolve().parents[1];OUT=P/'figures';A=json.loads((P/'results/recomputed/manuscript_audit_20261107.json').read_text())
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':10,'axes.labelsize':10,'xtick.labelsize':9.5,'ytick.labelsize':10,'legend.fontsize':9.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'axes.edgecolor':'#68717A','text.color':'#20262C','axes.labelcolor':'#20262C','xtick.color':'#46505A','ytick.color':'#46505A','xtick.major.width':.6,'ytick.major.width':.6,'savefig.pad_inches':0})
C=['#284E70','#438B83','#B56D50']; markers=['o','s','^']
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=240);plt.close(fig)
def grid(ax):
 ax.set_axisbelow(True);ax.grid(axis='x',color='#DFE4E8',lw=.55,ls=(0,(2,3)));ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=8)
# Each figure stands on its own; captions belong below in LaTeX.
fig,ax=plt.subplots(figsize=(5.5,2.35));fig.subplots_adjust(left=.14,right=.97,top=.82,bottom=.23);grid(ax)
for j,y in enumerate(['2024','2025','2026']):
 d=A[y];v=np.array(d['bucket_means']);ci=np.array(d['bucket_ci95']);pos=np.arange(3)+(j-1)*.19
 ax.errorbar(v,pos,xerr=np.array([v-ci[:,0],ci[:,1]-v]),fmt=markers[j],ms=4.5,elinewidth=1.15,capsize=2.2,capthick=.9,color=C[j],label=y,zorder=3)
ax.axvline(0,color='#89939B',lw=.7);ax.set_yticks(range(3),['Early','Mid','Late']);ax.invert_yaxis();ax.set_xlim(-.155,.155);ax.xaxis.set_major_locator(MultipleLocator(.05));ax.set_xlabel('Within-paper rating contrast (raw points)',labelpad=6)
fig.legend(ncol=3,frameon=False,loc='upper right',bbox_to_anchor=(.985,.995),handlelength=1.35,columnspacing=1.5,borderaxespad=0)
save(fig,'timing_association')
fig,ax=plt.subplots(figsize=(5.5,2.55));fig.subplots_adjust(left=.24,right=.97,top=.82,bottom=.22);grid(ax)
methods=['Joint','Confidence','OTCA','Heuristic']
for j,y in enumerate(['2024','2025','2026']):
 vals=[A[y]['gain_pp'][m] for m in methods];pos=np.arange(4)+(j-1)*.18
 ax.scatter(vals,pos,marker=markers[j],s=25,c=C[j],label=y,zorder=4)
ax.set_yticks(range(4),['Joint','Confidence','OTCA','Heuristic timing']);ax.invert_yaxis();ax.set_xlim(0,12);ax.xaxis.set_major_locator(MultipleLocator(2));ax.set_xlabel('Gain over uniform aggregation (percentage points)',labelpad=6)
fig.legend(ncol=3,frameon=False,loc='upper right',bbox_to_anchor=(.985,.995),columnspacing=1.45,handletextpad=.35,borderaxespad=0)
save(fig,'aggregation_gain')
# A compact method illustration with direct labels rather than a detached legend.
fig,ax=plt.subplots(figsize=(3.9,1.65));fig.subplots_adjust(left=.16,right=.97,top=.96,bottom=.32)
x=np.linspace(0,1,700);d=A['2026'];a=d['slope'];q1,q2=d['q33'],d['q66'];h=np.array(d['heuristic_weights'])
ax.plot(x,1+a*x,color=C[0],lw=1.6)
hx=[0,q1,q1,q2,q2,1];hy=[h[2],h[2],h[1],h[1],h[0],h[0]]
ax.plot(hx,hy,color=C[1],lw=1.3,ls=(0,(3.5,2)))
ax.text(.52,2.04,r'OTCA ($a=1.204$)',fontsize=9.5,color=C[0],ha='center')
ax.text(.53,1.13,'Heuristic',fontsize=9.5,color=C[1],ha='center')
ax.set_xlim(-.025,1.025);ax.set_ylim(.78,2.32);ax.set_xlabel('Relative position (0 = latest, 1 = earliest)',fontsize=9.5,labelpad=5)
ax.set_ylabel('Weight',fontsize=9.5,labelpad=5);ax.set_xticks([0,.5,1],['0','0.5','1']);ax.set_yticks([1,1.5,2],['1','1.5','2'])
ax.set_axisbelow(True);ax.grid(axis='y',color='#E8ECEF',lw=.5);ax.tick_params(labelsize=9)
save(fig,'weight_design')
# One heterogeneity plot, with uncertainty from paper-clustered standard errors.
d=A['2026']['heterogeneity'];gs=d['groups'];v=np.array([g['beta'] for g in gs]);se=np.array([g['se_cluster'] for g in gs])
fig,ax=plt.subplots(figsize=(5.5,2.25));fig.subplots_adjust(left=.15,right=.97,top=.97,bottom=.30)
ax.axhline(0,color='#929CA5',lw=.7)
for j in range(4):
 ax.errorbar(j,v[j],yerr=1.96*se[j],fmt='o',ms=5,color=C[0],mfc=C[0] if gs[j]['p_cluster']<.05 else 'white',capsize=3,lw=1.2)
ax.set_xticks(range(4),[f'Q{i+1}\n{100*g["accept"]:.1f}% accepted' for i,g in enumerate(gs)]);ax.set_ylabel('Late rating contrast');ax.set_xlim(-.5,3.5);ax.set_ylim(-.06,.225);ax.yaxis.set_major_locator(MultipleLocator(.05));ax.set_axisbelow(True);ax.grid(axis='y',color='#E5E8EC',lw=.55)
save(fig,'heterogeneity')
# Review-level decision loss is separate from paper-level aggregation loss.
fig,ax=plt.subplots(figsize=(5.5,1.85));fig.subplots_adjust(left=.14,right=.82,top=.77,bottom=.29);grid(ax)
for j,y in enumerate(['2024','2025','2026']):
 early,_,late=A[y]['bucket_brier']
 ax.plot([early,late],[j,j],color='#A8B3BD',lw=2,zorder=2)
 ax.scatter(early,j,s=30,color=C[0],marker='o',zorder=3,label='Early' if j==0 else None)
 ax.scatter(late,j,s=32,color=C[2],marker='s',zorder=3,label='Late' if j==0 else None)
 ax.text(1.025,j,f'+{late-early:.4f}',transform=ax.get_yaxis_transform(),va='center',fontsize=9.5,color='#46505A')
ax.set_yticks(range(3),['2024','2025','2026']);ax.set_ylim(2.5,-.5);ax.set_xlim(.173,.216);ax.xaxis.set_major_locator(MultipleLocator(.01));ax.set_xlabel('Review-level score-derived Brier value',labelpad=5)
ax.text(1.025,1.02,'Gap',transform=ax.transAxes,fontsize=9.5,color='#46505A')
fig.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.48,1.01),handletextpad=.3,columnspacing=1.4)
save(fig,'review_brier')
print('Wrote five independent vector figures with embedded Times New Roman fonts.')
