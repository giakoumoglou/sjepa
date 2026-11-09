"""Recreate the supplied two-panel S-JEPA ImageNet-vs-parameters figure.

Run: pip install matplotlib numpy && python recreate_sjepa_figure.py
The original source was unavailable: this is a visual reconstruction based on
its raster screenshot. Parameter counts are approximate backbone proxies.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

OUT = Path(__file__).resolve().parent
STEM = 'imagenet_1k'
BLUE = '#2968bd'
TEAL = '#419c9f'
BLACK = '#111111'

# (method, architecture, epochs, approximate parameters in M, top-1 accuracy)
LINEAR = [
 ('S-JEPA','ViT-B/16',600,87,73.5), ('S-JEPA','ViT-L/16',600,310,78.3),
 ('I-JEPA','ViT-B/16',600,86,72.9), ('I-JEPA','ViT-L/16',600,307,77.5),
 ('I-JEPA','ViT-H/14',300,632,79.3),
 ('MAE','ViT-B/16',1600,86,68.0), ('MAE','ViT-L/16',1600,307,76.0),
 ('MAE','ViT-H/14',1600,632,77.2),
 ('data2vec','ViT-L/16',1600,307,77.3),
 ('LeJEPA','ViT-L/16',100,307,77.1), ('LeJEPA','ViT-H/14',100,632,79.0),
 ('iBOT','ViT-B/16',400,86,79.5),
]
LOW_SHOT = [
 ('S-JEPA','ViT-B/16',600,87,70.1), ('S-JEPA','ViT-L/16',600,310,70.8),
 ('I-JEPA','ViT-B/16',600,86,68.9), ('I-JEPA','ViT-L/16',600,307,69.4),
 ('I-JEPA','ViT-H/14',300,632,73.3),
 ('MAE','ViT-L/16',1600,307,67.1), ('MAE','ViT-H/14',1600,632,71.5),
 ('data2vec','ViT-L/16',1600,307,73.3),
 ('iBOT','ViT-B/16',400,86,69.7),
]

plt.rcParams.update({
 'font.family':'serif', 'font.serif':['DejaVu Serif','STIXGeneral'],
 'mathtext.fontset':'dejavuserif', 'font.size':13,
 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
 'axes.linewidth':1.4,
})

# Labels given in SCREENSHOT PIXELS for a 2048 x 728 canvas, to reproduce its layout.
# Each entry: (method, arch, title_x, title_y, subtitle_x, subtitle_y, alignment)
LABELS_LINEAR = [
 ('iBOT','ViT-B/16',340,96,340,119,'center'),
 ('S-JEPA','ViT-B/16',281,308,280,332,'center'),
 ('I-JEPA','ViT-B/16',282,400,282,426,'center'),
 ('MAE','ViT-B/16',362,566,363,592,'left'),
 ('S-JEPA','ViT-L/16',616,122,613,147,'center'),
 ('I-JEPA','ViT-L/16',645,253,645,278,'left'),
 ('MAE','ViT-L/16',606,351,604,378,'right'),
 ('LeJEPA','ViT-L/16',648,335,648,363,'left'),
 ('data2vec','ViT-L/16',648,386,648,411,'left'),
 ('I-JEPA','ViT-H/14',834,98,835,123,'center'),
 ('LeJEPA','ViT-H/14',796,152,798,180,'left'),
 ('MAE','ViT-H/14',799,217,799,244,'left'),
]
LABELS_LOWSHOT = [
 ('S-JEPA','ViT-B/16',1283,275,1282,301,'center'),
 ('I-JEPA','ViT-B/16',1283,429,1282,457,'center'),
 ('iBOT','ViT-B/16',1259,370,1259,396,'right'),
 ('S-JEPA','ViT-L/16',1609,239,1609,263,'center'),
 ('I-JEPA','ViT-L/16',1633,391,1632,417,'left'),
 ('I-JEPA','ViT-H/14',1825,175,1825,201,'left'),
 ('data2vec','ViT-L/16',1609,99,1609,127,'center'),
 ('MAE','ViT-L/16',1610,529,1610,556,'center'),
 ('MAE','ViT-H/14',1825,238,1825,267,'left'),
]


# Reuse the coordinates from the supplied 2048 x 728 reference image.
# For individual panels, horizontally shift the right panel into a 1024 px canvas.
REF_HEIGHT = 728
REF_BOTH_WIDTH = 2048
REF_SINGLE_WIDTH = 1024
RIGHT_SHIFT = 1032

def setup(ax, which):
    ax.set_xscale('log')
    if which == 'linear':
        ax.set_xlim(30,1700)
        ax.set_ylim(66,82)
        ax.set_yticks(np.arange(66,83,2))
    else:
        ax.set_xlim(50,1500)
        ax.set_ylim(65,75)
        ax.set_yticks([66,68,70,72,74])
    ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4))
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10),numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='major', color='#bcbcbc', ls=':', lw=1.65, alpha=.88)
    ax.set_axisbelow(True)
    for edge in ('left','top','right'):
        ax.spines[edge].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.tick_params(axis='both',which='major',labelsize=16,width=1.6,length=9)
    ax.tick_params(axis='x',which='minor',length=7,width=1.3)
    ax.set_xlabel('Parameters (M)',fontsize=20,labelpad=6)
    ax.set_ylabel('Top-1 (%)',fontsize=20,labelpad=8)

def draw(ax, data):
    for method,color in [('S-JEPA',BLUE),('I-JEPA',TEAL)]:
        pts = sorted((r for r in data if r[0]==method),key=lambda r:r[3])
        ax.plot([p[3] for p in pts],[p[4] for p in pts],
                color=color,lw=3.3,ls=(0,(4.5,2.2,1.3,2.2)),zorder=3)
    for method,arch,epochs,x,y in data:
        color = BLUE if method=='S-JEPA' else TEAL if method=='I-JEPA' else BLACK
        marker = Line2D([x],[y],marker='o',markersize=15,linestyle='None',
                        markeredgewidth=3.5,markeredgecolor=color,
                        markerfacecolor=color if method in ('S-JEPA','I-JEPA') else 'white',
                        markerfacecoloralt='white',
                        fillstyle='top' if method in ('S-JEPA','I-JEPA') else 'full',zorder=6)
        ax.add_line(marker)

def draw_leaders(ax):
    # Thin leader lines for overlapping ViT-L/16 baseline points.
    for x,y,x2,y2 in [(307,76.0,285,73.0),(307,77.1,342,73.7),(307,77.3,355,72.0)]:
        ax.annotate('',xy=(x,y),xytext=(x2,y2),
                    arrowprops={'arrowstyle':'-','lw':1.0,'color':BLACK},zorder=2)

def make_figure(mode):
    if mode not in ('linear','lowshot','both'):
        raise ValueError(mode)
    both = mode == 'both'
    width = REF_BOTH_WIDTH if both else REF_SINGLE_WIDTH
    fig = plt.figure(figsize=(width/100, REF_HEIGHT/100), dpi=100, facecolor='white')

    def text_at(px,py,value,color,size,bold=True,ha='center'):
        fig.text(px/width,1-py/REF_HEIGHT,value,ha=ha,va='center',fontsize=size,
                 color=color,fontweight='bold' if bold else 'normal')

    def draw_labels(records,shift=0):
        for method,arch,x,y,x2,y2,ha in records:
            col=BLUE if method=='S-JEPA' else TEAL if method=='I-JEPA' else BLACK
            text_at(x-shift,y,method,col,15.2,True,ha)
            ep=next(p[2] for p in (LINEAR+LOW_SHOT) if p[0]==method and p[1]==arch)
            text_at(x2-shift,y2,f'{arch} ({ep}ep)',col,11.8,True,ha)

    if mode in ('linear','both'):
        ax = fig.add_axes([110/width,116/REF_HEIGHT,880/width,545/REF_HEIGHT])
        setup(ax,'linear')
        draw(ax,LINEAR)
        draw_leaders(ax)
        text_at(552,36,'ImageNet-1K Linear Evaluation vs Parameters',BLACK,19.1,False)
        draw_labels(LABELS_LINEAR)

    if mode in ('lowshot','both'):
        shift = 0 if both else RIGHT_SHIFT
        ax = fig.add_axes([(1142-shift)/width,116/REF_HEIGHT,882/width,545/REF_HEIGHT])
        setup(ax,'lowshot')
        draw(ax,LOW_SHOT)
        text_at(1585-shift,36,'ImageNet-1K 1% Low-Shot vs Parameters',BLACK,19.1,False)
        draw_labels(LABELS_LOWSHOT,shift=shift)

    name = {'linear': 'imagenet_1k_linear',
            'lowshot':'imagenet_1k_1percent_lowshot',
            'both':'imagenet_1k'}[mode]
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=160,facecolor='white')
    plt.close(fig)
    print('Saved:',name,'(PNG/PDF/SVG)')

if __name__=='__main__':
    for mode in ('linear','lowshot','both'):
        make_figure(mode)
