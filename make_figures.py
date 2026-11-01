"""Rebuild four descriptive, fully vector PDF figures and README PNG previews."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datasets import ROOT, load_dataset
from data_simulator import generate_synthetic_stock_data

COLORS=['#176B87','#E88C30','#8058A5','#289D8F','#C45156','#758A38']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#A3AEB8','axes.labelcolor':'#243949','text.color':'#243949',
                     'xtick.color':'#455968','ytick.color':'#455968','pdf.fonttype':42,
                     'ps.fonttype':42,'savefig.facecolor':'white'})


def finish(fig,out,name,title,subtitle):
    fig.suptitle(title,x=0.07,y=0.985,ha='left',fontsize=15,fontweight='bold')
    fig.text(0.07,0.915,subtitle,ha='left',fontsize=9,color='#536674')
    fig.subplots_adjust(top=0.79,bottom=0.18,left=0.08,right=0.98,wspace=0.30,
                        hspace=0.68 if len(fig.axes)>=4 else 0.25)
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{name}.pdf')
    fig.savefig(out/f'{name}.png',dpi=180)
    plt.close(fig)


def make_simulated(out):
    x,Y,_,_,_,_=generate_synthetic_stock_data(D=20,T=100,random_seed=0)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for j in range(len(Y)):
        axes[0].plot(x,Y[j],color=COLORS[j%6],lw=0.9,alpha=0.7)
        axes[1].plot(x,100*(Y[j]/Y[j,0]-1),color=COLORS[j%6],lw=0.9,alpha=0.7)
    for ax in axes:
        ax.set_xlabel('Observation index'); ax.grid(alpha=0.15)
    axes[0].set(title='Raw simulated paths',ylabel='Price (arbitrary units)')
    axes[1].set(title='Change relative to initial price',ylabel='Change (%)')
    fig.text(.08,.035,'Display: J = 20, T = 100, seed = 0. Fits use raw prices; simulated regime changes are stream-specific.',fontsize=8)
    finish(fig,out,'simulated','Simulated multivariate price paths',
           'Correlated market, sector and idiosyncratic variation with drift changes')


def make_era5(data,out):
    coords=np.asarray(data.metadata['coordinates']); lats=np.unique(coords[:,0]); lons=np.unique(coords[:,1])
    mean=data.Y.mean(axis=1); grid=np.full((len(lats),len(lons)),np.nan)
    grid[np.searchsorted(lats,coords[:,0]),np.searchsorted(lons,coords[:,1])]=mean
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    levels=np.linspace(np.floor(mean.min()/5)*5,np.ceil(mean.max()/5)*5,16)
    field=axes[0].contourf(lons,lats,grid,levels=levels,cmap='RdYlBu_r')
    axes[0].set(xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)',title='Mean t2m, 2000-2025')
    cb=fig.colorbar(field,ax=axes[0],pad=.025,shrink=.90); cb.set_label('Temperature (°C)')
    anomaly=data.Y-data.Y[:,[0]]
    q=np.quantile(anomaly,[.05,.5,.95],axis=0)
    axes[1].fill_between(data.x,q[0],q[2],color='#D5E8EE',label='5th-95th spatial percentiles')
    axes[1].plot(data.x,q[1],color=COLORS[0],lw=1.8,label='Median over grid locations')
    axes[1].axhline(0,color='#A3AEB8',lw=.7)
    axes[1].set(xlabel='Year',ylabel='Change from 2000 (°C)',title='Spatial distribution of annual changes')
    axes[1].legend(frameon=False,fontsize=7,loc='upper left'); axes[1].grid(alpha=.15)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
    fig.text(.08,.035,'40,401 grid cells × 26 years; 0.25° grid, land and ocean. Annual means derive from daily 17:00 UTC samples.',fontsize=8)
    finish(fig,out,'era5','ERA5 regional 2 m air temperature',
           'Every latitude-longitude location is a dimension; annual observations from 2000 through 2025')


def make_holoassist(data,out):
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,hand in zip(axes,['Left_hand','Right_hand']):
        for component,color in zip('xyz',COLORS):
            name=f'{hand}:Wrist_{component}_m'
            j=data.names.index(name)
            ax.plot(data.x,data.Y[j],color=color,lw=1.3,label=component.upper())
        ax.set(title=hand.replace('_',' '),xlabel='clip_time_s (s)',ylabel='Exported wrist position (m)')
        ax.grid(alpha=.15); ax.legend(frameon=False,ncol=3,fontsize=8)
    fig.text(.08,.035,'Display: 6 wrist-position channels. Fitting retains all 1,092 columns and 240 readings, including 8 repeated times.',fontsize=8)
    finish(fig,out,'holoassist','HoloAssist two-hand motion sample',
           'R0027-12-GoPro excerpt; original exported coordinates, without camera correction or smoothing')


def make_cmapss(data,out):
    fig,axes=plt.subplots(2,2,figsize=(9,5.5))
    for ax,symbol,color in zip(axes.flat,['T24','T50','Ps30','phi'],COLORS):
        j=data.metadata['symbols'].index(symbol)
        ax.plot(data.x,data.Y[j],color=color,lw=.85)
        ax.set(title=f'{symbol}: {data.metadata["sensor_descriptions"][j]}',xlabel='Operating cycle',ylabel=data.metadata['units'][j])
        ax.grid(alpha=.15)
    fig.text(.08,.025,'Display: 4 of 21 sensors; 192 cycles to simulated failure. Settings and RUL labels are excluded from the fitted responses.',fontsize=8)
    finish(fig,out,'cmapss','NASA C-MAPSS turbofan degradation',
           'FD001 training engine 1; x = initial RUL + 1 - current RUL = operating cycle')


def make_all_figures(data_dir=ROOT/'data/raw',out=ROOT/'figures',offline=False):
    out=Path(out); make_simulated(out)
    for name,fn in [('era5',make_era5),('holoassist',make_holoassist),('cmapss',make_cmapss)]:
        fn(load_dataset(name,data_dir,offline),out)
    print(f'Four vector PDFs and PNG previews saved in {out}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/raw')
    p.add_argument('--output-dir',type=Path,default=ROOT/'figures')
    p.add_argument('--offline',action='store_true')
    args=p.parse_args(); make_all_figures(args.data_dir,args.output_dir,args.offline)
