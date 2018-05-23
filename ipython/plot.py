import pickle
import numpy as np 
import matplotlib.pyplot as plt 

def smooth_plot(ax,x_raw,y_raw,smooth_width=1,alpha=0.4,**kwargs):
    nsamples = len(x_raw)
    x_avg,y_avg = [],[]
    for i in range(0,nsamples,smooth_width):
        x_avg.append(x_raw[i])
        y_avg.append(sum(y_raw[i:i+smooth_width])/smooth_width)
        
    p=ax.plot(x_avg, y_avg,**kwargs)
    if 'label' in kwargs:
        kwargs.pop('label')
    ax.plot(x_raw, y_raw, alpha=alpha,color=p[0].get_color(),**kwargs)
    return ax

def plot_curve(curve_list=[],labels=[],xlabel='',ylabel='',
				xlim=None,ylim=None,lw=4,linestyle='-',marker='',
				smooth_width=1):
    for (iters,values),label in zip(curve_list,labels):
        smooth_plot(plt,iters,values,
        			smooth_width=smooth_width,
        			alpha=0.3,
        			linestyle=linestyle,
        			marker = marker,
        			lw=lw,
        			label=label)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(fontsize=25)

def parse_loss_traj(data_file):
    with open(data_file,'rb') as f:
        loss_data = pickle.load(f)

    res = loss_data['train']
    res = sorted(res,key = lambda t:t['iter'])
    iter_s = [t['iter'] for t in res]
    trL_s = [t['info'][0] for t in res]
    trA_s = [t['info'][1] for t in res]

    res = loss_data['test']
    res = sorted(res,key = lambda t:t['iter'])
    teL_s = [t['info'][0] for t in res]
    teA_s = [t['info'][1] for t in res]
    return (iter_s,trL_s,trA_s,teL_s,teA_s)

