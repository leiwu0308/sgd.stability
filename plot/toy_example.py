import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 
mpl.rc('text',usetex=True)
mpl.rcParams['legend.fontsize']=15
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 22


f1 = lambda x: min(x*x,0.1*(x-1)*(x-1))
f2 = lambda x: min(x*x,1.9*(x-1)*(x-1))
f = lambda x: 0.5*(f1(x) + f2(x))


def df1(x):
    g1 = x*x 
    g2 = 0.1*(x-1)*(x-1)
    if g1<=g2:
        return 2*x
    else:
        return 0.2*(x-1)

def df2(x):
    g1 = x*x 
    g2 = 1.9*(x-1)*(x-1)
    if g1<= g2:
        return 2*x
    else:
        return 3.8*(x-1)


def plot_toy_example():
    # plot landscape
    x = np.linspace(-0.5,1.6,400)
    y = list(map(f, x))
    y1 = list(map(f1,x))
    y2 = list(map(f2,x))
    plt.plot(x,y,lw=3,color='k',label=r'$f=(f_1+f_2)/2$')
    plt.plot(x,y1,lw=2,color='g',linestyle='--',label=r'$f_1$')
    plt.plot(x,y2,lw=2,color='b',linestyle='--',label=r'$f_2$')
    plt.xlim([-0.5,1.6])
    plt.ylim([0.0,0.4])



    # Run SGD
    max_iters = 500
    lr = 0.7
    x_s = []
    y_s = []


    eps = 1e-5
    x_now = 1+eps
    for i in range(max_iters):
        x_s.append(x_now)
        y_s.append(f(x_now))
        z = np.random.rand()
        if z <= 0.5:
            g = - df1(x_now)
        else:
            g = - df2(x_now)
        x_now = x_now + lr * g 


    plt.plot(x_s,y_s,'-r',lw=2,label='SGD Trajectory')

    for i in range(100):
        dx = x_s[i+1] - x_s[i]
        dy = y_s[i+1] - y_s[i]
        plt.arrow(x_s[i],y_s[i],dx/2,dy/2,lw=0,length_includes_head=True, 
            head_width=.02,color='r',fc='red',ec='red')
    plt.legend()
    plt.xlabel(r'x')

    plt.savefig('figures/toy_example.pdf',bbox_inches='tight')

def plot_diagram():
    batch_sizes = [1,2,3]
    learning_rates = [0.1,0.2,0.3]

    plt.figure(figsize=(12,5))
    ax = plt.subplot(1,2,1)
    x = np.ones(100)*2/0.1
    y = np.linspace(0,4*np.sqrt(batch_sizes[-1])/0.1,100)
    plt.plot(x,y,'-k',lw=2)
    for bz in batch_sizes:
        x = np.linspace(0,2/0.1,100)
        y = np.ones(100)*np.sqrt(bz)/0.1
        plt.plot(x,y,'-k',lw=2)
        plt.text(1/0.1,np.sqrt(bz)/0.1-1.8,r'B=%d'%(bz),fontsize=20)
    plt.xlim([0,28])
    plt.ylim([0,28])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Sharpness: a')
    plt.ylabel('Non-uniformity: s')
    plt.text(2/0.1+1,20,r'$a=2/\eta$',fontsize=20)
    plt.arrow(26,0,2,0,lw=0,length_includes_head=True, 
            head_width=.5,color='k',fc='k',ec='k')
    plt.arrow(0,26,0,2,lw=0,length_includes_head=True, 
            head_width=.5,color='k',fc='k',ec='k')
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax=plt.subplot(1,2,2)
    for lr in learning_rates:
        x = np.linspace(0,2/lr,100)
        y = np.ones(100)*1/lr
        plt.plot(x,y,'-k',lw=2)

        x = np.ones(100)*2/lr
        y = np.linspace(0,1/lr,100)
        plt.plot(x,y,'-k',lw=2)

        plt.text(1/lr-2.2,1/lr-1.1,r'$\eta=%.1f$'%(lr),fontsize=20)
    plt.xlim([0,24])
    plt.ylim([0,14])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Sharpness: a')
    plt.ylabel('Non-uniformity: s')
    plt.arrow(22,0,2,0,lw=0,length_includes_head=True, 
            head_width=.4,color='k',fc='k',ec='k')
    plt.arrow(0,12,0,2,lw=0,length_includes_head=True, 
            head_width=.4,color='k',fc='k',ec='k')
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.savefig('figures/diagram.pdf',bbox_inches='tight')

if __name__ == '__main__':
    plot_diagram()