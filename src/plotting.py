from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
def draw_trajectories_2D(trajectories, fig=None, axs=None, colors=['b','y','c','k','m'], label=None, mark_beginning=False, plot_xt=False, segments=[], labelfun=None, title=""):
        if fig is None or axs is None:
            if(plot_xt):
                fig = plt.figure(constrained_layout=True)
                axs = fig.subplot_mosaic([['main', 'main'], ['x0', 'x1']], gridspec_kw={'height_ratios': [2, 1]})
                ax = axs['main']
            else:
                fig,axs = plt.subplots()
                ax = axs
        
        assert trajectories.d == 2 or trajectories.d is None, "Cannot draw in 2D, the dimension is not 2."
        
        ax = axs['main'] if plot_xt else axs


        for i, x in enumerate(trajectories.x_list):
            x0 = x.T[0]
            x1 = x.T[1]

            ax.scatter(x0, x1, c=colors[i % len(colors)], s=2, label = label if i == 0 else None)
            if mark_beginning:
                ax.scatter(x0[0], x1[0], c='b', s=6)
            if len(segments) > 0:
                for seg_index in segments[i]:
                    ax.scatter(x0[seg_index], x1[seg_index], marker='|', c='r', linewidths=0.7)
            


            if plot_xt:
                axs['x0'].scatter(trajectories.t_list[i], x0, c=colors[i % len(colors)], s=1)
                axs['x1'].scatter(trajectories.t_list[i], x1, c=colors[i % len(colors)], s=1)
                if len(segments) > 0:
                    for seg_index in segments[i]:
                        axs['x0'].scatter(trajectories.t_list[i][seg_index], x0[seg_index], marker='|', c='r', linewidths=0.7)
                        axs['x1'].scatter(trajectories.t_list[i][seg_index], x1[seg_index], marker='|', c='r', linewidths=0.7)

                    
        
        ax.set_ylabel('$x_1$')
        ax.set_xlabel('$x_0$')
        if not label is None:
            ax.legend()

        if plot_xt:
            axs['x0'].set_xlabel('$t$')
            axs['x0'].set_ylabel('$x_0$')
            axs['x1'].set_xlabel('$t$')
            axs['x1'].set_ylabel('$x_1$')

        # ADDING LABEL CONTROUR
        if not labelfun is None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            nx0 = np.linspace(xmin, xmax, 100)
            nx1 = np.linspace(ymin, ymax, 100)
            x0,x1 = np.meshgrid(nx0, nx1)

            label_result = [[labelfun([x0[i,j], x1[i,j]]) for j in range(x1.shape[1])] for i in range(x0.shape[0])]

            #TODO revisit
            #ax.contourf(x0,x1,label_result, cmap=ListedColormap(colors = ['c','r','y','k','m']), alpha=0.1)
            n = max([max(l) for l in label_result]) + 1
            ax.imshow(label_result, extent=(xmin,xmax,ymin,ymax), origin="lower", cmap=ListedColormap(colors = ['b','r','y','k','m'][:n]), alpha=0.2, aspect='auto')
            
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin, ymax)

        ax.set_title(title)

        return fig,axs

def draw_trajectories_3D(trajectories, fig=None, axs=None, colors=['b','y','c','k','m'], label=None, mark_beginning=False, plot_xt=False, segments=[], labelfun=None, title=""):
        if fig is None or axs is None:
            if(plot_xt):
                fig = plt.figure(constrained_layout=True)
                axs = fig.subplot_mosaic([['main', 'main', 'main'], ['x0', 'x1', 'x2']], gridspec_kw={'height_ratios': [2, 1]}, per_subplot_kw={'main' : {'projection': '3d'}})
                ax = axs['main']
            else:
                fig,axs = plt.subplots()
                ax = axs
        
        assert trajectories.d == 3 or trajectories.d is None, "Cannot draw in 3D, the dimension is not 3."
        
        ax = axs['main'] if plot_xt else axs


        for i, x in enumerate(trajectories.x_list):
            x0 = x.T[0]
            x1 = x.T[1]
            x2 = x.T[2]

            ax.scatter(x0, x1, x2, c=colors[i % len(colors)], s=2, label = label if i == 0 else None)
            if mark_beginning:
                ax.scatter(x0[0], x1[0], x2[0], c='b', s=6)
            if len(segments) > 0:
                for seg_index in segments[i]:
                    ax.scatter(x0[seg_index], x1[seg_index], x2[seg_index], marker='|', c='r', linewidths=0.7)
            


            if plot_xt:
                axs['x0'].scatter(trajectories.t_list[i], x0, c=colors[i % len(colors)], s=1)
                axs['x1'].scatter(trajectories.t_list[i], x1, c=colors[i % len(colors)], s=1)
                axs['x2'].scatter(trajectories.t_list[i], x2, c=colors[i % len(colors)], s=1)
                if len(segments) > 0:
                    for seg_index in segments[i]:
                        axs['x0'].scatter(trajectories.t_list[i][seg_index], x0[seg_index], marker='|', c='r', linewidths=0.7)
                        axs['x1'].scatter(trajectories.t_list[i][seg_index], x1[seg_index], marker='|', c='r', linewidths=0.7)
                        axs['x2'].scatter(trajectories.t_list[i][seg_index], x2[seg_index], marker='|', c='r', linewidths=0.7)

                    
        
        ax.set_ylabel('$x_1$')
        ax.set_xlabel('$x_0$')
        if not label is None:
            ax.legend()

        if plot_xt:
            axs['x0'].set_xlabel('$t$')
            axs['x0'].set_ylabel('$x_0$')
            axs['x1'].set_xlabel('$t$')
            axs['x1'].set_ylabel('$x_1$')
            axs['x2'].set_xlabel('$t$')
            axs['x2'].set_ylabel('$x_2$')

        # ADDING LABEL CONTROUR (TODO REVISIT)
        # if not labelfun is None:
        #     xmin, xmax = ax.get_xlim()
        #     ymin, ymax = ax.get_ylim()
        #     zmin, zmax = ax.get_zlim()
        #     nx0 = np.linspace(xmin, xmax, 100)
        #     nx1 = np.linspace(ymin, ymax, 100)
        #     nx2 = np.linspace(zmin, zmax, 100)
        #     x0,x1,x2 = np.meshgrid(nx0, nx1, nx2)

        #     label_result = [[[labelfun([x0[i, j, k], x1[i, j, k], x2[i, j, k]]) for k in range(x2.shape[2])] for j in range(x1.shape[1])] for i in range(x0.shape[0])]

        #     #ax.contourf(x0,x1,label_result, cmap=ListedColormap(colors = ['c','r','y','k','m']), alpha=0.1)
        #     n = max(max([max(l) for l in label_result])) + 1
        #     label_result = np.array(label_result)

        #     #ax.voxels(x0, x1, x2, label_result, cmap=ListedColormap(colors=['b', 'r', 'y', 'k', 'm'][:n]), alpha=0.2)

        #     #ax.imshow(label_result, extent=(xmin,xmax,ymin,ymax), origin="lower", cmap=ListedColormap(colors = ['b','r','y','k','m'][:n]), alpha=0.2, aspect='auto')
            
        #     ax.set_xlim(xmin,xmax)
        #     ax.set_ylim(ymin, ymax)

        ax.set_title(title)

        return fig,axs

def draw_trajectories_ND(trajectories, fig=None, axs=None, colors=['b','y','c','k','m'], label=None, mark_beginning=False, plot_xt=False, segments=[], labelfun=None, title=""):
    num_plots = trajectories.d
    if num_plots is None:
        return
    if fig is None or axs is None:
        if num_plots == 1:
            # Create the figure and subplots
            fig, axs = plt.subplots(1, 1)
            axs = [axs]
        else:
            fig = plt.figure()
            # Calculate the number of rows and columns based on the number of plots
            rows = (num_plots + 1) // 2
            cols = 2
            gs = gridspec.GridSpec(rows, 4)
            axs = []
            for i in range(num_plots - 1):
                if i % 2 == 0:
                    axs.append(plt.subplot(gs[i//2, 0:2]))
                else:
                    axs.append(plt.subplot(gs[i//2, 2:]))
            if num_plots % 2 == 0:
                axs.append(plt.subplot(gs[(num_plots-1)//2, 2:]))
            else:
                axs.append(plt.subplot(gs[(num_plots-1)//2, 1:3]))
            
    
    for i, x in enumerate(trajectories.x_list):
        for j in range(num_plots):
            ax = axs[j]
            ax.scatter(trajectories.t_list[i], x.T[j], c=colors[i % len(colors)], s=1, label = label if i == 0 else None)
            if len(segments) > 0:
                for seg_index in segments[i]:
                    ax.scatter(trajectories.t_list[i][seg_index], x.T[j][seg_index], marker='|', c='r', linewidths=0.7)
            ax.set_xlabel('$t$')
            ax.set_ylabel('$x_' + str(j) + '$')
            if not label is None:
                ax.legend()

    return fig, axs



def plot_with_modes_2D(X, modes, fig=None, ax=None, colors=['b','r','y','k','m'], labelfun=None):
    n = np.max(modes) + 1
    if n > len(colors):
        raise Exception("The number of modes is greater than the number of colors provided. Aborting to avoid confusion.")
    colors=colors[:n]
    
    if fig is None or ax is None:
        fig,ax = plt.subplots()
    
    labeled = [False] * n

    N = X.shape[0]
    for i in range(N):
        x = X[i]
        mode = modes[i]

        if labeled[mode]:
            label = None
        else:
            label = 'Mode ' + str(mode)
            labeled[mode] = True
        
        ax.scatter(x[0], x[1], c = colors[mode % len(colors)], label=label, s=2)
    
    ax.legend()

    # ADDING LABEL CONTROUR
    if not labelfun is None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        nx0 = np.linspace(xmin, xmax, 100)
        nx1 = np.linspace(ymin, ymax, 100)
        x0,x1 = np.meshgrid(nx0, nx1)

        label_result = [[labelfun([x0[i,j], x1[i,j]]) for j in range(x1.shape[1])] for i in range(x0.shape[0])]

        #TODO revisit
        #ax.contourf(x0,x1,label_result, cmap=ListedColormap(colors = ['c','r','y','k','m']), alpha=0.1)
        ax.imshow(label_result, extent=(xmin,xmax,ymin,ymax), origin="lower", cmap=ListedColormap(colors = ['b','r','y','k','m'][:n]), alpha=0.2, aspect='auto')
        
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin, ymax)

    return fig,ax

def plot_with_modes_3D(X, modes, fig=None, ax=None, colors=['b','r','y','k','m'], labelfun=None):
    n = np.max(modes) + 1
    if n > len(colors):
        raise Exception("The number of modes is greater than the number of colors provided. Aborting to avoid confusion.")
    colors=colors[:n]
    
    if fig is None or ax is None:
        fig,ax = plt.subplots(subplot_kw={'projection' : '3d'})
    
    labeled = [False] * n

    N = X.shape[0]
    for i in range(N):
        x = X[i]
        mode = modes[i]

        if labeled[mode]:
            label = None
        else:
            label = 'Mode ' + str(mode)
            labeled[mode] = True
        
        ax.scatter(x[0], x[1], x[2], c = colors[mode % len(colors)], label=label, s=2)
    
    ax.legend()

    # ADDING LABEL CONTROUR
    if not labelfun is None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        zmin, zmax = ax.get_zlim()
        nx0 = np.linspace(xmin, xmax, 100)
        nx1 = np.linspace(ymin, ymax, 100)
        nx2 = np.linspace(zmin, zmax, 100)
        x0,x1,x2 = np.meshgrid(nx0, nx1, nx2)

        label_result = [[[labelfun([x0[i, j, k], x1[i, j, k], x2[i, j, k]]) for k in range(x2.shape[2])] for j in range(x1.shape[1])] for i in range(x0.shape[0])]

        #TODO revisit
        #ax.contourf(x0,x1,label_result, cmap=ListedColormap(colors = ['c','r','y','k','m']), alpha=0.1)
        #ax.imshow(label_result, extent=(xmin,xmax,ymin,ymax), origin="lower", cmap=ListedColormap(colors = ['b','r','y','k','m'][:n]), alpha=0.2, aspect='auto')
        
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin, ymax)

    return fig,ax

def plot_classifier_linear(classifier, X, modes, fig=None, ax=None, colors=['b', 'r', 'y', 'k', 'm']):
    nc = classifier.classes_.shape[0]
    colors = colors[:nc]
    fig, ax = plot_with_modes_2D(X, modes, fig=fig, ax=ax, colors=colors)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    nx0 = np.linspace(xmin, xmax, 1000)
    nx1 = np.linspace(ymin, ymax, 1000)
    x0,x1 = np.meshgrid(nx0, nx1)
    xd = np.array([xmin, xmax])

    Z = classifier.predict(np.c_[x0.ravel(), x1.ravel()])
    ax.contourf(x0, x1, Z.reshape(x0.shape), cmap=ListedColormap(colors), alpha=0.3)

    # for i in range(nc):
    #     ax.scatter(*X[modes==i].T, s=2, alpha=0.9, color=colors[i])

    try:
        estis = classifier.estimators_
        coefs = []
        intercepts = []
        for clf in estis:
            coefs.append(clf.coef_)
            intercepts.append(clf.intercept_)
    except:
        coefs = classifier.coef_
        intercepts = classifier.intercept_

    for i in range(nc):
        # clf = estis[i]
        # b = clf.intercept_
        # w = clf.coef_

        b = intercepts[i]
        w = coefs[i]

        w1, w2 = w.T
        c = -b/w2
        m = -w1/w2
        yd = m*xd + c
        # yd = m[i]*xd + c[i]
        ax.plot(xd, yd, 'k', lw=1, ls='--')

        xd = np.linspace(xmin, xmax,1000)

    for i in range(nc):
        # esti1 = estis[i]
        # w1 = esti1.coef_
        # b1 = esti1.intercept_
        w1 = coefs[i]
        b1 = intercepts[i]
        slope1 = -w1.T[0]/w1.T[1]
        wnorm1 = np.linalg.norm(w1)
        for j in range(i+1, nc):
            # esti2 = estis[j]
            # w2 = esti2.coef_
            # b2 = esti2.intercept_
            w2 = coefs[j]
            b2 = intercepts[j]
            slope2 = -w2.T[0]/w2.T[1]
            wnorm2 = np.linalg.norm(w2)

            # decision function
            w = w1 - w2
            b = b1 - b2

            a1, a2 = w.T
            c = -b/a2
            m = -a1/a2
            slope_mid = m
            yd = m*xd + c
            
            ax.plot(xd, yd, 'k', lw=1, ls='dotted', alpha=0.6)
            newx = []
            newy = []
            for q in range(len(xd)):
                if classifier.predict([[xd[q], yd[q]]])[0] == i or classifier.predict([[xd[q], yd[q]]])[0] == j:
                    newx.append(xd[q])
                    newy.append(yd[q])
            ax.plot(newx, newy, 'k', lw = 1)

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin, ymax)
    return fig, ax

