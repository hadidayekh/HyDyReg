from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def draw_trajectories_2D(trajectories, fig=None, axs=None, colors=[plt.cm.tab10(i) for i in range(10)], label=None, mark_beginning=False, plot_xt=False, segments=[], labelfun=None, title=""):

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

            ax.scatter(x0, x1, color=colors[i%len(colors)], s=2, label = label if i == 0 else None)

            if mark_beginning:
                ax.scatter(x0[0], x1[0], color=colors[i%len(colors)], marker='x', s=8)
            if len(segments) > 0:
                ax.plot(x0[:segments[i][0]], x1[:segments[i][0]], color=colors[i%len(colors)], alpha=0.2)
                for j in range(len(segments[i])-1):
                    seg_index = segments[i][j]
                    ax.scatter(x0[seg_index], x1[seg_index], marker='|', color='r', linewidths=0.7)
                    ax.plot(x0[segments[i][j]:segments[i][j+1]], x1[segments[i][j]:segments[i][j+1]], color=colors[i%len(colors)], alpha=0.2)
                seg_index = segments[i][-1]
                ax.scatter(x0[seg_index], x1[seg_index], marker='|', color='r', linewidths=0.7)
                ax.plot(x0[segments[i][-1]:], x1[segments[i][-1]:], color=colors[i%len(colors)], alpha=0.2)
            else:
                ax.plot(x0, x1, color=colors[i%len(colors)], alpha=0.2)

            


            if plot_xt:
                axs['x0'].scatter(trajectories.t_list[i], x0, color=colors[i % len(colors)], s=1)
                axs['x1'].scatter(trajectories.t_list[i], x1, color=colors[i % len(colors)], s=1)
                if len(segments) > 0:
                    for seg_index in segments[i]:
                        axs['x0'].scatter(trajectories.t_list[i][seg_index], x0[seg_index], marker='|', color='r', linewidths=0.7)
                        axs['x1'].scatter(trajectories.t_list[i][seg_index], x1[seg_index], marker='|', color='r', linewidths=0.7)

                    
        
        ax.set_ylabel('$x^{(2)}$')
        ax.set_xlabel('$x^{(1)}$')
        if not label is None:
            ax.legend()

        if plot_xt:
            axs['x0'].set_xlabel('$t$')
            axs['x0'].set_ylabel('$x^{(1)}$')
            axs['x1'].set_xlabel('$t$')
            axs['x1'].set_ylabel('$x^{(2)}$')

        if not labelfun is None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            nx0 = np.linspace(xmin, xmax, 200)
            nx1 = np.linspace(ymin, ymax, 200)
            x0,x1 = np.meshgrid(nx0, nx1)
            label_result = [[labelfun([x0[i,j], x1[i,j]]) for j in range(x1.shape[1])] for i in range(x0.shape[0])]

            #TODO revisit
            n = max([max(l) for l in label_result]) + 1
            cmap = ListedColormap([plt.cm.Set3(i) for i in range(n)])
            c1 = plt.cm.tab20(np.linspace(0,19,20).astype(int))
            c2 = plt.cm.Pastel2(np.linspace(0,7,8).astype(int))
            cmap = ListedColormap(np.vstack((c1,c2)))
            cmap = ListedColormap([cmap(i % n) for i in range(n)])          
            ax.imshow(label_result, extent=(xmin,xmax,ymin,ymax), origin="lower", cmap=cmap, alpha=0.5, aspect='auto')
            # for c in cnt.collections:
            
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin, ymax)

        ax.set_title(title)

        return fig,axs

def draw_trajectories_3D(trajectories, fig=None, axs=None, colors=[plt.cm.tab10(i) for i in range(10)], label=None, mark_beginning=False, plot_xt=False, segments=[], labelfun=None, title=""):
        if fig is None or axs is None:
            if(plot_xt):
                fig = plt.figure(constrained_layout=True)
                axs = fig.subplot_mosaic([['main', 'main', 'main'], ['x0', 'x1', 'x2']], gridspec_kw={'height_ratios': [2, 1]}, per_subplot_kw={'main' : {'projection': '3d'}})
                ax = axs['main']
            else:
                fig,axs = plt.subplots(subplot_kw={'projection':'3d'})
                ax = axs
        
        assert trajectories.d == 3 or trajectories.d is None, "Cannot draw in 3D, the dimension is not 3."
        
        ax = axs['main'] if plot_xt else axs


        for i, x in enumerate(trajectories.x_list):
            x0 = x.T[0]
            x1 = x.T[1]
            x2 = x.T[2]
            ax.scatter(x0, x1, x2, color=colors[i % len(colors)], s=2, label = label if i == 0 else None)
            if mark_beginning:
                ax.scatter(x0[0], x1[0], x2[0], color='b', s=6)
            if len(segments) > 0:
                for seg_index in segments[i]:
                    ax.scatter(x0[seg_index], x1[seg_index], x2[seg_index], marker='|', color='r', linewidths=0.7)
            


            if plot_xt:
                axs['x0'].scatter(trajectories.t_list[i], x0, color=colors[i % len(colors)], s=1)
                axs['x1'].scatter(trajectories.t_list[i], x1, color=colors[i % len(colors)], s=1)
                axs['x2'].scatter(trajectories.t_list[i], x2, color=colors[i % len(colors)], s=1)
                if len(segments) > 0:
                    for seg_index in segments[i]:
                        axs['x0'].scatter(trajectories.t_list[i][seg_index], x0[seg_index], marker='|', color='r', linewidths=0.7)
                        axs['x1'].scatter(trajectories.t_list[i][seg_index], x1[seg_index], marker='|', color='r', linewidths=0.7)
                        axs['x2'].scatter(trajectories.t_list[i][seg_index], x2[seg_index], marker='|', color='r', linewidths=0.7)

                    

        ax.set_ylabel('$x^{(2)}$')
        ax.set_xlabel('$x^{(1)}$')
        if not label is None:
            ax.legend()

        if plot_xt:
            axs['x0'].set_xlabel('$t$')
            axs['x0'].set_ylabel('$x^{(1)}$')
            axs['x1'].set_xlabel('$t$')
            axs['x1'].set_ylabel('$x^{(2)}$')
            axs['x2'].set_xlabel('$t$')
            axs['x2'].set_ylabel('$x^{(3)}$')

        # ADDING LABEL CONTROUR (TODO REVISIT)
        # if not labelfun is None:




            

        ax.set_title(title)

        return fig,axs

def draw_trajectories_ND(trajectories, fig=None, axs=None, colors=[plt.cm.tab10(i) for i in range(10)], label=None, mark_beginning=False, plot_xt=False, segments=[], labelfun=None, title=""):
    num_plots = trajectories.d
    assert not num_plots is None
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
            gs = gridspec.GridSpec(rows, 4, hspace=0.4,wspace=0.8)
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
            ax.scatter(trajectories.t_list[i], x.T[j], color=colors[i % len(colors)], s=1, label = label if i == 0 else None)
            if len(segments) > 0:
                for seg_index in segments[i]:
                    ax.scatter(trajectories.t_list[i][seg_index], x.T[j][seg_index], marker='|', color='r', linewidths=0.7)
            ax.set_xlabel('$t$')
            ax.set_ylabel('$x^{(' + str(j+1) + ')}$')
            if not label is None:
                ax.legend()

    return fig, axs



def plot_with_modes_2D(X, modes, fig=None, ax=None, colors=[plt.cm.tab10(i) for i in range(10)], labelfun=None):
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
        
        ax.scatter(x[0], x[1], color = colors[mode % len(colors)], label=label, s=2)
    
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
        ax.imshow(label_result, extent=(xmin,xmax,ymin,ymax), origin="lower", cmap=ListedColormap(colors = ['b','r','y','k','m'][:n]), alpha=0.2, aspect='auto')
        
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin, ymax)

    return fig,ax

def plot_with_modes_3D(X, modes, fig=None, ax=None, colors=[plt.cm.tab10(i) for i in range(10)], labelfun=None):
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
        
        ax.scatter(x[0], x[1], x[2], color = colors[mode % len(colors)], label=label, s=2)
    
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

        b = intercepts[i]
        w = coefs[i]

        w1, w2 = w.T
        c = -b/w2
        m = -w1/w2
        yd = m*xd + c
        ax.plot(xd, yd, 'k', lw=1, ls='--')

        xd = np.linspace(xmin, xmax,1000)

    for i in range(nc):
        w1 = coefs[i]
        b1 = intercepts[i]
        slope1 = -w1.T[0]/w1.T[1]
        wnorm1 = np.linalg.norm(w1)
        for j in range(i+1, nc):
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

def plot_train_test_2D(X_train, X_test):
    X0train = X_train.T[0]
    X1train = X_train.T[1]
    X0test = X_test.T[0]
    X1test = X_test.T[1]

    f, a = plt.subplots()

    a.scatter(X0train, X1train, marker='.', label="Training points")
    a.scatter(X0test, X1test, marker='+', label="Validation points")
    a.set_xlabel("$x^{(1)}$")
    a.set_ylabel("$x^{(2)}$")
    a.legend()

    return f,a

def plot_Xfit(model, X=None, modes=None, with_contour=True):
    if X is None:
        X = model.Xfit_
        modes = model.modesfit_
    assert X.shape[1] == 2
    c1 = plt.cm.tab20(np.linspace(0,19,20).astype(int))
    c2 = plt.cm.Pastel2(np.linspace(0,7,8).astype(int))
    colors = ListedColormap(np.vstack((c1,c2)))
    Ncolors = colors.N

    f, a = plt.subplots()
    
    for mode in np.unique(modes):
        a.scatter(*X[modes==mode].T, color=colors(mode%Ncolors), label=f"Mode {mode+1}", s=3)

    a.set_xlabel("$x^{(1)}$")
    a.set_ylabel("$x^{(2)}$")
    a.legend()


    if with_contour:
        labelfun = model.labelfun_
        xmin, xmax = a.get_xlim()
        ymin, ymax = a.get_ylim()
        nx0 = np.linspace(xmin, xmax, 200)
        nx1 = np.linspace(ymin, ymax, 200)
        x0,x1 = np.meshgrid(nx0, nx1)
        label_result = [[labelfun([x0[i,j], x1[i,j]]) for j in range(x1.shape[1])] for i in range(x0.shape[0])]

        #TODO revisit
        n = max([max(l) for l in label_result]) + 1
        cmap = ListedColormap([plt.cm.Set3(i) for i in range(n)])
        c1 = plt.cm.tab20(np.linspace(0,19,20).astype(int))
        c2 = plt.cm.Pastel2(np.linspace(0,7,8).astype(int))
        cmap = ListedColormap(np.vstack((c1,c2)))
        cmap = ListedColormap([cmap(i % n) for i in range(n)])          
        a.imshow(label_result, extent=(xmin,xmax,ymin,ymax), origin="lower", cmap=cmap, alpha=0.2, aspect='auto')
        # for c in cnt.collections:
        
        a.set_xlim(xmin,xmax)
        a.set_ylim(ymin, ymax)
    return f, a

# ================================================================================================================
# Public plotting functions. Each returns (fig, ax); a figure is created only when ax is None; nothing is shown or
# saved (use fig.savefig). The functions above are the ones used by the thesis experiments.
# ================================================================================================================

MODE_COLORS = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set2(i) for i in range(8)]


def _mode_color(j):
    return MODE_COLORS[int(j) % len(MODE_COLORS)]


def _new_axes(ax, D, figsize=None):
    if ax is not None:
        return ax.figure, ax
    if D == 2:
        fig, ax = plt.subplots(figsize=figsize or (5.2, 4.4))
        return fig, ax
    if D == 3:
        fig = plt.figure(figsize=figsize or (5.6, 4.8))
        return fig, fig.add_subplot(111, projection="3d")
    fig, ax = plt.subplots(D, 1, sharex=True, figsize=figsize or (7, 1.6 * D + 0.6))
    return fig, ax


def _names(D, names=None):
    return list(names) if names else [f"$x_{{{i + 1}}}$" for i in range(D)]


def _draw(ax, t, x, D, color, lw=1.4, ls="-", label=None, alpha=1.0):
    if D == 2:
        ax.plot(x[:, 0], x[:, 1], ls, color=color, lw=lw, label=label, alpha=alpha)
    elif D == 3:
        ax.plot(x[:, 0], x[:, 1], x[:, 2], ls, color=color, lw=lw, label=label, alpha=alpha)
    else:
        for i in range(D):
            ax[i].plot(t, x[:, i], ls, color=color, lw=lw, label=label if i == 0 else None, alpha=alpha)


def _finish(ax, D, names=None):
    n = _names(D, names)
    if D == 2:
        ax.set_xlabel(n[0]); ax.set_ylabel(n[1])
    elif D == 3:
        ax.set_xlabel(n[0]); ax.set_ylabel(n[1]); ax.set_zlabel(n[2])
    else:
        for i in range(D):
            ax[i].set_ylabel(n[i])
        ax[-1].set_xlabel("$t$")


def plot_trajectories(trajs, ax=None, labelfun=None, names=None, mark_start=True, lw=1.4):
    """Trajectories in the state space (2D, 3D) or as functions of time (other dimensions). With `labelfun`
    (a function x -> mode, e.g. `system.labelfun` or `model.labelfun_`), each piece is coloured by its mode."""
    D = trajs.x_list[0].shape[1]
    fig, ax = _new_axes(ax, D)
    for k, (t, x) in enumerate(zip(trajs.t_list, trajs.x_list)):
        x = np.asarray(x)
        if labelfun is None:
            _draw(ax, t, x, D, _mode_color(k), lw=lw)
        else:
            lab = np.array([labelfun(p) for p in x])
            cuts = np.nonzero(np.diff(lab))[0] + 1
            for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(x)]):
                hi = min(b + 1, len(x))  # overlap one sample so that pieces join
                _draw(ax, np.asarray(t)[a:hi], x[a:hi], D, _mode_color(lab[a]), lw=lw)
        if mark_start and D in (2, 3):
            ax.plot(*x[0, :D], "o", color="k", ms=3)
    if labelfun is not None:
        modes = sorted({int(labelfun(p)) for x in trajs.x_list for p in np.asarray(x)})
        handles = [plt.Line2D([], [], color=_mode_color(j), lw=2, label=f"mode {j + 1}") for j in modes]
        (ax if D in (2, 3) else ax[0]).legend(handles=handles, fontsize=8, frameon=False)
    _finish(ax, D, names)
    return fig, ax


def plot_segmentation(trajs, segments, ax=None, names=None):
    """Trajectories with the change points found by segmentation (crosses) and the samples it removed (grey)."""
    D = trajs.x_list[0].shape[1]
    fig, ax = _new_axes(ax, D)
    og_t = getattr(trajs, "og_t_list", None)
    for k, (t, x) in enumerate(zip(trajs.t_list, trajs.x_list)):
        x, t = np.asarray(x), np.asarray(t)
        if og_t is not None:
            _draw(ax, np.asarray(og_t[k]), np.asarray(trajs.og_x_list[k]), D, "0.75", lw=3, alpha=0.6)
        bounds = [0] + list(segments[k]) + [len(x)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            _draw(ax, t[a:b], x[a:b], D, _mode_color(k), lw=1.4)
        for c in segments[k]:
            if 0 < c < len(x):
                if D == 2:
                    ax.plot(*x[c - 1], "x", color="k", ms=6)
                elif D > 3:
                    for i in range(D):
                        ax[i].axvline(t[c - 1], color="k", lw=0.6, ls=":")
    _finish(ax, D, names)
    return fig, ax


def plot_modes(model, ax=None, names=None):
    """Training points of a fitted model coloured by the mode they were assigned to (2D: in the plane; otherwise
    each component against the index of the point)."""
    X, modes = np.asarray(model.Xfit_), np.asarray(model.modesfit_)
    D = X.shape[1]
    n = _names(D, names)
    if D == 2:
        fig, ax = _new_axes(ax, 2)
        for j in np.unique(modes):
            ax.scatter(*X[modes == j].T, s=6, color=_mode_color(j), label=f"mode {j + 1}")
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlabel(n[0]); ax.set_ylabel(n[1])
        return fig, ax
    if ax is None:
        fig, ax = plt.subplots(D, 1, sharex=True, figsize=(7, 1.6 * D + 0.6))
    fig = np.atleast_1d(ax)[0].figure
    for i in range(D):
        for j in np.unique(modes):
            ax[i].scatter(np.nonzero(modes == j)[0], X[modes == j, i], s=4, color=_mode_color(j),
                          label=f"mode {j + 1}" if i == 0 else None)
        ax[i].set_ylabel(n[i])
    ax[0].legend(fontsize=8, frameon=False)
    ax[-1].set_xlabel("training point")
    return fig, ax


def plot_regions(model, bounds=None, ax=None, resolution=300, trajectories=None, names=None, true_labelfun=None):
    """Mode regions of a 2D model on a grid (coloured background), the learned boundaries (black), optionally the
    true boundaries (`true_labelfun`, dashed red) and trajectories on top. `bounds` = [[x1min, x1max],
    [x2min, x2max]] (default: the box of the classifier's training points, enlarged by 5%)."""
    from matplotlib.colors import ListedColormap
    if bounds is None:
        X = np.asarray(getattr(model, "X_classify_", model.Xfit_))
        lo, hi = X.min(axis=0), X.max(axis=0)
        pad = 0.05 * (hi - lo)
        bounds = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
    g1, g2 = np.meshgrid(np.linspace(*bounds[0], resolution), np.linspace(*bounds[1], resolution))
    pts = np.c_[g1.ravel(), g2.ravel()]
    lab = np.array([model.labelfun_(p) for p in pts]).reshape(g1.shape)
    fig, ax = _new_axes(ax, 2)
    n = int(max(model.n, lab.max() + 1))
    ax.pcolormesh(g1, g2, lab, cmap=ListedColormap([_mode_color(j) for j in range(n)]), vmin=-0.5, vmax=n - 0.5,
                  alpha=0.18, shading="auto")
    ax.contour(g1, g2, lab, levels=np.arange(n) + 0.5, colors="k", linewidths=1.0)
    if true_labelfun is not None:
        tl = np.array([true_labelfun(p) for p in pts]).reshape(g1.shape)
        ax.contour(g1, g2, tl, levels=np.arange(tl.max()) + 0.5, colors=[(0.75, 0.2, 0.2)], linewidths=1.0, linestyles="--")
    if trajectories is not None:
        plot_trajectories(trajectories, ax=ax, labelfun=model.labelfun_, mark_start=True, lw=1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_mode_color(j), alpha=0.4, label=f"mode {j + 1}") for j in range(n)]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
    _finish(ax, 2, names)
    return fig, ax


def plot_comparison(reference, learned, trajectories, ax=None, names=None, max_step_factor=10):
    """Trajectories of the reference (solid) and of the learned system (dashed) from the initial conditions of
    `trajectories`, on their time grids. `reference` is a system to simulate, or None to use `trajectories`
    themselves (recorded data). `learned` is an SNDS (e.g. `result.system`)."""
    D = trajectories.x_list[0].shape[1]
    fig, ax = _new_axes(ax, D)
    for k, (t, x) in enumerate(zip(trajectories.t_list, trajectories.x_list)):
        t, x = np.asarray(t, dtype=float), np.asarray(x)
        h, T = t[1] - t[0], t[-1] - t[0]
        ref = x if reference is None else reference.simulate_trajectory(h, T, [x[0]], max_step_factor=max_step_factor).x_list[0]
        _draw(ax, t[:len(ref)], ref, D, "0.35", lw=2.2, alpha=0.6, label="reference" if k == 0 else None)
        try:
            sim = learned.simulate_trajectory(h, T, [x[0]], max_step_factor=max_step_factor).x_list[0]
            _draw(ax, t[:len(sim)], sim, D, _mode_color(k), lw=1.3, ls="--", label="learned" if k == 0 else None)
        except Exception:
            pass
    (ax if D in (2, 3) else ax[0]).legend(fontsize=8, frameon=False)
    _finish(ax, D, names)
    return fig, ax
