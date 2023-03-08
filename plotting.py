import matplotlib.pyplot as plt
def draw_trajectories_2D(trajectories, fig=None, axs=None, color='b', label=None, mark_beginning=False, plot_xt=False, segments=[]):
        if fig is None or axs is None:
            if(plot_xt):
                fig = plt.figure(constrained_layout=True)
                axs = fig.subplot_mosaic([['main', 'main'], ['x0', 'x1']], gridspec_kw={'height_ratios': [2, 1]})
                ax = axs['main']
            else:
                fig,axs = plt.subplots()
                ax = axs
        
        assert trajectories.d == 2, "Cannot draw in 2D, the dimension is not 2."
        
        ax = axs['main'] if plot_xt else axs

        # colors = ['b','y','c','k','m']
        for i, x in enumerate(trajectories.x_list):
            x0 = x.T[0]
            x1 = x.T[1]

            ax.scatter(x0, x1, c=color, s=2, label = label if i == 0 else None)
            if mark_beginning:
                ax.scatter(x0[0], x1[0], c='r', s=6)
            if len(segments) > 0:
                for seg_index in segments[i]:
                    ax.scatter(x0[seg_index], x1[seg_index], marker='|', c='r', linewidths=0.1)
            
            if not label is None:
                ax.legend()

            if plot_xt:
                axs['x0'].scatter(trajectories.t_list[i], x0, c=color, s=1)
                axs['x1'].scatter(trajectories.t_list[i], x1, c=color, s=1)
                if len(segments) > 0:
                    for seg_index in segments[i]:
                        axs['x0'].scatter(trajectories.t_list[i][seg_index], x0[seg_index], marker='|', c='r', linewidths=0.1)
                        axs['x1'].scatter(trajectories.t_list[i][seg_index], x1[seg_index], marker='|', c='r', linewidths=0.1)
                if not label is None:
                    axs['x0'].legend()
                    axs['x1'].legend()
        
        ax.set_ylabel('$x_1$')
        ax.set_xlabel('$x_0$')

        if plot_xt:
            axs['x0'].set_xlabel('$t$')
            axs['x0'].set_ylabel('$x_0$')
            axs['x1'].set_xlabel('$t$')
            axs['x1'].set_ylabel('$x_1$')
        
        return fig,axs