import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
def arrange_subplots(num_plots):
    if num_plots == 1:
        # Create the figure and subplots
        fig, axs = plt.subplots(1, 1)
        axs = [axs]
    else:
        # Calculate the number of rows and columns based on the number of plots
        rows = (num_plots + 1) // 2
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
        
        fig = plt.gcf()

    # Customize the subplots as needed
    for i in range(num_plots):
        ax = axs[i]  # Get the appropriate subplot
        ax.plot([1, 2, 3], [4, 5, 6])  # Example plot for demonstration
        ax.set_title(str(i))

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
arrange_subplots(1)   # One subplot
arrange_subplots(2)   # Two subplots, arranged side by side
arrange_subplots(3)   # Three subplots, arranged 2 on top, 1 below
arrange_subplots(4)   # Four subplots, arranged 2 on top, 2 below
arrange_subplots(5)   # Five subplots, arranged 2 on top, 2 below with the last subplot in the middle of the last row
arrange_subplots(6)
arrange_subplots(7)
arrange_subplots(8)
arrange_subplots(9)
