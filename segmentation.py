import matplotlib.pyplot as plt
import numpy as np
def segment_time_series(t_points, x_points, tol):
  # Assert that the time points and x points arrays have the same length
  l = t_points.shape[0]
  assert l == x_points.shape[0]
  
  segments = []
  current_start_index = 0

  n=3
  
  
  i = 0
  while i < l:
    if i >= n and i < l -n:
        # Extract the time and x points for the current position and the
        # forward and backward positions
        current_time, current_value = t_points[i], x_points[i]
        forward_time_1, forward_value_1 = t_points[i+1], x_points[i+1]
        forward_time_2, forward_value_2 = t_points[i+2], x_points[i+2]
        backward_time_1, backward_value_1 = t_points[i-1], x_points[i-1]
        backward_time_2, backward_value_2 = t_points[i-2], x_points[i-2]
        
        forward_derivative = (-3*current_value + 4*forward_value_1 - forward_value_2) / (forward_time_2 - current_time)
        backward_derivative = (3*current_value - 4*backward_value_1 + backward_value_2) / (current_time - backward_time_2)
        
        if np.linalg.norm(forward_derivative - backward_derivative) > tol:
            # If the difference between the forward and backward derivatives is
            # greater than the tolerance, we start a new segment
            segments.append((current_start_index, i))
            current_start_index = i
            # skip n points once we notice a change
            i = i+n
    i+=1
  # Add the final segment to the list of segments
  segments.append((current_start_index, l))
  
  return segments


from paper_utils import simulation_ode_3, get_modetr, get_event, get_labeltest

modetr = get_modetr(0)
event = get_event(0)
labeltest = get_labeltest(0)
y0 = [[-1,1],[1,4],[2,-3]]

T = 5
stepsize = 0.05

t_list, y_list = simulation_ode_3(modetr, event, labeltest, y0, T, stepsize)

for i, temp_y in enumerate(y_list):
    y0_list = temp_y.T[0]
    y1_list = temp_y.T[1]
    plt.scatter(y0_list,y1_list,c='b',s=1)
    

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()


for i, temp_y in enumerate(y_list):
    segs = segment_time_series(t_list[i], y_list[i], 1)
    print(segs)

    for j in range(len(segs)):
        plt.scatter(y_list[i][segs[j][0]][0], y_list[i][segs[j][0]][1], c = 'r')

plt.show()