import numpy as np
def which_segment(i, segments):
    r = 0
    while r < len(segments) and i >= segments[r]:
        r += 1
    return r - 1

def which_segment_all(N, segments):
    segments_per_point = np.zeros(N)
    for i in range(N):
        segments_per_point[i] = which_segment(i, segments)