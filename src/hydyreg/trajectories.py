"""Trajectories: container of sampled trajectories, segmentation, and CSV / tar input-output."""
import os
import tarfile
import tempfile
from copy import deepcopy

import numpy as np
from typing import List, Optional, Sequence, Tuple, Union


class Trajectories:
    '''
    A collection of trajectories.

    Attributes
    ----------
    t_list : List[ndarray]
        List of arrays of time points.
    
    x_list : List[ndarray]
        List of arrays of x points (continuous state points), each x point has dimension d.
    
    num_trajectories : int
        The number of trajectories.
    
    segmented : Bool
        Indicated whether the trajectories are segmented or not.
    
    d : int
        Dimension of the continuous state vector.


    Methods
    -------
    segment(tol, m=2)
        Returns a segmented copy of the trajectories and the segments; `self` is not modified.

    segment_trajectories(tol, m=2)
        Same as `segment`, but in place. Segments each trajectory at points where the difference between the three-point forward and three-point backward derivatives is greater than a certain tolerance. Does not support variable time steps.
    
    append_trajs(trajs):
        Appends a collection of trajectories.
    
    return_unsegmented()
        Return the unsegmented version of trajectories including the missing points that are dropped due to segmentation.
    '''
    def __init__(self, t_list, x_list):
        self.t_list = t_list
        self.x_list = x_list
        self.num_trajectories = len(x_list)
        self.segmented = False
        if self.num_trajectories > 0:
            self.d = x_list[0].shape[1]
        else:
            self.d = None
        
    def return_unsegmented(self):
        '''
        Returns an instance of `Trajectories` with the unsegmented trajectories (thus without data loss).
        If the trajectories was not segmented, it returns a deep copy of `self`.
        '''
        if self.segmented:
            return Trajectories(self.og_t_list, self.og_x_list)
        else:
            return deepcopy(self)
    
    def append_trajs(self, trajs: "Trajectories"):
        '''
        Appends trajectories to `self`.

        Parameters
        ----------
        `trajs` : `Trajectories`
            Trajectories to be appended.

        Returns
        -------
            `None`
        '''
        assert self.d == trajs.d or self.d is None or trajs.d is None
        self.t_list.extend(trajs.t_list)
        self.x_list.extend(trajs.x_list)
        self.num_trajectories += trajs.num_trajectories
        if self.d is None:
            self.d = trajs.d
        pass

    
    def segment(self, tol: float, m: int = 2) -> Tuple["Trajectories", list]:
        '''
        Segments a copy of the trajectories and leaves `self` unchanged.

        Returns (trajs, segments): `trajs` is a new `Trajectories` holding the truncated trajectories (the
        samples around each change point removed), the original ones in `og_t_list` / `og_x_list`, and
        `segmented = True`; `segments` is as returned by `segment_trajectories`.
        '''
        trajs = deepcopy(self)
        segments = trajs._segment_in_place(tol, m)
        return trajs, segments

    def segment_trajectories(self, tol, m=2):
        '''
        In-place version of `segment`: truncates the trajectories of `self` and returns the segments.
        '''
        return self._segment_in_place(tol, m)

    def _segment_in_place(self, tol, m=2):
        '''
        m: half-width of the window of the change-point statistic (see `_segment_windowed`). m = 2 (default)
        is the statistic of the thesis, described below.

        - segments trajectories when the difference between fwd and bwd derivatives is greater than tol
        - truncates trajectories by removing 5 points around each suspected changepoint (including it) / this data is removed from the trajectory
        - returns Segments, a list of segment indices per trajectory
          Segments = [segments_1, ..., segments_k] where k is the number of trajectories
          segments_i contains the indices of *changepoints* in the *now truncated* trajectory i
        '''
        if m > 2:
            return self._segment_windowed(tol, m)
        self.og_x_list = deepcopy(self.x_list)
        self.og_t_list = deepcopy(self.t_list)
        Segments = []
        for j in range(len(self.x_list)):
            x_points = self.x_list[j]
            t_points = self.t_list[j]

            # Assert that the time points and x points arrays have the same length
            l = t_points.shape[0]
            assert l == x_points.shape[0], 't_points and x_points do not have the same length'
            
            segments = []
            current_start_index = 0
            
            i = 2
            while i <= l-3:
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
                    current_start_index = i-2
                    x_points = np.delete(x_points, np.s_[i-2:i+3], axis=0)
                    t_points = np.delete(t_points, np.s_[i-2:i+3], axis=0)
                    self.x_list[j] = x_points
                    self.t_list[j] = t_points
                    l = l - 5
                    if current_start_index > 0 and current_start_index < l:
                        segments.append(current_start_index)
                    
                i += 1
            Segments.append(segments)
        self.segmented = True
        return Segments

    def _segment_windowed(self, tol, m):
        '''
        Segmentation with the jump estimator on a window of 2m+1 samples (`segmentation.kink_statistic`): the
        least-squares estimate of a derivative jump at each sample, the smooth part of the trajectory being a
        cubic over the window. m = 2 is the statistic of the thesis; larger windows average measurement noise
        and respond fully to a jump wherever it falls between two samples.

        A change point is declared where the statistic first exceeds `tol`; it is placed at the maximum of the
        statistic over the following m samples (the statistic rises as the window starts to cover the switch).
        max(2, (m+1)//2) samples are removed on each side of it, and the search resumes after the removed
        samples, with windows that do not cover them. Same outputs and side effects as `_segment_in_place`.
        '''
        from .segmentation import kink_statistic
        self.og_x_list = deepcopy(self.x_list)
        self.og_t_list = deepcopy(self.t_list)
        r = max(2, (m + 1) // 2)
        Segments = []
        for j in range(len(self.x_list)):
            x_points, t_points = self.x_list[j], self.t_list[j]
            N = len(t_points)
            h = t_points[1] - t_points[0]
            keep = np.ones(N, dtype=bool)
            cuts = []           # first kept original index after each change point
            start = 0
            while True:
                piece = x_points[start:]
                if len(piece) < 2 * m + 1:
                    break
                stat = kink_statistic(piece, h, m)
                above = np.nonzero(stat > tol)[0]
                if len(above) == 0:
                    break
                i = above[0]
                i = i + int(np.argmax(stat[i:i + m + 1]))
                lo, hi = start + i - r, start + i + r + 1
                keep[max(lo, 0):min(hi, N)] = False
                if hi >= N:
                    break
                cuts.append(hi)
                start = hi
            new_index = np.cumsum(keep) - 1
            segs = sorted({int(new_index[c]) for c in cuts if keep[c] and 0 < new_index[c] < keep.sum()})
            self.x_list[j] = x_points[keep]
            self.t_list[j] = t_points[keep]
            Segments.append(segs)
        self.segmented = True
        return Segments

    def is_empty(self):
        return self.num_trajectories == 0


    @classmethod
    def from_path(cls, path):
        """
        Create a Trajectories object by loading CSV files from a directory or a tar archive.

        Each CSV file is assumed to represent a single trajectory, where:
        - The first column is time (1D, shape: [N])
        - The remaining columns are state variables (2D, shape: [N, D])

        Parameters
        ----------
        path : str
            Path to either a directory containing CSV files, or a `.tar` archive.

        Returns
        -------
        Trajectories
            A new Trajectories object with one entry per CSV file.
            `t_list[i]` is a 1D numpy array of time values.
            `x_list[i]` is a 2D numpy array of shape (N, D) for state values.

        Raises
        ------
        ValueError
            If a CSV file contains fewer than 2 columns.
        IOError
            If the path is invalid or extraction fails.
        """
        def load_csvs_from_dir(directory):
            t_list = []
            x_list = []
            for filename in sorted(os.listdir(directory)):
                if filename.endswith(".csv"):
                    filepath = os.path.join(directory, filename)
                    data = np.loadtxt(filepath, delimiter=",")
                    if data.ndim == 1 or data.shape[1] < 2:
                        raise ValueError(f"Invalid CSV format in {filename}: must have at least two columns")
                    t = data[:, 0]
                    x = data[:, 1:]
                    t_list.append(t)
                    x_list.append(x)
            return t_list, x_list

        if os.path.isdir(path):
            t_list, x_list = load_csvs_from_dir(path)
            return cls(t_list, x_list)

        elif os.path.isfile(path) and path.endswith(".tar"):
            with tempfile.TemporaryDirectory() as tmpdir:
                with tarfile.open(path, 'r') as tar:
                    tar.extractall(path=tmpdir)
                t_list, x_list = load_csvs_from_dir(tmpdir)
                return cls(t_list, x_list)

        else:
            raise IOError(f"Invalid input path: must be a directory or .tar file — got: {path}")

    def to_tarfile(self, tar_path, prefix="traj"):
        """
        Save this Trajectories object as a .tar archive of CSV files.

        Parameters
        ----------
        tar_path : str
            Path to the output .tar file.
        prefix : str, optional
            Prefix for the CSV filenames inside the archive (default is 'traj').
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (t, x) in enumerate(zip(self.t_list, self.x_list)):
                assert t.ndim == 1 and x.ndim == 2, f"Invalid shapes at index {i}: t={t.shape}, x={x.shape}"
                assert len(t) == x.shape[0], f"Mismatched time/state lengths at index {i}"
                data = np.column_stack((t, x))
                filename = f"{prefix}_{i}.csv"
                filepath = os.path.join(tmpdir, filename)
                np.savetxt(filepath, data, delimiter=",")

            with tarfile.open(tar_path, "w") as tar:
                for filename in sorted(os.listdir(tmpdir)):
                    fullpath = os.path.join(tmpdir, filename)
                    tar.add(fullpath, arcname=filename)

    @classmethod
    def from_csv(cls, paths: Union[str, Sequence[str]], time_column: Union[int, str, None] = 0,
                 state_columns: Optional[Sequence[Union[int, str]]] = None, delimiter: str = "auto",
                 skip_header: Union[str, bool] = "auto", dt: Optional[float] = None,
                 trajectory_column: Union[int, str, None] = None) -> "Trajectories":
        """
        Loads trajectories from text files: one file per trajectory, or one file with a column of trajectory
        identifiers (`trajectory_column`).

        Parameters
        ----------
        paths : str or list of str
            A file, a glob pattern ("runs/*.csv") or a list of them; files are read in sorted order.
        time_column : int, str or None, default 0
            Index or header name of the time column. None (or `dt` given): no time column.
        state_columns : list of int or str, default None
            Indices or header names of the state variables (default: every other column).
        delimiter : str or None, default "auto"
            "auto": "," for .csv files, whitespace otherwise; None: whitespace.
        skip_header : "auto", int or bool, default "auto"
            "auto": the first line is skipped (and gives the variable names) when it is not numeric.
        dt : float, default None
            Time step for files without a time column: t = dt * arange(N).
        trajectory_column : int or str, default None
            Column whose value identifies the trajectory of each row (one file, several trajectories).

        The result has an attribute `names` (names of the state variables; "x1", "x2", ... without header).

        Example: Trajectories.from_csv("runs/*.csv", time_column="t")
        """
        import glob as _glob
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        files = []
        for p in paths:
            hits = sorted(_glob.glob(str(p)))
            if not hits and not os.path.exists(str(p)):
                raise FileNotFoundError(f"no file matches {p}")
            files += hits or [str(p)]
        if dt is not None:
            time_column = None
        t_list, x_list, names = [], [], None
        for f in files:
            delim = ("," if f.lower().endswith(".csv") else None) if delimiter == "auto" else delimiter
            with open(f) as fh:
                first = fh.readline()
            header = None
            try:
                [float(v) for v in (first.split(delim) if delim else first.split()) if v.strip()]
                has_header = False
            except ValueError:
                has_header = True
            skip = int(has_header) if skip_header == "auto" else int(skip_header)
            if skip and has_header:
                header = [h.strip() for h in (first.split(delim) if delim else first.split())]
            data = np.loadtxt(f, delimiter=delim, skiprows=skip, ndmin=2)

            def col(c):
                if c is None or isinstance(c, (int, np.integer)):
                    return c
                if header is None or c not in header:
                    raise ValueError(f"{f}: no column named {c!r} (header: {header})")
                return header.index(c)
            tc, trc = col(time_column), col(trajectory_column)
            sc = [col(c) for c in state_columns] if state_columns is not None else \
                [i for i in range(data.shape[1]) if i not in (tc, trc)]
            if not sc:
                raise ValueError(f"{f}: no state column")
            groups = [np.arange(len(data))] if trc is None else \
                [np.nonzero(data[:, trc] == v)[0] for v in np.unique(data[:, trc])]
            for g in groups:
                x = data[g][:, sc]
                t = data[g][:, tc] if tc is not None else dt * np.arange(len(g)) if dt is not None else None
                if t is None:
                    raise ValueError(f"{f}: give a time column or dt")
                t_list.append(np.asarray(t, dtype=float))
                x_list.append(np.asarray(x, dtype=float))
            fnames = [header[i] for i in sc] if header else None
            if names is None:
                names = fnames
        if not x_list:
            raise ValueError("no trajectory read")
        out = cls(t_list, x_list)
        out.names = names or [f"x{i + 1}" for i in range(x_list[0].shape[1])]
        return out

    def to_csv(self, directory: str, prefix: str = "traj", names: Optional[Sequence[str]] = None) -> List[str]:
        """Writes one CSV file per trajectory (header: t and the variable names) into `directory`; returns the
        paths."""
        os.makedirs(directory, exist_ok=True)
        names = names or getattr(self, "names", None) or [f"x{i + 1}" for i in range(self.x_list[0].shape[1])]
        paths = []
        for i, (t, x) in enumerate(zip(self.t_list, self.x_list)):
            p = os.path.join(directory, f"{prefix}_{i + 1}.csv")
            np.savetxt(p, np.c_[t, x], delimiter=",", header=",".join(["t"] + list(names)), comments="", fmt="%.17g")
            paths.append(p)
        return paths

    def check_uniform_step(self, rtol=1e-6):
        """The common time step of all trajectories; raises ValueError if a step differs (relative tolerance
        rtol) or a trajectory has fewer than 5 samples."""
        from .learning.pipeline import check_trajectories
        return check_trajectories(self, rtol)
