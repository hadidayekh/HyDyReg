import numpy as np
from scipy.integrate import solve_ivp
from .differentiation import diff_method_fps
from .trajectories import Trajectories
from . import _logging as _log
# Cumulative simulation statistics of the process (see SNDS.simulate_trajectory); diagnostics only.
SIM_TOTALS = {'trajectories': 0, 'crossings': 0, 'sliding_segments': 0, 'sliding_time': 0.0, 'grazing': 0, 'forced': 0}


class SlidingMotionException(Exception):
    pass

class UnstableTrajException(Exception):
    pass
class SNDS:
    '''
    A state-dependant Switched Nonlinear Dynamical System.
    
    Attributes
    ----------
    n : int
        The number of modes.
    
    d : int
        The dimension of the continuous state vector
    
    eqs : List[List[Callable]]
        List of `n` lists of `d` differential equations, one per mode and dimension of the continuous state vector.
    
    eventlist : List[Callable]
        List of events to track when simulating trajectory, this is usually the conditions of crossing from one mode to another.
        This is usually a list of the boundary evaluations B_ij(x) that define the boundaries between modes i and j.
        The evaluation is tracked and the simulation could switch when the sign changes so that the boundary crossing point (if done) is searched (zero-crossing value).
    
    labelfun : Callable
        A function that returns the mode in which state x is in, from {0,...,n-1}

    
    Methods
    -------
    simulate_trajectory(step, T, x0s, t0=0, stop_at_sliding_motion=False, compare_with_traj=None, compare_tol=0.5, max_step_factor=10)
        Simulate trajectory given initial conditions.

    generate_points_between_segments(trajectories, segments, seg_tol, timestep='default')
        Given trajectories as well as change points indicating segments, generate points between consecutive segments of each trajectory.

    get_derivatives(trajectories)
        Given trajectories of the SNDS, return the corresponding exact derivatives of each point (using equations and labelfun of the SDNS).
    '''

    def __init__(self, n, d, eqs, eventlist, labelfun):
        assert (n == len(eqs))
        self.number_of_modes = n
        self.dim = d
        self.eqs = eqs
        self.labelfun = labelfun
        self.eventlist = eventlist
        # optional: function x -> array of n mode scores (labelfun(x) = argmax); enables exact mode events
        self.mode_scores = None

    def simulate_trajectory(self, step, T, x0s, t0=0, stop_at_sliding_motion=False, compare_with_traj=None, compare_tol=0.5, max_step_factor=10):
        '''
        Simulates trajectories of the SNDS, with Filippov sliding motion on switching surfaces.

        Each mode is integrated with `solve_ivp` (RK45, rtol=1e-7, atol=1e-9) until it leaves its region:
        - if `self.mode_scores` is set (learned models), mode k is left when another mode's score
          overtakes it, i.e. when `s_k(x) - max_{j != k} s_j(x)` decreases through 0;
        - otherwise, when one of the functions in `self.eventlist` changes sign (each one is oriented by the
          side of the surface the trajectory is on, so only crossings away from that side are events).

        At a crossing from mode k towards mode j through a surface g = 0 (g > 0 on the side of k), with
        a_k = grad(g).f_k and a_j = grad(g).f_j:
        - a_j < 0: transversal crossing, integration continues in mode j from the crossing point;
        - a_k < 0 < a_j: attractive sliding; the Filippov vector field
          f_s = (a_j f_k - a_k f_j) / (a_j - a_k) (convex combination with grad(g).f_s = 0) is integrated
          until a_k rises through 0 (exit into k), a_j falls through 0 (exit into j), or another surface
          / mode is reached;
        - otherwise (a_k >= 0, grazing): integration continues in mode k.
        Integration always restarts from the exact event time and state.

        Parameters
        ----------

        `step` : `float`
            Time step.

        `T` : `float`
            Time range.

        `x0s` : list of `ndarray (D)`
            List of initial data points.

        `t0` : `float`, default=`0`
            Starting time.

        `stop_at_sliding_motion` : `bool`, default=`False`
            Raises a `SlidingMotionException` when a trajectory enters sliding motion (before the end of the time range).

        `compare_with_traj`, `compare_tol` :
            No longer supported (must be `None`).

        `max_step_factor` : `float`, default=`10`
            Inidcates the factor by which to divide `step` to get the simulation maximum time step. The simulation maximum time step is hence `step/max_step_factor`.
        '''
        if compare_with_traj is not None:
            raise NotImplementedError("compare_with_traj is no longer supported")
        num = round(T/step + 1)
        ivp_kw = dict(method='RK45', rtol=1e-7, atol=1e-9, max_step=step/max_step_factor, dense_output=True)
        peek = step/1000  # look-ahead time used to identify the next mode (systems without mode_scores)
        self.sim_stats_ = {'crossings': 0, 'sliding_segments': 0, 'sliding_time': 0.0, 'grazing': 0, 'forced': 0}

        t_list = []
        x_list = []
        for x0 in x0s:
            x0 = np.asarray(x0, dtype=float)
            t_points = np.linspace(t0, T+t0, num)
            t_end = T + t0 + 1.1*step
            X_points = np.empty((num, len(x0)))
            n_done = 0  # number of t_points already evaluated

            label = self.labelfun(x0)
            sides = self._initial_sides(x0)
            t_start = t0
            n_stuck = 0
            while True:
                events = self._mode_events(label, sides)
                res = solve_ivp(self.eqs[label], (t_start, t_end), x0, events=events, **ivp_kw)
                if res.status == -1:
                    _log.debug("STATUS -1")
                    raise SlidingMotionException
                t_stop = res.t[-1]
                n_done = self._fill(res.sol, t_points, X_points, n_done, t_stop)
                if res.status == 0 or n_done == num:
                    break
                x_ev = res.y[:, -1]
                fired = self._fired(res, events)
                # guard against Zeno-like repeated events without progress
                n_stuck = n_stuck + 1 if t_stop - t_start <= 1e-12 * max(1.0, abs(t_stop)) else 0
                if n_stuck > 100:
                    self.sim_stats_['forced'] += 1
                    res = solve_ivp(self.eqs[label], (t_stop, min(t_stop + step/max_step_factor, t_end)), x_ev, **ivp_kw)
                    n_done = self._fill(res.sol, t_points, X_points, n_done, res.t[-1])
                    t_start, x0 = res.t[-1], res.y[:, -1]
                    label = self.labelfun(x0)
                    sides = self._initial_sides(x0)
                    n_stuck = 0
                    continue

                g, j = self._switching_surface(label, sides, fired, t_stop, x_ev, peek)
                if j == label:  # surface crossed without mode change (only possible with eventlist)
                    sides[fired] = -sides[fired]
                    t_start, x0 = t_stop, x_ev
                    continue
                gk = self._normal_speed(g, label, t_stop, x_ev)
                gj = self._normal_speed(g, j, t_stop, x_ev)
                if gj < 0:  # transversal crossing
                    self.sim_stats_['crossings'] += 1
                    if self.mode_scores is None:
                        sides[fired] = -sides[fired]
                    label = j
                    t_start, x0 = t_stop, x_ev
                    continue
                if gk >= 0:  # grazing: stays in mode `label`
                    self.sim_stats_['grazing'] += 1
                    t_start, x0 = t_stop, x_ev
                    continue

                # attractive sliding between `label` and j along g = 0
                if stop_at_sliding_motion:
                    raise SlidingMotionException
                self.sim_stats_['sliding_segments'] += 1
                k = label
                fs, sl_events, kinds = self._sliding_system(g, k, j, sides, fired)
                try:
                    res = solve_ivp(fs, (t_stop, t_end), x_ev, events=sl_events, **ivp_kw)
                except ValueError:
                    # degenerate sliding segment (the solver made steps of zero length): as for repeated
                    # events without progress, integrate one small step in mode `label` and continue
                    self.sim_stats_['forced'] += 1
                    res = solve_ivp(self.eqs[label], (t_stop, min(t_stop + step/max_step_factor, t_end)), x_ev, **ivp_kw)
                    n_done = self._fill(res.sol, t_points, X_points, n_done, res.t[-1])
                    t_start, x0 = res.t[-1], res.y[:, -1]
                    if n_done == num or t_start >= t_end:
                        break
                    label = self.labelfun(x0)
                    sides = self._initial_sides(x0)
                    n_stuck = 0
                    continue
                if res.status == -1:
                    _log.debug("STATUS -1")
                    raise SlidingMotionException
                self.sim_stats_['sliding_time'] += res.t[-1] - t_stop
                t_start, x0 = res.t[-1], res.y[:, -1]
                n_done = self._fill(res.sol, t_points, X_points, n_done, t_start)
                if res.status == 0 or n_done == num:
                    break
                kind = kinds[self._fired(res, sl_events)]
                if kind == 'exit_k':
                    label = k
                elif kind == 'exit_j':
                    label = j
                    if self.mode_scores is None:
                        sides[fired] = -sides[fired]
                else:  # another surface or mode reached: decide from the state just ahead
                    x_ahead = x0 + peek * np.asarray(fs(t_start, x0))
                    label = self.labelfun(x_ahead)
                    if self.mode_scores is None:
                        sides = self._initial_sides(x_ahead, sides)
            t_list.append(np.linspace(t0, t0+T, num))
            x_list.append(X_points)
        for key, v in self.sim_stats_.items():
            SIM_TOTALS[key] += v
        SIM_TOTALS['trajectories'] += len(x0s)
        trajs = Trajectories(t_list, x_list)
        trajs.truncated = False
        return trajs

    # ---------------------------------------------------------------- simulation helpers

    def _eventlist(self):
        if self.eventlist is None:
            return []
        return list(self.eventlist) if isinstance(self.eventlist, (list, tuple)) else [self.eventlist]

    def _initial_sides(self, x, previous=None):
        '''Side (+1/-1) of each surface of `eventlist` that x is on (keeps `previous` where g(x) == 0).'''
        sides = []
        for i, ev in enumerate(self._eventlist()):
            v = ev(0, x)
            if v == 0:
                sides.append(previous[i] if previous is not None else 1.0)
            else:
                sides.append(1.0 if v > 0 else -1.0)
        return np.array(sides)

    def _mode_events(self, label, sides):
        '''Terminal events that fire when the trajectory leaves the region of `label`.'''
        if self.mode_scores is not None:
            scores = self.mode_scores

            def leave(t, x, k=label):
                s = scores(x)
                return s[k] - np.max(np.delete(s, k))
            leave.terminal, leave.direction = True, -1
            return [leave]
        events = []
        for i, ev in enumerate(self._eventlist()):
            def oriented(t, x, ev=ev, s=sides[i]):
                return s * ev(t, x)
            oriented.terminal, oriented.direction = True, -1
            events.append(oriented)
        return events

    @staticmethod
    def _fired(res, events):
        '''Index of the event that terminated `res` (the earliest one if several).'''
        times = [(te[0], i) for i, te in enumerate(res.t_events) if len(te) > 0]
        return min(times)[1]

    def _fill(self, sol, t_points, X_points, n_done, t_stop):
        '''Evaluate the dense solution at the not-yet-filled sample times <= t_stop.'''
        n_new = n_done + np.searchsorted(t_points[n_done:], t_stop, side='right')
        if n_new > n_done:
            X_points[n_done:n_new] = sol(t_points[n_done:n_new]).T
        return n_new

    def _switching_surface(self, k, sides, fired, t, x, peek):
        '''Surface function g (g > 0 on the side of mode k) and the mode j on the other side.'''
        if self.mode_scores is not None:
            scores = self.mode_scores
            s = scores(x)
            s_others = s.copy()
            s_others[k] = -np.inf
            j = int(np.argmax(s_others))
            return (lambda x, k=k, j=j: (lambda s: s[k] - s[j])(scores(x))), j
        ev, side = self._eventlist()[fired], sides[fired]
        j = self.labelfun(x + peek * np.asarray(self.eqs[k](t, x)))
        return (lambda x, ev=ev, side=side: side * ev(t, x)), j

    @staticmethod
    def _grad(g, x):
        x = np.asarray(x, dtype=float)
        grad = np.empty_like(x)
        for i in range(len(x)):
            h = 1e-6 * max(1.0, abs(x[i]))
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            grad[i] = (g(xp) - g(xm)) / (2*h)
        return grad

    def _normal_speed(self, g, mode, t, x):
        return float(np.dot(self._grad(g, x), self.eqs[mode](t, x)))

    def _sliding_system(self, g, k, j, sides, fired):
        '''Filippov vector field on g = 0 between modes k and j, and its terminal exit events.'''
        fk, fj = self.eqs[k], self.eqs[j]

        def speeds(t, x):
            grad = self._grad(g, x)
            vk, vj = np.asarray(fk(t, x)), np.asarray(fj(t, x))
            return vk, vj, float(np.dot(grad, vk)), float(np.dot(grad, vj))

        def fs(t, x):
            vk, vj, ak, aj = speeds(t, x)
            if aj - ak <= 0:  # outside the sliding region (only at its border, handled by the events)
                return vk if ak >= 0 else vj
            return (aj * vk - ak * vj) / (aj - ak)

        def exit_k(t, x):
            return speeds(t, x)[2]
        exit_k.terminal, exit_k.direction = True, 1

        def exit_j(t, x):
            return speeds(t, x)[3]
        exit_j.terminal, exit_j.direction = True, -1

        events, kinds = [exit_k, exit_j], ['exit_k', 'exit_j']
        if self.mode_scores is not None:
            scores = self.mode_scores
            if self.number_of_modes > 2:
                def other_mode(t, x):
                    s = scores(x)
                    rest = np.delete(s, [k, j])
                    return (np.max(rest) if len(rest) else -np.inf) - max(s[k], s[j])
                other_mode.terminal, other_mode.direction = True, 1
                events.append(other_mode)
                kinds.append('other')
        else:
            for i, ev in enumerate(self._eventlist()):
                if i == fired:
                    continue

                def oriented(t, x, ev=ev, s=sides[i]):
                    return s * ev(t, x)
                oriented.terminal, oriented.direction = True, -1
                events.append(oriented)
                kinds.append('other')
        return fs, events, kinds

    def generate_points_between_segments(self, trajectories, segments, seg_tol, timestep='default'):
        '''
        Simulates the SNDS to generate points between consecutive segments with fine time steps, derivates the generated points and returns them with their derivatives.

        Parameters
        ----------
        `trajectories` : `Trajectories`
            The concerned trajectories in which points will be generated.

        `segments` : `list` of `list` of `int`
            A list of lists indicating change point indices of each trajectory.

        `seg_tol` : `float`
            The segmentation tolerance used to re-segment the finely generated points between consecutive segments.

        `timestep` : `float` or `'default'`, default=`default`
            The simulation time step.
            Defaults to the simulation time step of `trajectories` divided by 20.


        Returns
        -------
        `final_X, final_Xdot` : Points generated between consecutive segments as well as their estimated derivatives.
        *Note*: This is bad design as the function does the derivative estimation too.
        '''
        t_list = trajectories.t_list
        x_list = trajectories.x_list
        final_X, final_X_dot = np.array([]), np.array([])
        if timestep == 'default':
            timestep = (t_list[0][1] - t_list[0][0])/20
        first = True
        # for every trajectory
        for i in range(len(t_list)):
            x_points = x_list[i]
            t_points = t_list[i]
            seg_curr = segments[i]
            # if the trajectory contains only one segment, continue
            if len(seg_curr) == 0:
                continue
            # otherwise, for each sgement
            for r in range(len(seg_curr)):
                seg_index = seg_curr[r]
                if seg_index - 1 < 0:
                    continue
                try:
                    x0 = x_points[seg_index - 1]
                    t_start = t_points[seg_index - 1]
                except:
                    x0 = x_points[0]
                    t_start = t_points[0]
                try:
                    t_end = t_points[seg_index]
                except:
                    t_end = t_points[-1]
                T = t_end - t_start

                fine_trajs = self.simulate_trajectory(timestep, T, [x0], t_start)
                fine_trajs, segments_fine = fine_trajs.segment(seg_tol)

                X, X_dot, seg_res, _ = diff_method_fps(np.array(fine_trajs.t_list), np.array(fine_trajs.x_list), segments_fine)
                
                if first:
                    final_X = X
                    final_X_dot = X_dot
                    first = False
                else:
                    final_X = np.r_[final_X, X]
                    final_X_dot = np.r_[final_X_dot, X_dot]
        return final_X, final_X_dot

    def get_derivatives(self, trajectories):
        '''
        Given trajectories of the SNDS, return the corresponding exact derivatives of each point (using equations and labelfun of the SDNS).
        '''
        x_list = trajectories.x_list
        X_dot = np.ndarray((0,self.dim))
        for xpoints in x_list:
            X_dot_traj = np.ndarray((len(xpoints), self.dim))
            for i in range(len(xpoints)):
                x = xpoints[i]
                X_dot_traj[i] = self.eqs[self.labelfun(x)](0, x)
            X_dot = np.r_[X_dot, X_dot_traj]
        return X_dot

class SNDS_affine(SNDS):
    '''
    A state-dependant Switched Nonlinear Dynamical System with affine dynamics.
    
    Attributes
    ----------
    
    params : ndarray of shape `(n, d, d+1)`
        Affine parameters describing the system dynamics such that
        
        f_j(x)[0] = params[j][0][0] * x[0] + params[j][0][1] * x[1] + ... + params[j][0][d-1] * x[d-1] + params[j][0][d]
        .
        .
        f_j(x)[d] = params[j][d][0] * x[0] + params[j][d][1] * x[1] + ... + params[j][d][d-1] * x[d-1] + params[j][d][d]
    
    eventlist : List[Callable]
        List of events to track when simulating trajectory, this is usually the conditions of crossing from one mode to another.
    
    labelfun : Callable
        A function that returns the mode in which state x is in, from {0,...,n-1}
    '''
    def __init__(self,  params, eventlist, labelfun):
        self.params = params
        n = params.shape[0]
        d = params.shape[1]
        assert d+1 == params.shape[2], "Linear systems' parameters are d+1 per equation."
        eqs = []

        # we need one equation function per mode
        for i in range(n):
            # append the function to eqs which is a list of functions
            # cf. https://stackoverflow.com/questions/3431676/creating-functions-or-lambdas-in-a-loop-or-comprehension 
            eqs.append(SNDS_affine.make_eq(i, params))

        # call super class constructor
        super().__init__(n, d, eqs, eventlist, labelfun)

    @staticmethod
    def make_eq(i, params):
        '''Right-hand side of mode i: x -> A x + b with A = params[i][:, :d], b = params[i][:, d].'''
        d = params.shape[1]
        A = np.array(params[i][:, :d], dtype=float)
        b = np.array(params[i][:, d], dtype=float)

        def eq(t, x):
            return A @ np.asarray(x, dtype=float) + b
        return eq
    
    def gen_latex_cont_eq(self):
        n = self.params.shape[0]
        D = self.params.shape[1]
        s = ""
        for i in range(n):
            s += f"Mode {i+1}:\n"
            for d in range(D):
                s += f"\\xddim{{{d+1}}} = "
                for d2 in range(D):
                    s += f"{self.params[i][d][d2]} \\xdim{{{d2+1}}} + "
                s += f"{self.params[i][d][-1]}\n"
            s += "\n"
        return s
        