"""Optional outputs of a learning run (files and figures). A Reporter with everything off does nothing."""
import os

STAGE_SUFFIX = {"passive": "", "refined": "_gen", "final": "_EQ"}


class Reporter:
    """Writes a text log (experi-info/<name>/<name>.txt) and the equations in LaTeX
    (experi-info/latex-files/<name>-eqs.tex) when `write_on`, and draws trajectory figures when `with_plots`
    (saved to experi-plots/ when `write_on`, at `dpi`)."""

    def __init__(self, name="run", write_on=False, with_plots=False, plot_xt=False, dpi=600, save_ext=".png"):
        self.name, self.write_on, self.with_plots, self.plot_xt = name, write_on, with_plots, plot_xt
        self.dpi, self.save_ext = dpi, save_ext
        self._file = None
        self.dir_plots = "experi-plots/"
        if write_on:
            d = "experi-info/" + name + "/"
            for directory in (self.dir_plots, d, "experi-info/latex-files/"):
                os.makedirs(directory, exist_ok=True)
            self._file = open(d + name + ".txt", "w")

    def log(self, text):
        if self._file is not None:
            self._file.write(text)

    def _plotting_function(self, D):
        from ..plotting import draw_trajectories_2D, draw_trajectories_3D, draw_trajectories_ND
        return draw_trajectories_2D if D == 2 else (draw_trajectories_3D if D == 3 else draw_trajectories_ND)

    def stage_figures(self, stage, model, train_ref, test_ref, learned):
        """Training and test trajectories of the system and of the learned model after a stage."""
        if not self.with_plots or learned[0] is None:
            return
        trajs_train_learned, trajs_test_learned = learned
        f = self._plotting_function(train_ref.d if hasattr(train_ref, 'd') else len(train_ref.x_list[0][0]))
        ftrain, atrain = f(train_ref, colors=['b'], label='From SUL', mark_beginning=True, plot_xt=self.plot_xt)
        ftrain, atrain = f(trajs_train_learned, fig=ftrain, axs=atrain, colors=['m'], label='From learned system',
                           mark_beginning=True, plot_xt=self.plot_xt, labelfun=model.labelfun_)
        ftest = None
        if test_ref is not None:
            ftest, atest = f(test_ref, colors=['b'], label='From SUL', mark_beginning=True, plot_xt=self.plot_xt)
            ftest, atest = f(trajs_test_learned, fig=ftest, axs=atest, colors=['r'], label='From learned system',
                             mark_beginning=True, plot_xt=self.plot_xt, labelfun=model.labelfun_)
        if self.write_on:
            sfx = STAGE_SUFFIX[stage]
            ftrain.savefig(self.dir_plots + self.name + "_training" + sfx + self.save_ext, dpi=self.dpi)
            if ftest is not None:
                ftest.savefig(self.dir_plots + self.name + "_test" + sfx + self.save_ext, dpi=self.dpi)

    def query_figure(self, count, model, sul_trajs, cexs_learned, conforming_trajs_learned):
        """Trajectories of an equivalence query: system (black), counterexamples (red), conforming (green)."""
        if not self.with_plots:
            return
        f = self._plotting_function(len(sul_trajs.x_list[0][0]))
        f3, a3 = f(sul_trajs, colors=['k'], label='From SUL', mark_beginning=True, plot_xt=self.plot_xt, labelfun=None)
        f3, a3 = f(cexs_learned, fig=f3, axs=a3, colors=['r'], label='From learned system / CEX', mark_beginning=True,
                   plot_xt=self.plot_xt, labelfun=None)
        f3, a3 = f(conforming_trajs_learned, fig=f3, axs=a3, colors=['g'], label='From learned system',
                   mark_beginning=True, plot_xt=self.plot_xt, labelfun=model.labelfun_)
        if self.write_on:
            f3.savefig(self.dir_plots + self.name + f"_EQ_{count}" + self.save_ext, dpi=self.dpi)

    def finish(self, result):
        """Writes the errors and the equations, and closes the log."""
        if not self.write_on:
            return
        for stage in ("passive", "refined", "final"):
            e = result.errors.get(stage, {})
            for k, v in e.items():
                self.log(f"{k} RD after stage '{stage}' = {v}\n")
        with open("experi-info/latex-files/" + self.name + "-eqs.tex", "w") as f3:
            f3.write(result.model.gen_full_latex_eq())
        self._file.close()
        self._file = None
