import matplotlib.transforms as mtransforms
import matplotlib.scale as mscale
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib

from .util import get_percentile_limits

from pathlib import Path
import numpy as np
import re

__all__ = ['format_vibr_freq_plot',
           'format_ampl_hist_plot',
           'vibration_plots',
           'remove_duplicate_legend_labels',
           'format_sqrtscaled_ax',
           'register_squarerootscale',
          ]


def format_vibr_freq_plot(ax, labelsize=12, ylim=(0.05,10), 
                          xlim=(1,2e3), title=""):
    ax.set_xlabel("Frequency [Hz]", fontsize=labelsize)
    ax.set_ylabel("P2P amplitude [nm]", fontsize=labelsize)
    ax.set_yscale('squareroot')
    ax.set_xscale('squareroot')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if xlim[1] == 2e3: 
        ticks = [1, 5, 14, 28, 50, 100, 300, 500, 700, 
                 900, 1100, 1300, 1500, 1700, 1900]
        format_sqrtscaled_ax(ax, set_ticks=ticks)
    ax.grid(b=True, which='both', ls="--")
    ax.set_title(title)
    ax.legend()


def format_ampl_hist_plot(ax, labelsize=12, ylim=(None, None),
                          xlim=(None, None), title="",
                          hline=None, ticksize=14):
    ax.set_xscale('squareroot')
    ticks = ax.get_xticks()
    ax.set_xlim(0, ticks[-1])
    ax.set_ylim(ylim)
    ax.set_xlabel("\nCounts [-]", fontsize=labelsize)
    ax.set_ylabel("Displacement [nm]", fontsize=labelsize)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    if type(hline) == tuple:
        _, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        l1, l2 = hline
        ax.axhline(l1, color='b', ls="--", lw=1)
        ax.axhline(l2, color='b', ls="--", lw=1)
        width = l2 - l1
        ax.text(xmax/10, l2+0.1*ymax, r"${:.1f} $ nm".format(width), 
                fontsize=labelsize, color='b')
        ax.annotate("", xy=(0.9*xmax, l2), xytext=(0.9*xmax, l1),
                    arrowprops=dict(arrowstyle="<->", color='blue'))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right() 
    ax.set_title(title, fontsize=labelsize)   


def vibration_plots(dfs, directions=('x', 'y'), dpi=150, save=False,
                            plot_mean=True, plot_median=True, ylim=None):
    plt.style.use('seaborn-colorblind')
    if not type(dfs) == list: dfs = [dfs]
    for df in dfs:
        for d, direction in zip(['x', 'y'], directions):
            if d in list(df.columns.levels[0]): 
                fig = plt.figure(figsize=(14,4), dpi=dpi)
                axs = (plt.subplot2grid((1,7), (0,0), colspan=6), 
                       plt.subplot2grid((1,7), (0,6)))
                axs[0].plot(df[d].xs("Frequency [Hz]", level=1, axis=1), 
                            df[d].xs("P2P amplitude [nm]", level=1, axis=1), 
                            c='k', alpha=0.02)
                if plot_mean: axs[0].plot(df[d, 'Average', "Frequency [Hz]"], 
                                          df[d, 'Average', "P2P amplitude [nm]"], 
                                          '-r', lw=1, label="Mean")
                if plot_median: axs[0].plot(df[d, 'Median', "Frequency [Hz]"], 
                                            df[d, 'Median', "P2P amplitude [nm]"], 
                                            '-b', lw=0.5, label="Median")
                if type(ylim) == tuple: 
                    format_vibr_freq_plot(axs[0], title=direction, ylim=ylim)
                else: 
                    format_vibr_freq_plot(axs[0], title=direction)
                
                all_data = df[d].xs("Displacement [nm]", level=1, axis=1).values
                all_data = all_data[~np.isnan(all_data)]
                n, bins, patches = axs[1].hist(all_data, bins='auto', color='k', 
                                               orientation=u'horizontal')
                l1, l2 = get_percentile_limits(n, bins, 0.99)
                ymax = np.max(np.abs(all_data))
                format_ampl_hist_plot(axs[1], hline=(l1, l2), 
                                           ylim=(-1*ymax, ymax))
                plt.tight_layout()
                if save:
                    for label in list(df.columns.levels[1]):
                        if "*.tif" in label: break
                    dir_name = Path(label).parents[0]
                    # avoid figure being picked up when loading data
                    fig.savefig(dir_name / (direction + ".jpg"))
                    plt.close('all')


def vibration_plot_overview(dfs, directions=['x', 'y'], label=None):
    plt.style.use('seaborn-colorblind')
    if not type(dfs) == list: dfs = [dfs]
    for d, direction in zip(['x', 'y'], directions):
        if label == 'fromdir':
            fig = plt.figure(figsize=(14,4), dpi=150)
            axs = (plt.subplot2grid((1,7), (0,0), colspan=6), 
                   plt.subplot2grid((1,7), (0,6)))
        else:
            fig, ax = plt.subplots(figsize=(14,4), dpi=150)
            axs = [ax]
        for df in dfs:
            if d in list(df.columns.levels[0]): 
                if label == None: use_label=""
                elif label == 'fromdir':
                    for use_label in list(df.columns.levels[1]):
                        if "*.tif" in use_label: break
                    use_label = Path(use_label).parents[0].parts[-1]
                    cooler_T = float(use_label.split(" K")[0])
                    all_data = df[d].xs("Displacement [nm]", level=1, axis=1).values
                    all_data = all_data[~np.isnan(all_data)]
                    n, bins = np.histogram(all_data, bins='auto')
                    l1, l2 = get_percentile_limits(n, bins, 0.99)
                    axs[1].plot(cooler_T, l2-l1, 'ok')
                axs[0].plot(df[d, 'Average', "Frequency [Hz]"], 
                         df[d, 'Average', "P2P amplitude [nm]"], 
                         lw=1, label=use_label)
        format_vibr_freq_plot(axs[0], title=direction)
        axs[0].set_yscale('squareroot')
        axs[0].set_ylim((0,5))
        if label == 'fromdir':
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right() 
            axs[1].set_ylabel("Displacement [nm]")
            axs[1].set_xlabel("Cooler temperature [K]")
            axs[1].grid(b=True, which='both', ls="--")
            axs[1].set_ylim(0, 30)



def remove_duplicate_legend_labels(ax, alpha=0):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys())
    if alpha > 0: 
        for lh in leg.legendHandles: lh.set_alpha(alpha)
    

def format_sqrtscaled_ax(ax, set_ticks=[], label_rotation=30):
    if set_ticks:
        locs = [float(xt) for xt in set_ticks]
        labels = [matplotlib.text.Text(xt, 0, str(xt)) for xt in set_ticks]
        ax.set_xticks(locs)
        ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', rotation=label_rotation)


def register_squarerootscale():
    """Registers squareroot scale for matplotlib
    
    References
    ----------
    [1] https://stackoverflow.com/a/39662359/5285918
    """
    mscale.register_scale(SquareRootScale)
   
   
class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    
    References
    ----------
    [1] https://stackoverflow.com/questions/42277989/square-root-scale-using-matplotlib-python
    """
 
    name = 'squareroot'
 
    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)
 
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())
 
    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax
 
    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True
 
        def transform_non_affine(self, a): 
            return np.array(a)**0.5
 
        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()
 
    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True
 
        def transform(self, a):
            return np.array(a)**2
 
        def inverted(self):
            return SquareRootScale.SquareRootTransform()
 
    def get_transform(self):
        return self.SquareRootTransform()