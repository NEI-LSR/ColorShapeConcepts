import random

from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
from typing import Tuple, Optional
import os
import seaborn as sns
# make sure text is saved in svgs as text, not path
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.serif'] = ['Times']
plt.rcParams['lines.linewidth'] = .75 * .4


# Spencer & Helen Standardized Plotting Scripts

def _compute_ticks(min_val, max_val, ax_pos=0.):
    scale = (max_val - min_val)
    spacing = float(scale / 4)
    # first some special cases
    dec = math.floor(-math.log10(spacing)) + 1
    if math.ceil(math.log10(scale)) == math.log10(scale):
        # is a power of 10, get extra decimal for quarters
        # spacing = 5 * 10 ** (-dec)
        dec = dec + 1
    else:
        spacing = round(spacing, dec)
    if dec == 0:
        spacing = int(spacing)
    tick_pos = np.round(np.arange(ax_pos, 1.05 * max_val, spacing), dec)
    tick_neg = np.round(np.arange(ax_pos, min_val, -spacing), dec)[1:]
    tick = np.concatenate([tick_neg, tick_pos])
    if dec == 0:
        tick = tick.astype(int)
    tick = list(tick)
    return tick


def _set_size(w,h, ax=None):
    """
        force axis to take set size.
        w, h: width, height in inches
    """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    return ax


def _set_fig_params(axs, fig, minyval=None, maxyval=None, xpos=0.):
    #minyval = min(minyval, -.075 * maxyval)
    scale = maxyval - minyval
    maxyval = maxyval + .05 * scale
   # minyval = minyval - 0 * scale
    axs.set_ylim((minyval, maxyval))
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_position(('data', xpos))
    axs.margins(.01)
    axs.tick_params(axis="both", length=2., pad=0.)
    axs.tick_params(axis='x', pad=1) #  + 25 * (xpos - minyval)
    fig.tight_layout()
    return axs, fig


def create_save_barplot(axs: plt.Axes, fig: plt.Figure, title: str, boot_data: np.ndarray, group_labels: Tuple[str],
                        xposition=0., out_dir="plots", data_spread="ci", y_axis_label=None, set_size=None, ylim=None,
                        rotate_x_labels=False, suppress_x_label=False):
    """
    expect data as 3d nparray of groups, within group, data
    """
    if data_spread not in ["ci", "full"]:
        raise ValueError
    colors = ["none", "#686868", "#B8B8B8"]
    edge_colors = ["black", "#686868", "#B8B8B8"]
    zorders = [3,2,2] # Prevent first group border from being covered by second group
    m_val = xposition
    min_val = xposition
    space = min(.1 * boot_data.shape[1], .2)
    global_min = []
    for i, r in enumerate(boot_data):
        w = (1 - space) / (len(r))
        for j, g in enumerate(r):
            x = w * (j + .5 * space) + i
            effect = g.mean(axis=-1)
            # want spacing of half bar between groups
            axs.bar(x=x, height=effect - xposition, bottom=xposition, width=w, color=colors[j], edgecolor=edge_colors[j], zorder=zorders[j], linewidth=.5)
            if y_axis_label:
                axs.set_ylabel(y_axis_label, fontsize=6)
            if data_spread == "ci":
                lower_quant = np.quantile(g, .025, axis=-1)
                upper_quant = np.quantile(g, .975, axis=-1)
                try:
                    axs.errorbar(x=x, y=effect, yerr=np.array([effect - lower_quant, upper_quant - effect])[:, None], color="black", linewidth=.5)
                    m_val = max(upper_quant, m_val)
                    min_val = min(lower_quant, min_val)
                except Exception:
                    m_val = m_val * 1.1
                    min_val = min_val
                global_min.append(min_val)
            elif data_spread == "full":
                axs.scatter([x]*len(g), g, color=colors[j], edgecolors= "black", alpha=.7)
                m_val = max(g)
                min_val = min(g)
                global_min.append(min_val)
            else:
                raise ValueError
    if ylim is not None:
        min_val = ylim[0]
        m_val = ylim[1]
    ytick = _compute_ticks(min_val, m_val, ax_pos=xposition)
    axs.set_yticks(ytick, labels=[str(yt) for yt in ytick], fontsize=6)
    axs, fig = _set_fig_params(axs, fig, min_val, m_val, xposition)
    print(min(global_min))
    if rotate_x_labels:
        rt = 45
    else:
        rt = 0
    if set_size is not None:
        _set_size(set_size[0], set_size[1], axs)
    axs.set_xticks(np.array(range(len(group_labels))) + .5 * ((len(r) - 1) * w + space - .01))
    axs.set_xticklabels(labels=group_labels, rotation=rt, fontsize=6, y=min(global_min)) # place labels below lowest error bar or data point
    fig.savefig(os.path.join(out_dir, title + ".svg"))
    return fig


def create_save_boxplot(
    axs: plt.Axes,
    fig: plt.Figure,
    title: str,
    boot_data: np.ndarray,
    group_labels: Tuple[str],
    xposition: float = 0.,
    out_dir: str = "plots",
    data_spread: str = "ci",
    y_axis_label: Optional[str] = None,
    set_size: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    rotate_x_labels: bool = False,
    suppress_x_label: bool = False,
    subgroup_labels: Optional[Tuple[str]] = None
):
    """
     whiskers are set to the 99% ci
    """
    if boot_data.ndim != 3:
        raise ValueError("`boot_data` must be a 3D numpy array.")
    if boot_data.shape[0] != len(group_labels):
        raise ValueError("Mismatch between number of groups in `boot_data` and `group_labels`.")

    user_provided_subgroups = subgroup_labels is not None
    num_subgroups = boot_data.shape[1]
    if subgroup_labels is None:
        subgroup_labels = tuple(f"Subgroup {i + 1}" for i in range(num_subgroups))
    elif len(subgroup_labels) != num_subgroups:
        raise ValueError("Mismatch between number of subgroups in `boot_data` and `subgroup_labels`.")

    data_list = []
    for i, group_data in enumerate(boot_data):
        for j, subgroup_data in enumerate(group_data):
            for value in subgroup_data:
                data_list.append({
                    'value': value,
                    'group': group_labels[i],
                    'subgroup': subgroup_labels[j]
                })
    df = pd.DataFrame(data_list)

    palette = ["lightgray", "gray"]
    sns.boxplot(
        data=df, x='group', y='value', hue='subgroup',
        ax=axs, palette=palette,
        whis=[1., 99.],  # Set whiskers to 99% CI
        fliersize=0.,      # Hide outliers beyond the CI whiskers
        linewidth=0.4,
        linecolor="black"
    )

    if y_axis_label:
        axs.set_ylabel(y_axis_label, fontsize=6)

    min_val, max_val = (df['value'].min(), df['value'].max())
    if ylim is not None:
        min_val, max_val = ylim

    ytick = _compute_ticks(min_val, max_val, ax_pos=xposition)
    axs.set_yticks(ytick, labels=[f"{yt}" for yt in ytick], fontsize=6)

    axs, fig = _set_fig_params(axs, fig, min_val, max_val, xposition)

    if set_size is not None:
        axs = _set_size(set_size[0], set_size[1], ax=axs)

    global_min = boot_data.min()
    rt = 45 if rotate_x_labels else 0
    axs.set_xticklabels(labels=group_labels, rotation=rt, fontsize=6, y=global_min)
    axs.set_xlabel('' if suppress_x_label else 'Groups')

    if not user_provided_subgroups and axs.get_legend() is not None:
        axs.get_legend().remove()
    elif axs.get_legend() is not None:
        handles, labels = axs.get_legend_handles_labels()
        axs.legend(handles, labels, title=None, frameon=False, fontsize=6)

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, title + ".svg")
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    return fig


def create_save_line_plot(axs: plt.Axes, fig: plt.Figure, title: str, boot_data: np.ndarray, group_labels: Tuple[str]=None,
                          x=None, ylim=None, xposition=0., yposition=0., out_dir="plots", set_size=None, save=True):
    """
    expect boot_data as 3d nparray of groups, time (xaxis), data
    """
    colors = ["tab:gray", "tab:blue", "tab:pink", "tab:green", "tab:orange"]
    boot_data = boot_data.squeeze()
    if boot_data.ndim > 2:
        effect = np.mean(boot_data, axis=2)
        low_quant = np.quantile(boot_data, .05, axis=2)
        up_quant = np.quantile(boot_data, .95, axis=2)
        max_val = np.max(up_quant)
        min_val = np.min(low_quant)
    else:
        effect = boot_data
        max_val = np.max(effect)
        min_val = np.min(effect)
        low_quant = None
        up_quant = None
    if x is None:
        x = np.arange(effect.shape[1])
    elif type(x) is not np.ndarray:
        x = np.array(x)

    if ylim is not None:
        min_val = ylim[0]
        max_val = ylim[1]
    spread = max_val - min_val
    # max_val = max_val + .05 * spread
    # min_val = min_val - .05 * spread
    min_x, max_x = np.min(x), np.max(x)
    axs, fig = _set_fig_params(axs, fig, min_val, max_val, xposition)
    for i in range(len(effect)):
        x_jit = x + random.choice([0, ((i + 1) % 3) * .005 * (max_x - min_x), -1 *  ((i - 1) % 3) * .005 * (max_x - min_x)])
        y_jit = effect[i] + random.choice([0, ((i - 1) % 3) * .005 * spread, -1 * ((i + 1) % 3) * .005 * spread])
        axs.plot(x_jit, y_jit, color=colors[i], linewidth=1.5, linestyle=(2 * i, (3, 3)))
        if low_quant is not None and up_quant is not None:
            axs.fill_between(x, low_quant[i], up_quant[i], alpha=.5, color=colors[:len(effect)])
    ytick = _compute_ticks(min_val, max_val, xposition)
    xtick = _compute_ticks(min_x, max_x, yposition)
    axs.set_yticks(ytick)
    axs.set_xticks(xtick)
    if set_size is not None:
        _set_size(set_size[0], set_size[1], axs)
    if save:
        fig.savefig(os.path.join(out_dir, title + ".svg"))
    return fig


def create_save_learning_plot(axs: plt.Axes, fig: plt.Figure, title: str, boot_data: np.ndarray, raw_data: np.ndarray,
                              group_labels: Tuple[str] = None, x=None, ylim=None, xposition=0., out_dir="plots"):
    """
    expect boot_data as 3d nparray of groups, time (xaxis), data
    expect raw_data as  3d nparray of groups, (x,y), data (need to supply exact x vals in case does not match boot data
    """
    colors = ["#D95319", "tab:gray"]
    boot_data = boot_data.squeeze()
    if boot_data.ndim > 2:
        effect = np.mean(boot_data, axis=2)
        low_quant = np.quantile(boot_data, .05, axis=2)
        up_quant = np.quantile(boot_data, .95, axis=2)
        max_val = np.max(up_quant)
        min_val = np.min(low_quant)
    else:
        effect = boot_data
        max_val = np.max(effect)
        min_val = np.min(effect)
        low_quant = None
        up_quant = None
    if x is None:
        x = np.arange(effect.shape[1])
    if ylim is not None:
        min_val = ylim[0]
        max_val = ylim[1]
    spread = max_val - min_val
    min_x, max_x = np.min(x), np.max(x)
    axs, fig = _set_fig_params(axs, fig, min_val, max_val, xposition)
    for i in range(len(effect)):
        axs.scatter(aw_data[i][0], raw_data[i][1], color=colors[i], s=10)
        axs.plot(x_jit, y_jit, color=colors[i], linewidth=1.5, linestyle=(2 * i, (3, 3)))
        if low_quant is not None and up_quant is not None:
            axs.fill_between(x, low_quant[i], up_quant[i], alpha=.5, color=colors[i])
    ytick = _compute_ticks(min_val, max_val, xposition)
    xtick = _compute_ticks(min_x, max_x, 0.)
    axs.set_yticks(ytick)
    axs.set_xticks(xtick)
    fig.savefig(os.path.join(out_dir, title + ".png"), dpi=300)
    return fig


def create_save_learning_curve(axs: plt.Axes, fig: plt.Figure, title: str, bin_data: np.ndarray, ci_data: np.ndarray,
                               fit_data: np.ndarray, year_data: np.ndarray, group_labels: Tuple[str] = None, x=None,
                               yceil=None, xposition=0., out_dir="plots", open_c=False, set_size=None):
    """
    expect bin_data as 2d nparray of (x,y), groups
    expect raw_data as  3d nparray of groups, (x,y), data (need to supply exact x vals in case does not match boot data
    """
    colors = ["#D95319", "tab:gray"]
    for i in range(bin_data.shape[1]):
        if open_c:
            axs.scatter(bin_data[0][i], bin_data[1][i], facecolor='none', edgecolor=colors[i], s=8, linewidth=.2,
                        rasterized=False)
        else:
            axs.scatter(bin_data[0][i], bin_data[1][i], facecolor=colors[i], edgecolor=colors[i], s=8, linewidth=.5,
                        rasterized=False)
        if ci_data is not None:
            axs.fill_between(ci_data[0][i], np.array(ci_data[1][i]).T[0], np.array(ci_data[1][i]).T[1], alpha=.2,
                             color=colors[i], rasterized=False)  # rasterize CIs else get svg rendering issues
        if fit_data is not None:
            axs.plot(fit_data[0][i], fit_data[1][i], color='black', linewidth=.2)
    if np.max(bin_data[0]) > 20000:
        xtick = list(range(0, int(np.max(bin_data[0][0])), 20000))
    elif np.max(bin_data[0]) < 10000:
        xtick = [0, 10000]
    else:
        xtick = list(range(0, int(np.max(bin_data[0][0])), 1000))

    axs, fig = _set_fig_params(axs, fig, 0.2, 1., .2)
    axs.margins(.05)
    if np.max(bin_data[0]) < 10000:
        axs.margins(.5)

    axs.tick_params(axis="both", length=2., pad=1)
    axs.tick_params(axis='x', pad=6)

    # ytick = _compute_ticks(min_val, max_val, xposition)
    # xtick = _compute_ticks(min_x, max_x, 0.)
    if yceil is not None:
        ytick = [.25, .5, yceil]
    else:
        ytick = [.25, .5, 1.]
    axs.set_yticks(ytick, labels=[str(yt) for yt in ytick], fontsize=6)
    axs.set_xticks(xtick)
    axs.set_xticklabels([int(xt / 10000) for xt in xtick], fontsize=6)

    if year_data is not None:
        yr_ax = axs.secondary_xaxis(location=0)
        yr_ax.set_xticks(year_data[0], year_data[1], fontsize=6)

    if set_size is not None:
        _set_size(set_size[0], set_size[1], axs)
    # if ci_data is not None:
    #    fig.savefig(os.path.join(out_dir, title + '.png'), dpi=300)
    fig.savefig(os.path.join(out_dir, title + '.svg'))
    fig.tight_layout()
    return fig


def create_save_scatter_plot(axs: plt.Axes, fig: plt.Figure, title: str, scatter_data: list, max_val: float,
                             min_val: float,
                             fit_lm=False, xposition=0., out_dir="plots", set_size=None):
    """
    expect scatter_data as list (len=groups) of 2d nparrays of x,y dim, number data points
    """
    dot_color = ["none", "none"]  # ["#A9A9A9", "#404040", "none", "none"]
    edge_color = ["black", "#686868"]  # ["#A9A9A9", "#404040", "#686868", "black"]
    for j, g in enumerate(scatter_data):
        axs.scatter(g[0], g[1], alpha=.55, color=dot_color[j], s=2, edgecolor=edge_color[j], linewidth=.25)

    use_min_max = max_val + .1
    min_val = min_val - .1  # -1*use_min_max
    m_val = use_min_max
    axs.set_xlim(min_val, m_val)  # if x and y axes are shared, this will adjust lim on all subplots
    axs.set_ylim(min_val, m_val)
    axs.vlines(0, min_val, m_val, color='black', linewidth=.25)
    axs.hlines(0, min_val, m_val, color='black', linewidth=.25)
    xytick = _compute_ticks(min_val, m_val, ax_pos=xposition)
    axs.set_yticks(xytick, labels=[str(yt) for yt in xytick], fontsize=6)
    axs.set_xticks(xytick, labels=[str(yt) for yt in xytick], fontsize=6)
    plt.tight_layout()
    if set_size is not None:
        _set_size(set_size[0], set_size[1], axs)
    fig.savefig(os.path.join(out_dir, title + ".svg"))
    return fig


def create_save_prop_line_plot(axs: plt.Axes, fig: plt.Figure, title: str, boot_data: np.ndarray, x=None, xposition=0.,
                               yposition=1.2, out_dir="plots", set_size=None, rotate_x_labels=False):
    """
    expect boot_data as 3d nparray of groups, prop voxel, data
    """
    fill_colors = ["black", "#A9A9A9", "none", "none"]
    marker_colors = ["black", "#A9A9A9", "black", "#A9A9A9"]
    zorders = [1, 3, 5, 7]
    # boot_data = boot_data.squeeze()
    if boot_data.ndim > 2:
        effect = np.nanmean(boot_data, axis=2)
        low_quant = np.nanquantile(boot_data, .025, axis=2)
        up_quant = np.nanquantile(boot_data, .975, axis=2)
        max_val = np.nanmax(up_quant)
        min_val = np.nanmin(low_quant)
        print(max_val, min_val)
    else:
        effect = boot_data
        max_val = np.nanmax(effect)
        min_val = np.nanmin(effect)
        low_quant = None
        up_quant = None
    if x is None:
        x = np.arange(effect.shape[1])
    elif type(x) is not np.ndarray:
        x = np.array(x)
    min_x, max_x = np.min(x), np.max(x)
    axs, fig = _set_fig_params(axs, fig, min_val, max_val, xposition)
    for i in range(len(effect)):
        x_jit = x - i * .02  # + random.choice([0, ((i + 1) % 3) * .02 * (max_x - min_x), -1 *  ((i - 1) % 3) * .02 * (max_x - min_x)])
        y = effect[i]
        print(x_jit, y)
        axs.scatter(x_jit, y, facecolors=fill_colors[i], edgecolors=marker_colors[i], linewidth=.5, s=10,
                    zorder=zorders[i])
        if low_quant is not None and up_quant is not None:
            axs.errorbar(x_jit, y, yerr=[y - low_quant[i], up_quant[i] - y], color=marker_colors[i], linewidth=.5,
                         zorder=zorders[i])
    ytick = _compute_ticks(min_val, max_val, xposition)
    xtick = x
    if rotate_x_labels:
        rt = 45
    else:
        rt = 0
    axs.set_yticks(ytick, labels=[str(yt) for yt in ytick], fontsize=6)
    # axs.set_xticks(xtick, labels=[str(xt) for xt in xtick], fontsize=6)
    axs.set_xticks(xtick)
    axs.set_xticklabels(labels=[str(round(xt * 100)) for xt in xtick], fontsize=6, y=min_val,
                        rotation=rt)  # place labels below lowest error bar or data point
    axs.set_xbound(np.min(x) - .1, np.max(x) + .1)
    axs.invert_xaxis()
    if set_size is not None:
        _set_size(set_size[0], set_size[1], axs)
    fig.savefig(os.path.join(out_dir, title + ".svg"))
    return fig


def create_save_surf_comparison(axs: plt.Axes, fig: plt.Figure, title: str, boot_data: np.ndarray,
                                group_labels: Tuple[str] = None,
                                divs=None, x=None, ylim=None, xposition=0., yposition=0., out_dir="plots",
                                set_size=None):
    """
    expect boot_data as 3d nparray of groups, bins (xaxis), data
    """
    colors = ["tab:gray", "#D95319"]
    boot_data = boot_data.squeeze()
    if boot_data.ndim > 2:
        effect = np.mean(boot_data, axis=2)
        low_quant = np.quantile(boot_data, .05, axis=2)
        up_quant = np.quantile(boot_data, .95, axis=2)
        max_val = np.max(up_quant)
        min_val = np.min(low_quant)
    else:
        effect = boot_data
        max_val = np.max(effect)
        min_val = np.min(effect)
        low_quant = None
        up_quant = None
    if x is None:
        x = np.arange(effect.shape[1])
    elif type(x) is not np.ndarray:
        x = np.array(x)

    if ylim is not None:
        min_val = ylim[0]
        max_val = ylim[1]
    spread = max_val - min_val
    # max_val = max_val + .05 * spread
    # min_val = min_val - .05 * spread
    min_x, max_x = np.min(x), np.max(x)
    axs, fig = _set_fig_params(axs, fig, min_val, max_val, xposition)
    for i in range(len(effect)):
        axs.plot(x, effect[i], color=colors[i], linewidth=1.)
        if low_quant is not None and up_quant is not None:
            axs.fill_between(x, low_quant[i], up_quant[i], alpha=.5, color=colors[i])
        if divs is not None:
            for boundary in divs:
                axs.vlines(boundary, min_val, max_val, color='black', alpha=.6, linestyles='--', linewidth=0.5)
    ytick = _compute_ticks(min_val, max_val, xposition)
    xtick = _compute_ticks(min_x, max_x, yposition)
    axs.set_yticks(ytick, labels=[str(yt) for yt in ytick], fontsize=6)
    axs.set_xticks(xtick, labels=[str(xt) for xt in xtick], fontsize=6, y=min_val)
    if set_size is not None:
        _set_size(set_size[0], set_size[1], axs)
    fig.savefig(os.path.join(out_dir, title + ".svg"))
    return fig
