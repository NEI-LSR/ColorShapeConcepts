import random

from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
from typing import Tuple
import os
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
                        xposition=0., out_dir="plots", data_spread="ci", y_axis_label=None, ymin=None, ymax=None):
    """
    expect data as 3d nparray of groups, within group, data
    """
    if data_spread not in ["ci", "full"]:
        raise ValueError
    colors = ["tab:gray", "tab:blue", "tab:pink", "tab:green", "tab:orange"]
    m_val = xposition
    min_val = xposition
    if ymax is not None:
        m_val = max(ymax, xposition)
    if ymin is not None:
        min_val = min(ymin, xposition)
    space = .1
    global_min = []
    for i, r in enumerate(boot_data):
        w = (1 - space) / (len(r))
        for j, g in enumerate(r):
            x = w * (j + .5 * space) + i
            effect = g.mean(axis=-1)
            # want spacing of half bar between groups
            axs.bar(x=x, height=effect - xposition, bottom=xposition, width=w, color=colors[j])
            if y_axis_label:
                axs.set_ylabel(y_axis_label, fontsize=8)
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
    ytick = _compute_ticks(min_val, m_val, ax_pos=xposition)
    axs.set_yticks(ytick, labels=[str(yt) for yt in ytick], fontsize=8)
    axs, fig = _set_fig_params(axs, fig, min_val, m_val, xposition)
    print(min(global_min))
    axs.set_xticks(np.array(range(len(group_labels))) + .5 * ((len(r) - 1) * w + space - .01))
    axs.set_xticklabels(labels=group_labels, rotation=45, fontsize=8, y=min(global_min)) # place labels below lowest error bar or data point
    fig.savefig(os.path.join(out_dir, title + ".svg"))
    return fig


def create_save_line_plot(axs: plt.Axes, fig: plt.Figure, title: str, boot_data: np.ndarray, group_labels: Tuple[str]=None,
                          x=None, ylim=None, xposition=0., yposition=0., out_dir="plots", set_size=None):
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
    fig.savefig(os.path.join(out_dir, title + ".svg"))
    return fig


def create_save_learning_plot(axs: plt.Axes, fig: plt.Figure, title: str, boot_data: np.ndarray, raw_data: np.ndarray, group_labels: Tuple[str]=None, x=None, ylim=None, xposition=0., out_dir="plots"):
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


def create_save_scatter_plot(axs: plt.Axes, fig: plt.Figure, title: str, scatter_data: list, x_name: str, y_name: str,
                             group_labels: Tuple[str]=None, fit_lm=False, xposition=0., out_dir="plots"):
    """
    expect scatter_data as list (len=groups) of 2d nparrays of x,y dim, number data points
    """
    global_min_max = []
    for i, r in enumerate(scatter_data):
        axs[i].scatter(r[0], r[1],  alpha=.8, color='lightgrey', edgecolors='black',s=20, linewidth=.5)
        global_min_max.append(max([abs(r.min()), abs(r.max())])) # to set x and y lim as same and keep all points, set max as max among max of +x, -x, +y, -y
    
    use_min_max = max(global_min_max)+.1 # calculate max and add padding
    min_val = -1*use_min_max
    m_val = use_min_max
    axs[i].set_xlim(-1*use_min_max, use_min_max) # if x and y axes are shared, this will adjust lim on all subplots
    axs[i].set_ylim(-1*use_min_max, use_min_max)
    for i, r in enumerate(scatter_data): # after calculating lims, add other features back in
        axs[i].vlines(0, -1*use_min_max, use_min_max, color='black', linewidth=.75)
        axs[i].hlines(0, -1*use_min_max, use_min_max, color='black', linewidth=.75)
        axs[i].set_title(group_labels[i], fontsize=8)
    xytick = _compute_ticks(min_val, m_val, ax_pos=xposition)
    axs[i].set_yticks(xytick, labels=[str(yt) for yt in xytick], fontsize=8)
    axs[i].set_xticks(xytick, labels=[str(yt) for yt in xytick], fontsize=8)
    fig.text(0.5, -0.01, x_name, ha='center', fontsize=8)  # Centered x-label
    fig.text(-0.01, 0.5, y_name, va='center', rotation='vertical', fontsize=8)  # Centered y-label
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, title + ".svg"))
    return fig
    
