"""Functions used for processing the results of the optimisation runs and
plotting them.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter
from scipy.stats import median_absolute_deviation, wilcoxon
from statsmodels.stats.multitest import multipletests
from .. import test_problems


def process_results(results_dir, problem_names, method_names, budget=250,
                    exp_no_start=1, exp_no_end=51):
    """Processes the optimisation runs and returns a dictionary of results.
    The function reads attempts to read in npz files in the ``results_dir``
    directory that match the mask:
        f'{problem_name:}_{run_no:}_{budget:}_{method_name:}.npz',
    where problem_name is each of the problem names in ``problem_names``,
    method_name is each of the methods in ``method_names``, and
    run_no is in the range [exp_no_start, exp_no_end].

    It returns a dictionary of the form:
        results[problem_name][method_name] = Y, numpy.npdarray (N_EXPS, budget)
    where and N_EXPS = exp_no_end - exp_no_start + 1.

    The numpy.ndarray contains the minimum seen expensive function evaluations
    from each optimisation run in its rows, such that Y[i, j] >= Y[i, k] for
    optimisation runs 'i' and elements 'j' and 'k'.
    """
    # results[problem_name][method_name] = array of shape (N_EXPS, budget)
    results = {}

    for problem_name in problem_names:
        results[problem_name] = {}

        # load the problem
        f_class = getattr(test_problems, problem_name)
        f = f_class()
        f_yopt = f.yopt

        for method_name in method_names:
            D = np.zeros((exp_no_end - exp_no_start + 1, budget))

            # get the raw results for each problem instance
            for i, run_no in enumerate(range(exp_no_start, exp_no_end + 1)):
                fn = f'{problem_name:}_{run_no:}_{budget:}_{method_name:}.npz'
                filepath = os.path.join(results_dir, fn)

                try:
                    with np.load(filepath, allow_pickle=True) as data:
                        Ytr = np.squeeze(data['Ytr'])
                        if Ytr.size != budget:
                            print('{:s} does not contain enough function '
                                  + ' evaluations: '
                                  + '{:d} (budget = {:d})'.format(Ytr.size,
                                                                  budget))
                        D[i, :] = np.squeeze(data['Ytr'])
                except Exception as e:
                    print(fn)
                    print(e)

            # calculate the absolute distance to the minima
            D = np.abs(D - f_yopt)

            # calculate the best (lowest) value seen at each iteration
            D = np.minimum.accumulate(D, axis=1)

            results[problem_name][method_name] = D

    return results


def plot_convergence(data,
                     problem_names,
                     problem_names_for_paper,
                     problem_logplot,
                     method_names,
                     method_names_for_paper,
                     LABEL_FONTSIZE,
                     TITLE_FONTSIZE,
                     TICK_FONTSIZE,
                     LEGEND_FONTSIZE,
                     save=False):

    for problem_name, paper_problem_name, logplot in zip(problem_names,
                                                         problem_names_for_paper,
                                                         problem_logplot):
        # load the problem
        f_class = getattr(test_problems, problem_name)
        f = f_class()
        dim = f.dim

        D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

        method_to_col = {}
        col_counter = 0

        for method_name, paper_method_name in zip(method_names,
                                                  method_names_for_paper):
            res = data[problem_name][method_name]

            # take into account the fact the first 2 * dim points are LHS
            # so we start plotting after it
            xvals = np.arange(2 * dim + 1, res.shape[1] + 1)
            res = res[:, 2 * dim:]

            D['yvals'].append(res)
            D['y_labels'].append(paper_method_name)
            D['xvals'].append(xvals)

            if method_name not in method_to_col:
                method_to_col[method_name] = col_counter
                col_counter += 1

            D['col_idx'].append(method_to_col[method_name])

        # create total colour range
        colors = plt.cm.rainbow(np.linspace(0, 1, col_counter))

        # plot!
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharey=False)
        results_plot_maker(ax,
                           D['yvals'],
                           D['y_labels'],
                           D['xvals'],
                           D['col_idx'],
                           xlabel='Function Evaluations',
                           ylabel='Regret',
                           title='{:s} ({:d})'.format(paper_problem_name, dim),
                           colors=colors,
                           LABEL_FONTSIZE=LABEL_FONTSIZE,
                           TITLE_FONTSIZE=TITLE_FONTSIZE,
                           TICK_FONTSIZE=TICK_FONTSIZE,
                           semilogy=logplot,
                           use_fill_between=True)
        if save:
            fname = f'convergence_{problem_name:s}.pdf'
            plt.savefig(fname, bbox_inches='tight')
        plt.show()

    # create separate legend image
    fig, ax = plt.subplots(1, 1, figsize=(19, 1), sharey=False)
    results_plot_maker(ax,
                       D['yvals'],
                       D['y_labels'],
                       D['xvals'],
                       D['col_idx'],
                       xlabel='Function Evaluations',
                       ylabel='Regret',
                       title='',
                       colors=colors,
                       LABEL_FONTSIZE=LABEL_FONTSIZE,
                       TITLE_FONTSIZE=TITLE_FONTSIZE,
                       TICK_FONTSIZE=TICK_FONTSIZE,
                       semilogy=True,
                       use_fill_between=False)

    legend = plt.legend(loc=3, framealpha=1, frameon=False,
                        fontsize=LEGEND_FONTSIZE,
                        handletextpad=0.25,
                        columnspacing=1,
                        ncol=9)

    # increase legend line widths
    for legobj in legend.legendHandles:
        legobj.set_linewidth(5.0)

    # remove all plotted lines
    for _ in range(len(ax.lines)):
        ax.lines.pop(0)

    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*[bbox.extents + np.array([-5, -5, 5, 5])])
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    if save:
        fname = f'convergence_LEGEND.pdf'
        fig.savefig(fname, dpi='figure', bbox_inches=bbox)
    plt.show()


def plot_convergence_combined(data,
                              problem_names,
                              problem_names_for_paper,
                              problem_logplot,
                              method_names,
                              method_names_for_paper,
                              LABEL_FONTSIZE,
                              TITLE_FONTSIZE,
                              TICK_FONTSIZE,
                              save=False):

    N = len(problem_names)

    fig, ax = plt.subplots(N // 2, 2, figsize=(16, 4 * N // 2), sharex='all')

    for a, problem_name, paper_problem_name, logplot in zip(ax.flat,
                                                            problem_names,
                                                            problem_names_for_paper,
                                                            problem_logplot):
        # load the problem
        f_class = getattr(test_problems, problem_name)
        f = f_class()
        dim = f.dim

        D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

        method_to_col = {}
        col_counter = 0

        for method_name, paper_method_name in zip(method_names,
                                                  method_names_for_paper):
            res = data[problem_name][method_name]

            # take into account the fact the first 2 * dim points are LHS
            # so we start plotting after it
            xvals = np.arange(2 * dim + 1, res.shape[1] + 1)
            res = res[:, 2 * dim:]

            D['yvals'].append(res)
            D['y_labels'].append(paper_method_name)
            D['xvals'].append(xvals)

            if method_name not in method_to_col:
                method_to_col[method_name] = col_counter
                col_counter += 1

            D['col_idx'].append(method_to_col[method_name])

        # create total colour range
        colors = plt.cm.rainbow(np.linspace(0, 1, col_counter))

        # only the bottom row should have x-axis labels
        if problem_name in problem_names[-2:]:
            xlabel = 'Function Evaluations'
        else:
            xlabel = None

        # only the left column should have y-axis labels
        if problem_name in problem_names[::2]:
            ylabel = 'Regret'
        else:
            ylabel = None

        results_plot_maker(a,
                           D['yvals'],
                           D['y_labels'],
                           D['xvals'],
                           D['col_idx'],
                           xlabel=xlabel,
                           ylabel=ylabel,
                           title='{:s} ({:d})'.format(paper_problem_name, dim),
                           colors=colors,
                           LABEL_FONTSIZE=LABEL_FONTSIZE,
                           TITLE_FONTSIZE=TITLE_FONTSIZE,
                           TICK_FONTSIZE=TICK_FONTSIZE,
                           semilogy=logplot,
                           use_fill_between=True)

        # ensure labels are all in the same place!
        a.get_yaxis().set_label_coords(-0.08, 0.5)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                        wspace=0.1, hspace=0.14)
    if save:
        probs = '_'.join(problem_names)
        fname = f'convergence_combined_{probs:s}.pdf'
        plt.savefig(fname, bbox_inches='tight')
    plt.show()


def results_plot_maker(ax, yvals, y_labels, xvals, col_idx,
                       xlabel, ylabel, title, colors,
                       LABEL_FONTSIZE, TITLE_FONTSIZE, TICK_FONTSIZE,
                       semilogy=False, use_fill_between=True):
    # here we assume we're plotting to a matplotlib axis object
    # and yvals is a LIST of arrays of size (n_runs, iterations),
    # where each can be different sized
    # and if xvals is given then len(xvals) == len(yvals)

    # set the labelling
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    for c, x, Y, Y_lbl in zip(col_idx, xvals, yvals, y_labels):
        color = colors[c]

        # calculate median run and upper/lower percentils
        bot, mid, top = np.percentile(Y, [25, 50, 75], axis=0)

        if use_fill_between:
            ax.fill_between(x, bot.flat, top.flat, color=color, alpha=0.25)

        ax.plot(x, mid, color=color, label='{:s}'.format(Y_lbl))
        ax.plot(x, bot.flat, '--', color=color, alpha=0.25)
        ax.plot(x, top.flat, '--', color=color, alpha=0.25)

    # set the xlim
    min_x = np.min([np.min(x) for x in xvals])
    max_x = np.max([np.max(x) for x in xvals])
    ax.set_xlim([0, max_x + 1])

    if semilogy:
        ax.semilogy()
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x: >4.1f}'))

    ax.axvline(min_x, linestyle='dashed', color='gray', linewidth=1, alpha=0.5)

    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)


def plot_boxplots(data,
                  budgets,
                  problem_names,
                  problem_names_for_paper,
                  problem_logplot,
                  method_names,
                  method_names_for_paper,
                  LABEL_FONTSIZE,
                  TITLE_FONTSIZE,
                  TICK_FONTSIZE,
                  save=False):

    for problem_name, paper_problem_name, logplot in zip(problem_names,
                                                         problem_names_for_paper,
                                                         problem_logplot):
        # load the problem
        f_class = getattr(test_problems, problem_name)
        f = f_class()
        dim = f.dim

        D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

        method_to_col = {}
        col_counter = 0

        for method_name, paper_method_name in zip(method_names,
                                                  method_names_for_paper):
            res = data[problem_name][method_name]

            D['yvals'].append(res)
            D['y_labels'].append(paper_method_name)

            if method_name not in method_to_col:
                method_to_col[method_name] = col_counter
                col_counter += 1

            D['col_idx'].append(method_to_col[method_name])

        # create total colour range
        colors = plt.cm.rainbow(np.linspace(0, 1, col_counter))

        # plot!
        fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True)

        for i, (a, budget) in enumerate(zip(ax, budgets)):
            YV = [Y[:, :budget] for Y in D['yvals']]
            title = '{:s} ({:d}): T = {:d}'.format(paper_problem_name, dim, budget)

            y_axis_label = 'Regret' if i == 0 else None

            box_plot_maker(a,
                           YV,
                           D['y_labels'],
                           D['col_idx'],
                           colors,
                           y_axis_label,
                           title,
                           logplot,
                           LABEL_FONTSIZE,
                           TITLE_FONTSIZE,
                           TICK_FONTSIZE,
                           )

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=0.03, hspace=0.16)

        if save:
            fname = f'boxplots_{problem_name:s}.pdf'
            plt.savefig(fname, bbox_inches='tight')

        plt.show()


def box_plot_maker(a, yvals, y_labels, col_idx, colors, y_axis_label,
                   title, logplot,
                   LABEL_FONTSIZE, TITLE_FONTSIZE, TICK_FONTSIZE):

    data = [Y[:, -1] for Y in yvals]

    medianprops = dict(linestyle='-', color='black')

    bplot = a.boxplot(data, patch_artist=True, medianprops=medianprops)

    if y_labels is not None:
        a.set_xticklabels(y_labels, rotation=90)

    for patch, c in zip(bplot['boxes'], col_idx):
        patch.set(facecolor=colors[c])

    a.set_ylabel(y_axis_label, fontsize=LABEL_FONTSIZE)
    a.set_title(title, fontsize=TITLE_FONTSIZE)

    if logplot:
        a.semilogy()
    else:
        a.yaxis.set_major_formatter(StrMethodFormatter('{x: >4.1f}'))
    a.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    a.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)


def plot_boxplots_combined(data,
                           budgets,
                           problem_names,
                           problem_names_for_paper,
                           problem_logplot,
                           method_names,
                           method_names_for_paper,
                           LABEL_FONTSIZE,
                           TITLE_FONTSIZE,
                           TICK_FONTSIZE,
                           save=False):

    N = len(problem_names)

    fig, ax = plt.subplots(N, 3, figsize=(16, 3.5 * N),
                           sharex='all', sharey='row')

    for a, problem_name, paper_problem_name, logplot in zip(ax,
                                                            problem_names,
                                                            problem_names_for_paper,
                                                            problem_logplot):
        # load the problem
        f_class = getattr(test_problems, problem_name)
        f = f_class()
        dim = f.dim

        D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

        method_to_col = {}
        col_counter = 0

        for method_name, paper_method_name in zip(method_names,
                                                  method_names_for_paper):
            res = data[problem_name][method_name]

            D['yvals'].append(res)
            D['y_labels'].append(paper_method_name)

            if method_name not in method_to_col:
                method_to_col[method_name] = col_counter
                col_counter += 1

            D['col_idx'].append(method_to_col[method_name])

        # create total colour range
        colors = plt.cm.rainbow(np.linspace(0, 1, col_counter))

        for i, (aa, budget) in enumerate(zip(a, budgets)):
            YV = [Y[:, :budget] for Y in D['yvals']]
            title = '{:s} $({:d})$: T = {:d}'.format(paper_problem_name, dim, budget)

            y_axis_label = 'Regret' if i == 0 else None

            box_plot_maker(aa,
                           YV,
                           D['y_labels'],
                           D['col_idx'],
                           colors,
                           y_axis_label,
                           title,
                           logplot,
                           LABEL_FONTSIZE,
                           TITLE_FONTSIZE,
                           TICK_FONTSIZE,
                           )

            # ensure labels are all in the same place!
            aa.get_yaxis().set_label_coords(-0.13, 0.5)

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=0.03, hspace=0.18)

        if save:
            probs = '_'.join(problem_names)
            fname = f'boxplots_combined_{probs:s}.pdf'
            plt.savefig(fname, bbox_inches='tight')

    plt.show()


def plot_egreedy_comparison(data,
                            budgets,
                            problem_names,
                            problem_names_for_paper,
                            problem_logplot,
                            method_names,
                            x_labels,
                            LABEL_FONTSIZE,
                            TITLE_FONTSIZE,
                            TICK_FONTSIZE,
                            save=False):

    for problem_name, paper_problem_name, logplot in zip(problem_names,
                                                         problem_names_for_paper,
                                                         problem_logplot):
        # load the problem
        f_class = getattr(test_problems, problem_name)
        f = f_class()
        dim = f.dim

        D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

        for method_name in method_names:
            res = data[problem_name][method_name]

            D['yvals'].append(res)

        # plot!
        fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True)

        for i, (a, budget) in enumerate(zip(ax, budgets)):
            YV = [Y[:, :budget] for Y in D['yvals']]

            N = len(method_names)
            # offset indicies for each box location to space them out
            box_inds = np.arange(N)
            c = 0
            for i in range(0, N, 2):
                box_inds[i:i + 2] += c
                c += 1

            medianprops = dict(linestyle='-', color='black')
            bplot = a.boxplot([Y[:, -1] for Y in YV],
                              positions=box_inds,
                              patch_artist=True,
                              medianprops=medianprops)

            a.set_xticks(np.arange(3 * len(x_labels))[::3] + 0.5)
            a.set_xticklabels(x_labels, rotation=0)

            for i, patch in enumerate(bplot['boxes']):
                if i % 2 == 0:
                    patch.set_facecolor('g')
                else:
                    patch.set_facecolor('r')
                    patch.set(hatch='//')

            if budget == budgets[0]:
                a.set_ylabel('Regret',
                             fontsize=LABEL_FONTSIZE)

            title = '{:s} ({:d}): T = {:d}'.format(paper_problem_name, dim, budget)
            a.set_title(title, fontsize=TITLE_FONTSIZE)

            if logplot:
                a.semilogy()
            else:
                a.yaxis.set_major_formatter(StrMethodFormatter('{x: >4.1f}'))

            a.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            a.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=0.03, hspace=0.16)

        if save:
            fname = f'egreedy_compare_{problem_name:s}.pdf'
            plt.savefig(fname, bbox_inches='tight')

        plt.show()


def plot_egreedy_comparison_combined(data,
                                     budgets,
                                     problem_names,
                                     problem_names_for_paper,
                                     problem_logplot,
                                     method_names,
                                     x_labels,
                                     LABEL_FONTSIZE,
                                     TITLE_FONTSIZE,
                                     TICK_FONTSIZE,
                                     save=False):

    N = len(problem_names)

    fig, ax = plt.subplots(N, 3, figsize=(16, 3.5 * N),
                           sharex='all', sharey='row')

    for a, problem_name, paper_problem_name, logplot in zip(ax,
                                                            problem_names,
                                                            problem_names_for_paper,
                                                            problem_logplot):
        # load the problem
        f_class = getattr(test_problems, problem_name)
        f = f_class()
        dim = f.dim

        D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

        for method_name in method_names:
            res = data[problem_name][method_name]

            D['yvals'].append(res)

        for i, (aa, budget) in enumerate(zip(a, budgets)):
            YV = [Y[:, :budget] for Y in D['yvals']]

            N = len(method_names)
            # offset indicies for each box location to space them out
            box_inds = np.arange(N)
            c = 0
            for i in range(0, N, 2):
                box_inds[i:i + 2] += c
                c += 1

            medianprops = dict(linestyle='-', color='black')
            bplot = aa.boxplot([Y[:, -1] for Y in YV],
                               positions=box_inds,
                               patch_artist=True,
                               medianprops=medianprops)

            aa.set_xticks(np.arange(3 * len(x_labels))[::3] + 0.5)
            aa.set_xticklabels(x_labels, rotation=0)

            for i, patch in enumerate(bplot['boxes']):
                if i % 2 == 0:
                    patch.set_facecolor('g')
                else:
                    patch.set_facecolor('r')
                    patch.set(hatch='//')

            if budget == budgets[0]:
                aa.set_ylabel('Regret', fontsize=LABEL_FONTSIZE)

            title = '{:s} ({:d}): T = {:d}'.format(paper_problem_name, dim, budget)
            aa.set_title(title, fontsize=TITLE_FONTSIZE)

            if logplot:
                aa.semilogy()
            else:
                aa.yaxis.set_major_formatter(StrMethodFormatter('{x: >4.1f}'))

            aa.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            aa.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)

            # ensure labels are all in the same place!
            aa.get_yaxis().set_label_coords(-0.11, 0.5)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                        wspace=0.03, hspace=0.18)

    if save:
        probs = '_'.join(problem_names)
        fname = f'egreedy_compare_combined_{probs:s}.pdf'
        plt.savefig(fname, bbox_inches='tight')

    plt.show()


def create_table_data(results, problem_names, method_names, n_exps):
    """

    """
    method_names = np.array(method_names)
    n_methods = len(method_names)

    # table_data[problem_name] = {'median', 'MAD', 'stats_equal_to_best_mask'}
    table_data = {}

    for problem_name in problem_names:
        best_seen_values = np.zeros((n_methods, n_exps))

        for i, method_name in enumerate(method_names):
            # best seen evaluate at the end of the optimisation run
            best_seen_values[i, :] = results[problem_name][method_name][:, -1]

        medians = np.median(best_seen_values, axis=1)
        MADS = median_absolute_deviation(best_seen_values, axis=1)

        # best method -> lowest median value
        best_method_idx = np.argmin(medians)

        # mask of methods equivlent to the best
        stats_equal_to_best_mask = np.zeros(n_methods, dtype='bool')
        stats_equal_to_best_mask[best_method_idx] = True

        # perform wilcoxon signed rank test between best and all other methods
        p_values = []
        for i, method_name in enumerate(method_names):
            if i == best_method_idx:
                continue
            # a ValueError will be thrown if the runs are all identical,
            # therefore we can assign a p-value of 0 as they are identical
            try:
                _, p_value = wilcoxon(best_seen_values[best_method_idx, :],
                                      best_seen_values[i, :])
                p_values.append(p_value)

            except ValueError:
                p_values.append(0)

        # calculate the Holm-Bonferroni correction
        reject_hyp, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05,
                                                          method='holm')

        for reject, method_name in zip(reject_hyp, [m for m in method_names
                                       if m != method_names[best_method_idx]]):
            # if we can't reject the hypothesis that a technique is
            # statistically equivilent to the best method
            if not reject:
                idx = np.where(np.array(method_names) == method_name)[0][0]
                stats_equal_to_best_mask[idx] = True

        # store the data
        table_data[problem_name] = {'medians': medians,
                                    'MADS': MADS,
                                    'stats_equal_to_best_mask': stats_equal_to_best_mask}

    return table_data


def create_table(table_data, problem_rows, problem_paper_rows,
                 problem_dim_rows, method_names, method_names_for_table):
    """

    """

    head = r"""
  \begin{table}[t]
  \setlength{\tabcolsep}{2pt}
  \sisetup{table-format=1.2e-1,table-number-alignment=center}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{l | SS| SS| SS| SS| SS}"""

    foot = r"""  \end{tabular}
  }
  \vspace*{0.1mm}
  \caption{}
  \label{tbl:synthetic_results}
  \end{table}"""

    print(head)
    for probs, probs_paper, probs_dim in zip(problem_rows, problem_paper_rows,
                                             problem_dim_rows):

        print(r'    \toprule')
        print(r'    \bfseries Method')

        # column titles: Problem name (dim).
        print_string = ''
        for prob, dim in zip(probs_paper, probs_dim):
            print_string += r'    & \multicolumn{2}{c'
            # last column does not have a vertical dividing line
            if prob != probs_paper[-1]:
                print_string += r'|'
            print_string += r'}{\bfseries '
            print_string += r'{:s} ({:d})'.format(prob, dim)
            print_string += '} \n'

        print_string = print_string[:-2] + ' \\\\ \n'

        # column titles: Median MAD
        for prob in probs:
            print_string += r'    & \multicolumn{1}{c}{Median}'
            print_string += r' & \multicolumn{1}{c'
            # last column does not have a vertical dividing line
            if prob != probs[-1]:
                print_string += r'|'
            print_string += '}{MAD}\n'
        print_string = print_string[:-1] + '  \\\\ \\midrule'
        print(print_string)

        # results printing
        for i, (method_name, method_name_table), in enumerate(zip(method_names,
                                                                  method_names_for_table)):
            print_string = '    '
            print_string += method_name_table + ' & '

            # table_data[problem_name] = {'median', 'MAD', 'stats_equal_to_best_mask'}
            for prob in probs:
                med = '{:4.2e}'.format(table_data[prob]['medians'][i])
                mad = '{:4.2e}'.format(table_data[prob]['MADS'][i])

                best_methods = table_data[prob]['stats_equal_to_best_mask']
                best_idx = np.argmin(table_data[prob]['medians'])

                if i == best_idx:
                    med = r'\best ' + med
                    mad = r'\best ' + mad

                elif best_methods[i]:
                    med = r'\statsimilar ' + med
                    mad = r'\statsimilar ' + mad

                print_string += med + ' & ' + mad + ' & '

            print_string = print_string[:-2] + '\\\\'
            print(print_string)

        print('\\bottomrule')

    print(foot)
    print()
