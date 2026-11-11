"""Export aligned three-row paper figures and independent shared legends."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plot_experiment_summaries as p
import plot_rho_experiment as r
import plot_user_study_surveys as u

OUT = p.DEFAULT_FIGURE_DIR / 'layout_shared'


def save_legend(handles, labels, name, columns=None, balanced=False):
    fig = plt.figure(figsize=(32, 3))
    kwargs = dict(frameon=False, fontsize=20, handlelength=1.8,
                  markerscale=.5, columnspacing=1.2, labelspacing=.5)
    artists = []
    if balanced:
        fig.set_size_inches(32, 1.2)
        for hs, ls, y in [(handles[:5], labels[:5], .75), (handles[5:], labels[5:], .25)]:
            artists.append(fig.legend(hs, ls, loc='center', bbox_to_anchor=(.5, y),
                                      ncol=len(ls), **kwargs))
    else:
        artists.append(fig.legend(handles, labels, loc='center', ncol=columns, **kwargs))
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = Bbox.union([a.get_window_extent(renderer) for a in artists])
    bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).padded(.08)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches=bounds)
    plt.close(fig)


def plot_pair(name, time, questions, style='line'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    plot = p.plot_bars if style == 'bar' else p.plot_lines
    plot(axes[0], time, 'Time (seconds)')
    plot(axes[1], questions, '# of questions')
    fig.subplots_adjust(top=.94, bottom=.32, left=.14, right=.99, wspace=.55)
    fig.savefig(OUT / f'{name}.pdf')
    handles, labels = axes[1].get_legend_handles_labels()
    plt.close(fig)
    return handles, labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for prefix, name, style in [('02_alpha_beta_ratio','alpha_beta','line'),
                                ('03_dataset_count','dataset_count','line'),
                                ('04_dimension','dimension','line'),
                                ('05_dataset_size','dataset_size','line'),
                                ('06_data_type_ratio','data_type_3','bar')]:
        handles, labels = plot_pair(name, p.DEFAULT_SUMMARY_DIR/f'{prefix}_time.csv',
                                   p.DEFAULT_SUMMARY_DIR/f'{prefix}_questions.csv', style)
        if name == 'alpha_beta': save_legend(handles, labels, 'sharedq_legend', columns=1)
        if name == 'dimension': save_legend(handles, labels, 'methods_legend_one_row', columns=9)
        if name == 'data_type_3': save_legend(handles, labels, 'bar_legend', columns=3)
    for family in ['Google_Playstore','Realtor']:
        handles, labels = plot_pair(family, p.DEFAULT_REAL_SUMMARY_DIR/f'{family}_time.csv',
                                   p.DEFAULT_REAL_SUMMARY_DIR/f'{family}_questions.csv')
    save_legend(handles, labels, 'real_legend_two_rows', balanced=True)
    fig, axes = plt.subplots(1, 2, figsize=(16,7.5))
    rows = r.read_rows(r.DEFAULT_RESULT_DIR/'experiment_rho_summary.csv')
    grouped = r.group_by_method(rows)
    r.plot_metric(axes[0],grouped,r.TIME_COL,'Time (seconds)')
    r.plot_metric(axes[1],grouped,r.QUESTION_COL,'# of questions')
    r.set_rho_ticks(axes,rows)
    fig.subplots_adjust(top=.94,bottom=.32,left=.14,right=.99,wspace=.55)
    fig.savefig(OUT/'rho.pdf')
    plt.close(fig)

    # Reuse the user-study data and bar rendering, but export its legend separately.
    summary = u.summarize_by_method(u.read_surveys(u.DEFAULT_SURVEY_PATH))
    methods = u.ordered_methods(summary)
    original_save = u.save_figure_both
    def export_study(fig, path, **kwargs):
        for legend in list(fig.legends): legend.remove()
        fig.set_size_inches(16,7.5)
        fig.subplots_adjust(top=.94,bottom=.32,left=.14,right=.86)
        for ax in fig.axes:
            ax.yaxis.label.set_fontsize(38)
            ax.xaxis.label.set_fontsize(38)
            ax.tick_params(labelsize=32)
        fig.axes[0].set_xticklabels(['# of\nQuestions','Annoyance\nScore','Satisfaction\nScore'])
        fig.savefig(OUT/'user_study.pdf')
    u.save_figure_both = export_study
    u.plot_combined(summary, methods, OUT/'user_study.pdf')
    u.save_figure_both = original_save
    handles = [Patch(facecolor='white',edgecolor=u.COLORS[m],hatch=u.HATCHES[m],
                     label=u.METHOD_LABELS[m]) for m in methods]
    save_legend(handles,[u.METHOD_LABELS[m] for m in methods],'user_study_legend',columns=3)
    print(OUT)


if __name__ == '__main__':
    main()
