import argparse
from collections import namedtuple
from contextlib import contextmanager
import json
import multiprocessing
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import ATTACKS, PROJECTIONS

matplotlib.use('agg')
# Monkey patch matplotlib.font_manager._get_font to not use an LRU cache. This disables font caching
# to prevent figures from possibly being distorted when generated in a multiprocessing context.
# As of matplotlib PR #19618, _get_font is keyed by thread ID to prevent segfaults (testing suggests
# the thread ID matches across multiprocess processes).
def _get_font(filename, hinting_factor, *, _kerning_factor, thread_id):
    return matplotlib.ft2font.FT2Font(
        filename, hinting_factor, _kerning_factor=_kerning_factor)
matplotlib.font_manager._get_font = _get_font


CMAP_DARK = plt.cm.get_cmap('tab20').colors[0::2]
CMAP_LIGHT = plt.cm.get_cmap('tab20').colors[1::2]
FONT_SIZE = 22
FIG_SIZE = 18
LEGEND_KWARGS = {'markerscale': 2}
LOCK = multiprocessing.Lock()

PlotParams = namedtuple(
    'PlotParams', [
        'attack',
        'projection_alg',
        'classes',
        'ground_truth_lookup',
        'outdir',
        'projections',
        'adv_projections',
        'xlim',
        'ylim',
        'width',
        'approximate',
    ])

@contextmanager
def mpl_font_size(font_size):
    old_font_size = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': font_size})
    try:
        yield
    finally:
        matplotlib.rcParams.update({'font.size': old_font_size})


def get_limits(data, unit_aspect_ratio=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:, 0], data[:, 1])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.close(fig)
    if unit_aspect_ratio:
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        diff = abs(xrange - yrange)
        if xrange > yrange:
            ylim = (ylim[0] - diff / 2, ylim[1] + diff / 2)
        else:
            xlim = (xlim[0] - diff / 2, xlim[1] + diff / 2)
    return xlim, ylim


def savefig(fig, path):
    fig.savefig(path, transparent=True)


def plot(plot_params):
    with LOCK:
        print(plot_params.attack, plot_params.projection_alg, time.time())
    os.makedirs(plot_params.outdir, exist_ok=True)
    with mpl_font_size(FONT_SIZE):
        height = float(FIG_SIZE * np.diff(plot_params.ylim) / np.diff(plot_params.xlim))
        fig = plt.figure(figsize=(plot_params.width, height))
        ax = fig.add_subplot(1, 1, 1)
        collections = []
        collection_classes = []
        for class_idx, indices_ in enumerate(plot_params.ground_truth_lookup):
            color = CMAP_DARK[class_idx]
            label = plot_params.classes[class_idx]
            if plot_params.approximate:
                scatter = ax.scatter(
                    plot_params.projections[indices_, 0], plot_params.projections[indices_, 1], c=[color], label=label)
                collections.append(scatter)
                collection_classes.append(class_idx)
            else:
                # Plot one point at a time, as this permits specifying a z-index for each point
                # (as opposed to having a per-class z-index, which would result in some classes taking
                # visual precedence over others). z-index is set as a function of the point's index in
                # the dataset, as the ordering of classes in the dataset is shuffled.
                # WARN: This is noticeably slower than the alternative.
                for idx in indices_:
                    scatter = ax.scatter(
                        plot_params.projections[idx, 0],
                        plot_params.projections[idx, 1],
                        c=[color], label=label, zorder=idx / len(plot_params.projections)
                    )
                    collections.append(scatter)
                    collection_classes.append(class_idx)
                    label = None
        ax.set_xlim(plot_params.xlim[0], plot_params.xlim[1])
        ax.set_ylim(plot_params.ylim[0], plot_params.ylim[1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
        fig.tight_layout()
        savefig(fig, os.path.join(plot_params.outdir, 'non_adversarial.pdf'))
        legend = ax.legend(**LEGEND_KWARGS)
        savefig(fig, os.path.join(plot_params.outdir, 'non_adversarial_with_legend.pdf'))
        legend.remove()
        # Change non-adversarial points from dark colors to light.
        for collection, class_idx in zip(collections, collection_classes):
            collection.set_color(CMAP_LIGHT[class_idx])
        for class_idx, indices in enumerate(plot_params.ground_truth_lookup):
            adv_class = plot_params.classes[class_idx]
            # z-index does not have to be specified. It's default value is higher than any used for
            # the non-adversarial points. This way, adversarial points will always appear above the
            # non-adversarial points.
            adv_points = ax.scatter(
                plot_params.adv_projections[indices, 0],
                plot_params.adv_projections[indices, 1],
                c=['black'],
                label='adversarial ' + adv_class
            )
            by_class_outdir = os.path.join(plot_params.outdir, 'non_adversarial_and_adversarial_by_class')
            os.makedirs(by_class_outdir, exist_ok=True)
            savefig(fig, os.path.join(
                by_class_outdir, f'non_adversarial_and_adversarial_by_class_{class_idx}_{adv_class}.pdf'))
            legend = ax.legend(**LEGEND_KWARGS)
            savefig(fig, os.path.join(
                by_class_outdir, f'non_adversarial_and_adversarial_by_class_with_legend_{class_idx}_{adv_class}.pdf'))
            legend.remove()
            adv_points.remove()
        plt.close(fig)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--num-jobs', type=int, default=max(16, os.cpu_count()))
    parser.add_argument(
        '--approximate', action='store_true', help='Don\'t handle z-index (runs faster)')
    parser.add_argument('--workspace', type=str, default='workspace')
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    with open(os.path.join(args.workspace, 'classes.json'), 'r') as f:
        classes = json.load(f)
    eval_correct = np.loadtxt(os.path.join(args.workspace, 'eval', 'correct.csv'), dtype=bool, delimiter=',')
    ground_truth = np.loadtxt(os.path.join(args.workspace, 'eval', 'ground_truth.csv'), delimiter=',', dtype=int)
    # Only correctly classified images were attacked and used for projections.
    ground_truth = ground_truth[eval_correct]
    ground_truth_lookup = [[] for _ in range(10)]
    for idx, x in enumerate(ground_truth):
        ground_truth_lookup[x].append(idx)
    plot_args_list = []
    for attack in ATTACKS:
        for projection_alg in PROJECTIONS:
            projections_path = os.path.join(args.workspace, 'projection', str(args.trial), projection_alg, attack)
            projections = np.loadtxt(os.path.join(projections_path, 'projections.csv'), delimiter=',')
            adv_projections = np.loadtxt(os.path.join(projections_path, 'adv_projections.csv'), delimiter=',')
            outdir = os.path.join(args.workspace, 'visualize', attack, projection_alg)
            xlim, ylim = get_limits(np.concatenate((projections, adv_projections)), unit_aspect_ratio=False)
            plot_params = PlotParams(
                attack=attack,
                projection_alg=projection_alg,
                classes=classes,
                ground_truth_lookup=ground_truth_lookup,
                outdir=outdir,
                projections=projections,
                adv_projections=adv_projections,
                xlim=xlim,
                ylim=ylim,
                width=FIG_SIZE,
                approximate=args.approximate,
            )
            plot_args_list.append(plot_params)
    with multiprocessing.Pool() as pool:
        pool.map_async(plot, plot_args_list)
        pool.close()
        pool.join()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
