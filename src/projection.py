import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from umap.parametric_umap import ParametricUMAP

from utils import ATTACKS, PROJECTIONS, set_seed


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    eval_correct = np.loadtxt(os.path.join(args.workspace, 'eval', 'correct.csv'), dtype=bool, delimiter=',')
    representations = np.loadtxt(os.path.join(args.workspace, 'eval', 'representations.csv'), delimiter=',')
    # Limit to images that were correctly classified initially.
    # This was already done earlier for the adversarial images.
    representations = representations[eval_correct]
    base_outdir = os.path.join(args.workspace, 'projection')
    os.makedirs(base_outdir, exist_ok=True)
    with open(os.path.join(base_outdir, 'meta.json'), 'w') as f:
        json.dump({'num_trials': args.num_trials}, f, indent=2)
    for trial in range(args.num_trials):
        if args.skip_existing and os.path.exists(os.path.join(base_outdir, str(trial))):
            continue
        seed = args.seed + trial
        for projection_alg in PROJECTIONS:
            print(trial, projection_alg, time.time())
            set_seed(seed)
            oos_model = None
            if projection_alg == 'pca_oos':
                oos_model = PCA(n_components=2, random_state=seed)
            elif projection_alg == 'umap_oos':
                oos_model = UMAP(n_components=2, n_jobs=-1, random_state=seed)
            elif projection_alg == 'parametric_umap_oos':
                oos_model = ParametricUMAP(n_components=2, random_state=seed)
            if oos_model is not None:
                oos_model.fit(representations)
                projections = oos_model.transform(representations)
            for attack in ATTACKS:
                set_seed(seed)
                print('  ', trial, projection_alg, attack, time.time())
                adv_repr_path = os.path.join(args.workspace, 'attack', attack, 'representations.csv')
                adv_representations = np.loadtxt(adv_repr_path, delimiter=',')
                if oos_model is not None:
                    adv_projections = oos_model.transform(adv_representations)
                else:
                    X = np.concatenate((representations, adv_representations))
                    if projection_alg == 'pca':
                        model = PCA(n_components=2, random_state=seed)
                    elif projection_alg == 'umap':
                        model = UMAP(n_components=2, n_jobs=-1, random_state=seed)
                    elif projection_alg == 'parametric_umap':
                        model = ParametricUMAP(n_components=2, random_state=seed)
                    elif projection_alg == 'tsne':
                        model = TSNE(n_components=2, n_jobs=-1, random_state=seed)
                    else:
                        raise RuntimeError('Unsupported projection: ' + projection_alg)
                    X = model.fit_transform(X)
                    projections = X[:len(representations)]
                    adv_projections = X[len(representations):]
                outdir = os.path.join(base_outdir, str(trial), projection_alg, attack)
                os.makedirs(outdir, exist_ok=True)
                np.savetxt(os.path.join(outdir, 'projections.csv'), projections, delimiter=',', fmt='%f')
                np.savetxt(os.path.join(outdir, 'adv_projections.csv'), adv_projections, delimiter=',', fmt='%f')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
