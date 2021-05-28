import argparse
from collections import namedtuple
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from utils import ATTACKS, PROJECTIONS


ScoreRecord = namedtuple('Record', 'attack projection trial score')


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace')
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    eval_correct = np.loadtxt(os.path.join(args.workspace, 'eval', 'correct.csv'), dtype=bool, delimiter=',')
    ground_truth = np.loadtxt(os.path.join(args.workspace, 'eval', 'ground_truth.csv'), delimiter=',', dtype=int)
    # Limit to images that were correctly classified initially.
    # This was already done earlier for the adversarial images.
    ground_truth = ground_truth[eval_correct]
    score_records = []
    os.makedirs(args.workspace, exist_ok=True)
    with open(os.path.join(args.workspace, 'projection', 'meta.json')) as f:
        meta = json.load(f)
        num_trials = meta['num_trials']
    for attack in ATTACKS:
        adv_pred_path = os.path.join(args.workspace, 'attack', attack, 'pred.csv')
        adv_pred = np.loadtxt(adv_pred_path, delimiter=',', dtype=int)
        for projection_alg in PROJECTIONS:
            for trial in range(num_trials):
                # Log time
                print(attack, projection_alg, trial, time.time())
                projections_path = os.path.join(args.workspace, 'projection', str(trial), projection_alg, attack)
                projections = np.loadtxt(os.path.join(projections_path, 'projections.csv'), delimiter=',')
                adv_projections = np.loadtxt(os.path.join(projections_path, 'adv_projections.csv'), delimiter=',')
                nn_clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean', algorithm='brute')
                nn_clf.fit(projections, ground_truth)
                nn_pred = nn_clf.predict(adv_projections)
                score = (nn_pred == adv_pred).sum() / len(adv_pred)
                score_record = ScoreRecord(
                    attack=attack,
                    projection=projection_alg,
                    trial=trial,
                    score=score
                )
                score_records.append(score_record)
    scores_df = pd.DataFrame.from_records(score_records, columns=ScoreRecord._fields)
    scores_df.to_csv(os.path.join(args.workspace, 'scores.csv'), index=False)
    score_stats_df = scores_df.groupby(['attack', 'projection'], as_index=False).agg({'score': ['mean', 'std']})
    score_stats_df.columns = ['attack', 'projection', 'mean', 'std']
    score_stats_df = score_stats_df.pivot(index='projection', columns='attack', values=['mean', 'std'])
    score_stats_df = score_stats_df.reindex(ATTACKS, axis=1, level='attack')
    score_stats_df.columns = ['_'.join(col).strip() for col in score_stats_df.columns.values]
    score_stats_df.reset_index(inplace=True)
    score_stats_df['mean_mean'] = score_stats_df.loc[:, ['mean_' + attack for attack in ATTACKS]].mean(axis=1)
    score_stats_df.sort_values(by='mean_mean', ascending=False, inplace=True)
    score_stats_df.reset_index(drop=True, inplace=True)
    score_stats_df.to_csv(os.path.join(args.workspace, 'score_stats.csv'), index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
