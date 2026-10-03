"""Robust final submission for a HIDDEN final ranking.

The public leaderboard showed that the LightGBM-imputed model's score moves by about 0.02 depending only on the
random seed (seed 527 scored better than seeds 528/529 on the public rows). That luck won't repeat on the hidden
rows, so this file averages 10 seeds, and the blend weight comes from evidence that covers all test rows:
    - cross-validation with test-like blanks: best around 0.35-0.6
    - the same, weighted to the test set's building/event mix (test inputs only): best around 0.75-0.8
    - public leaderboard scores: about 0.8
    -> weight 0.7 on the LightGBM model, 0.3 on v4 (costs < 0.002 on either cross-validation check).

Run from the project folder (platform environment):  python final_robust.py
"""
import numpy as np

import final_model as F

SEEDS = tuple(range(527, 537))       # 10 seeds; 527-529 are the ones already used
WEIGHT_NEW = 0.70

if __name__ == '__main__':
    train, test = F.load_data()
    v4 = F.v4_predictions(train, test, source='file')
    runs = np.vstack([F.new_model_predictions(train, test, seeds=(s,)) for s in SEEDS])
    np.save(F.ROOT / 'experiments' / 'lightgbm_model_10_seeds.npy', runs)
    spread = np.sqrt(np.mean((runs - runs.mean(axis=0)) ** 2))
    print(f'{len(SEEDS)} seeds: typical difference of one seed from the average {spread:.3f}')
    pred = F.blend(v4, runs.mean(axis=0), WEIGHT_NEW)
    F.write_submission(F.ROOT / 'candidate_final.csv', test, pred)
    print('wrote candidate_final.csv')
