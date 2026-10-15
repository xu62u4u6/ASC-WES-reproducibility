"""Development-only final candidate fit and feature-named deployment inference."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .model_evaluation import MODALITIES, fit_standalone
from .model_learners import COMBINATIONS
from .model_stacking import fit_stacking


@dataclass
class FinalBranch:
    name: str
    modality: str
    features: tuple[str, ...]
    estimator: object


@dataclass
class FinalModel:
    name: str
    branches: tuple[FinalBranch, ...]
    meta_estimator: object | None
    training_count: int
    classes: tuple[str,str] = ('LUAD','LUSC')
    cluster_policy: str = 'frozen historical clusters include holdout-derived information'

    def predict(self, matrices):
        """Matrices are sample-by-feature DataFrames; align genes by column name.

        No probability calibration is implied. Outputs are probability_LUSC and
        predicted class code (0 LUAD, 1 LUSC). No patient records are saved.
        """
        order=None;probabilities=[];last_x=None
        with threadpool_limits(limits=1):
            for branch in self.branches:
                frame=matrices[branch.modality]
                if not isinstance(frame,pd.DataFrame) or frame.columns.has_duplicates or frame.index.has_duplicates:
                    raise ValueError('Prediction requires uniquely keyed DataFrames')
                if order is not None and not frame.index.equals(order):
                    raise ValueError('Prediction sample order differs across modalities')
                order=frame.index
                if not set(branch.features).issubset(frame.columns):
                    raise ValueError('Required prediction features are missing')
                x=frame.loc[:,list(branch.features)].to_numpy(dtype=float)
                if not np.isfinite(x).all() or (branch.modality=='snv' and not np.isin(x,[0,1]).all()):
                    raise ValueError('Prediction requires finite values and binary SNV')
                probabilities.append(branch.estimator.predict_proba(x)[:,1]);last_x=x
            if self.meta_estimator is None:
                return probabilities[0],self.branches[0].estimator.predict(last_x)
            x=np.column_stack(probabilities)
            return self.meta_estimator.predict_proba(x)[:,1],self.meta_estimator.predict(x)


def fit_final_candidate(training, labels, candidate, *, diagnostics=None):
    """Pure fit API; caller must supply development rows and a CV-selected name.

    This low-level function does not establish CV completion. The production
    final-fit CLI must enforce that gate before invoking it. Holdout/ASC arrays
    and identifiers are neither accepted separately nor stored in the result.
    """
    if candidate not in set(MODALITIES)|set(COMBINATIONS): raise ValueError('Unknown candidate')
    if not isinstance(labels,pd.Series) or labels.index.has_duplicates:
        raise ValueError('Training labels must be a uniquely keyed Series')
    arrays={}
    for modality in MODALITIES.values():
        frame=training[modality]
        if not isinstance(frame,pd.DataFrame) or frame.columns.has_duplicates or not frame.index.equals(labels.index):
            raise ValueError('Training features/labels are not aligned')
        if not all(isinstance(name,str) and name for name in frame.columns):
            raise ValueError('Training feature names must be nonempty strings')
        arrays[modality]=frame.to_numpy(dtype=float)
    branches=[]
    if candidate in COMBINATIONS:
        fitted=fit_stacking(arrays['snv'],arrays['cnv'],labels.to_numpy(),seed=42)
        if diagnostics is not None:
            diagnostics.update(curves={k:v.copy(deep=True) for k,v in fitted.base.curves.items()},
                selected_counts={k:len(v) for k,v in fitted.base.selected.items()},
                selected_branches=list(COMBINATIONS[candidate]),training_count=len(labels))
        for name in COMBINATIONS[candidate]:
            modality='snv' if name.startswith('snv') else 'cnv'
            features=tuple(training[modality].columns[fitted.base.selected[name]])
            branches.append(FinalBranch(name,modality,features,fitted.base.models[name]))
        meta=fitted.meta_models[candidate]
    else:
        fitted=fit_standalone(arrays,labels.to_numpy(),seed=42)
        if diagnostics is not None:
            diagnostics.update(curves={},selected_counts={},selected_branches=[candidate],training_count=len(labels))
        modality=MODALITIES[candidate]
        features=tuple(training[modality].columns[fitted.selected[candidate]])
        branches.append(FinalBranch(candidate,modality,features,fitted.models[candidate]))
        meta=None
    # Deliberately discard OOF arrays, unused candidate models and row indices.
    return FinalModel(candidate,tuple(branches),meta,len(labels))
