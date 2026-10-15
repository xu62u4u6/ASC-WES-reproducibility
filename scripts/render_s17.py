"""Reproduce S17 notebook boxplots and package-backed similarity summaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from pymaftools.core.PairwiseMatrix import SimilarityMatrix

from asc_reproduction.cohort_inputs import read_prepared_table
from asc_reproduction.figure_style import (
    SUBTYPE_COLORS, plt, save_tiff_and_preview, setup_plotting,
)
from asc_reproduction.similarity_panels import (
    SimilarityPanel, draw_similarity_panel, sample_similarity,
)
from compare_s16_permutations import draw_summary


def render(output, *, data_root, s16_source):
    """Save raw pairs, original U tests and 0.5.0 subtype permutations."""
    setup_plotting()
    output.mkdir(parents=True, exist_ok=False)
    metadata = read_prepared_table(data_root, 'tumor', 'SNV').sample_metadata
    assert metadata.subtype.value_counts().to_dict() == {
        'LUAD': 89, 'ASC': 29, 'LUSC': 82,
    }
    specifications = [
        ('jaccard', 'SNV&INDEL freq 005 Jaccard', 'A'),
        ('smc', 'SNV&INDEL freq 005 SMC', 'B'),
        ('gene', 'CNV Gene-level', None),
        ('cluster', 'CNV Cluster-level', 'C'),
        ('arm', 'CNV Arm-level', None),
    ]
    images = {}
    tests = []
    order = ['LUAD–ASC', 'LUSC–ASC']
    for name, title, cached in specifications:
        if cached:
            matrix = pd.read_csv(
                s16_source / f'S16{cached}_matrix.csv', index_col=0,
            )
        else:
            table = read_prepared_table(data_root, 'tumor', f'CNV_{name}')
            assert table.columns.equals(metadata.index)
            matrix = pd.DataFrame(
                sample_similarity(pd.DataFrame(table), 'cosine'),
                index=table.columns, columns=table.columns,
            )
        assert matrix.index.equals(metadata.index)
        assert matrix.columns.equals(metadata.index)
        matrix.to_csv(output / f'{name}_matrix.csv')
        groups = metadata.subtype
        first = matrix.loc[groups.eq('LUAD'), groups.eq('ASC')]
        second = matrix.loc[groups.eq('ASC'), groups.eq('LUSC')]
        assert first.size == 89 * 29 and second.size == 29 * 82
        values = [first.to_numpy().ravel(), second.to_numpy().ravel()]
        data = pd.DataFrame({
            'similarity': np.concatenate(values),
            'pair': np.repeat(order, [len(v) for v in values]),
        })
        data.to_csv(output / f'{name}_pairs.csv', index=False)
        statistic, p = mannwhitneyu(*values, alternative='two-sided')
        tests.append({'feature': name, 'U': statistic, 'p': p})
        figure, axis = plt.subplots(figsize=(5, 5))
        sns.boxplot(
            data=data, x='pair', y='similarity', hue='pair', order=order,
            palette={order[0]: SUBTYPE_COLORS['LUAD'],
                     order[1]: SUBTYPE_COLORS['LUSC']},
            legend=False, ax=axis,
        )
        annotator = Annotator(
            axis, [(order[0], order[1])], data=data,
            x='pair', y='similarity', order=order,
        )
        annotator.configure(
            test='Mann-Whitney', text_format='star', loc='inside', verbose=0,
        )
        annotator.apply_and_annotate()
        axis.set(title=title, xlabel='', ylabel='Similarity')
        images[name] = save_tiff_and_preview(
            figure, output / name, plot_axis=axis,
        )
        plt.close(figure)
        if cached:
            continue
        similarity = SimilarityMatrix(matrix)
        means = similarity.get_mean_group_similarity(groups, ['LUAD', 'ASC', 'LUSC'])
        null = similarity.generate_permutation_list(
            groups, ['LUAD', 'ASC', 'LUSC'], 1000, random_state=42,
        )
        pvalues = similarity.calculate_group_similarity_pvalues(
            means, null, ['LUAD', 'ASC', 'LUSC'], tail='right',
        )
        np.save(output / f'{name}_null.npy', np.stack(null))
        for kind, data in [('mean', means), ('p', pvalues)]:
            data.to_csv(output / f'{name}_{kind}.csv')
            images[f'{name}_{kind}'] = draw_summary(
                data, 'Group Mean Similarity' if kind == 'mean'
                else 'Group Similarity P-values', output / f'{name}_{kind}',
                lower=-1 if kind == 'mean' else 0, pvalues=kind == 'p',
            )
        panel = SimilarityPanel(
            matrix.to_numpy(), metadata, 'cosine', len(table), False,
            f'{title} cosine Similarity',
        )
        figure = plt.figure(figsize=(8, 8.1))
        draw_similarity_panel(figure, figure.add_gridspec(1, 1)[0], panel)
        for axis in figure.axes:
            for patch in axis.patches:
                patch.set_alpha(0.7)
        images[f'{name}_matrix'] = save_tiff_and_preview(
            figure, output / f'{name}_matrix', plot_axis=figure.axes[0],
        )
        plt.close(figure)
    pd.DataFrame(tests).to_csv(output / 'mannwhitney.csv', index=False)
    (output / 'panels.json').write_text(json.dumps(images, indent=2))


def main():
    """Read explicit inputs without relying on a historical output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--s16-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.output, data_root=args.data_root, s16_source=args.s16_source)


if __name__ == '__main__':
    main()
