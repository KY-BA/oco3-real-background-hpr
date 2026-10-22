"""Recalculate manuscript metrics from released predictions; no model inference."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(predictions, references):
    bias = [predictions[p] - references[p] for p in sorted(references)]
    ae = [abs(v) for v in bias]
    ape = [100.0 * abs(predictions[p] - references[p]) / references[p]
           for p in sorted(references)]
    return dict(MAE=statistics.mean(ae), MAPE_pct=statistics.mean(ape),
                Median_AE=statistics.median(ae), Median_APE_pct=statistics.median(ape),
                Mean_bias=statistics.mean(bias))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/reproduced_metrics')
    parser.add_argument('--predictions', type=Path, default=ROOT / 'results/reference/seed_predictions.csv')
    parser.add_argument('--tolerance', type=float, default=1e-9,
                        help='Absolute metric tolerance; CPU/GPU inference may require a small tolerance.')
    parser.add_argument('--report-only', action='store_true',
                        help='Write the comparison even when values differ; do not claim equality.')
    args = parser.parse_args()
    if args.tolerance < 0 or not math.isfinite(args.tolerance):
        parser.error('--tolerance must be finite and nonnegative.')
    protocol = json.loads((ROOT / 'manifests/paper_protocol.json').read_text(encoding='utf-8'))
    with args.predictions.open(encoding='utf-8', newline='') as h:
        rows = list(csv.DictReader(h))
    with (ROOT / 'manifests/evaluation_scenes.csv').open(encoding='utf-8', newline='') as h:
        scene_rows = list(csv.DictReader(h))
    refs = {p: float(v) for p, v in protocol['annual_emission_references_mt_co2_yr'].items()}
    seeds = set(protocol['seeds'])
    models = protocol['models']
    expected_scenes = {(r['plant'], r['scene']) for r in scene_rows}
    if len(expected_scenes) != 33 or len(refs) != 8:
        raise ValueError('Expected exactly 33 distinct scenes and eight plants.')
    values = {}
    for row in rows:
        key = (row['model'], int(row['seed']), row['plant'], row['scene'])
        if key in values:
            raise ValueError('Duplicate prediction key: %r' % (key,))
        if key[0] not in models or key[1] not in seeds or key[2:] not in expected_scenes:
            raise ValueError('Unexpected prediction key: %r' % (key,))
        value = float(row['predicted_emission'])
        if not math.isfinite(value):
            raise ValueError('Non-finite prediction: %r' % (key,))
        values[key] = value
    expected_keys = {(m, s, p, f) for m in models for s in seeds for p, f in expected_scenes}
    if set(values) != expected_keys:
        raise ValueError('Missing predictions; all models must use the same scenes and seeds.')

    plant_estimates = {}
    plant_rows, scene_out, ensemble_out, seed_out, seed_summary, group_out = [], [], [], [], [], []
    for model in models:
        scene_means = {(p, f): statistics.mean(values[(model, s, p, f)] for s in seeds)
                       for p, f in expected_scenes}
        estimates = {p: statistics.mean(v for (plant, f), v in scene_means.items() if plant == p)
                     for p in refs}
        plant_estimates[model] = estimates
        ensemble_out.append(dict(model=model, **metrics(estimates, refs)))
        for (plant, scene), value in sorted(scene_means.items()):
            scene_out.append(dict(model=model, plant=plant, scene=scene, ensemble_prediction=value))
        for plant in sorted(refs):
            bias = estimates[plant] - refs[plant]
            plant_rows.append(dict(model=model, plant=plant,
                                   n_scenes=sum(p == plant for p, _ in expected_scenes),
                                   annual_reference=refs[plant], prediction=estimates[plant],
                                   bias=bias, absolute_error=abs(bias),
                                   absolute_percentage_error=100.0 * abs(bias) / refs[plant]))
        seed_metrics = []
        for seed in sorted(seeds):
            sp = {p: statistics.mean(values[(model, seed, pp, f)]
                                      for pp, f in expected_scenes if pp == p) for p in refs}
            sm = metrics(sp, refs)
            seed_metrics.append(sm)
            seed_out.append(dict(model=model, seed=seed, **sm))
        seed_summary.append(dict(model=model,
                                 MAE_mean=statistics.mean(x['MAE'] for x in seed_metrics),
                                 MAE_sample_sd=statistics.stdev(x['MAE'] for x in seed_metrics),
                                 MAPE_mean=statistics.mean(x['MAPE_pct'] for x in seed_metrics),
                                 MAPE_sample_sd=statistics.stdev(x['MAPE_pct'] for x in seed_metrics)))
        for group in ('Low', 'Medium', 'High'):
            subset = {p: a for p, a in refs.items()
                      if ('Low' if a <= 8 else 'Medium' if a <= 18 else 'High') == group}
            group_out.append(dict(model=model, group=group, n_plants=len(subset),
                                  **metrics({p: estimates[p] for p in subset}, subset)))

    # Exploratory appendix: apply the fixed rule AFTER plant-level aggregation.
    low, high = protocol['post_hoc_gate']['low'], protocol['post_hoc_gate']['high']
    hpr, cnn = plant_estimates['hpr'], plant_estimates['mixed_cnn']
    gate = {p: hpr[p] if hpr[p] < low or hpr[p] > high else cnn[p] for p in refs}
    gate_rows = [dict(plant=p, selected_model='hpr' if hpr[p] < low or hpr[p] > high else 'mixed_cnn',
                      prediction=gate[p], annual_reference=refs[p], bias=gate[p] - refs[p]) for p in sorted(refs)]
    gate_metric = dict(model='post_hoc_plant_gate', **metrics(gate, refs))
    expected = json.loads((ROOT / 'results/reference/expected_metrics.json').read_text(encoding='utf-8'))
    checks = []
    for observed in ensemble_out + [gate_metric]:
        for metric, value in expected[observed['model']].items():
            ok = math.isclose(observed[metric], value, rel_tol=0, abs_tol=args.tolerance)
            checks.append(dict(model=observed['model'], metric=metric, actual=observed[metric],
                               expected=value, passed=ok))
    same = all(c['passed'] for c in checks)

    out = args.output_dir.resolve()
    protected = (ROOT / 'results/reference').resolve()
    if out == protected or protected in out.parents:
        raise ValueError('Output must not overwrite the released reference results.')
    out.mkdir(parents=True, exist_ok=True)
    for name, table in [('ensemble_metrics.csv', ensemble_out), ('plant_metrics.csv', plant_rows),
                        ('scene_predictions.csv', scene_out), ('individual_seed_metrics.csv', seed_out),
                        ('individual_seed_summary.csv', seed_summary), ('group_metrics.csv', group_out),
                        ('post_hoc_gate_plants.csv', gate_rows), ('post_hoc_gate_metrics.csv', [gate_metric])]:
        write_csv(out / name, table)
    report = dict(status='PASS' if same else 'DIFFERENT', n_prediction_rows=len(rows), n_scenes=len(expected_scenes),
                  n_plants=len(refs), seeds=sorted(seeds), checks=checks,
                  interpretation='Agreement with annual emission references; not instantaneous ground truth.',
                  scope='Reaggregation of saved predictions, not rerun training or satellite inference.')
    (out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in ensemble_out + [gate_metric]:
        print('%-20s MAE %.8f  MAPE %.8f%%' % (row['model'], row['MAE'], row['MAPE_pct']))
    print(('PASS' if same else 'DIFFERENT') + ': 495 predictions, 33 scenes, 8 plants, 5 seeds. Output:', out)
    if not same:
        print('See verification.json for all differences. Reference predictions were not changed.')
        if not args.report_only:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
