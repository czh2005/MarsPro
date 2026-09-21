from pathlib import Path
import csv
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/火星蛋白/AAA数据集/AAAA并集来源0_v3_20260910')
TASKS = ['cold', 'desiccation', 'oxidative', 'perchlorate', 'radiation', 'salt']

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((ROOT/'核查资料/selection_manifest.json').read_text(encoding='utf-8'))
for path, expected in manifest['input_hashes'].items():
    assert sha(Path(path)) == expected, path
for name, expected in manifest['output_hashes'].items():
    assert sha(ROOT/name) == expected, name
train = {r['sequence_id']: r for r in read(DATA/'train.csv')}
held = read(DATA/'validation.csv') + read(DATA/'test.csv')
old = read(Path('D:/火星蛋白/mpnn数据集/train_with_esmfold2.csv'))
new = read(ROOT/'待预测_2021条.csv')
total = read(ROOT/'正例总表_10000条.csv')
ledger = read(ROOT/'全部结构台账_12021条.csv')
excluded = read(ROOT/'核查资料/旧候选暂不纳入正例训练集_2021条.csv')
positive = lambda r: any(r[t] == '1' for t in TASKS)
assert len(new) == len(excluded)
assert len(total) == len(old) and len(ledger) == len(old) + len(new)
assert all(positive(r) for r in total) and not any(positive(r) for r in excluded)
for key in ['sequence_id', 'sequence', 'id50_cluster_id']:
    assert len({r[key] for r in total}) == len(total), key
    assert len({r[key] for r in ledger}) == len(ledger), key
    assert not ({r[key] for r in new} & {r[key] for r in old + held}), key
for row in ledger:
    original = train[row['sequence_id']]
    assert row['sequence'] == original['sequence']
    for task in TASKS:
        assert row[task] == original[task]
        assert row[task+'_mask'] == str(int(original[task] in ('0', '1')))
old_idx = {r['sequence_id']: r for r in ledger}
for row in old:
    for field in ['prediction_id', 'structure_file', 'planned_structure_file', 'prediction_status', 'plddt_mean_0_1', 'ptm']:
        assert row[field] == old_idx[row['sequence_id']][field], field
for row in new:
    assert row['prediction_status'] == 'pending'
    assert all(not row[k] for k in ['structure_file', 'cif_file', 'plddt_mean_0_1', 'ptm'])
assert len({r['prediction_id'] for r in ledger}) == len(ledger)
lines = (ROOT/'待预测_2021条.fasta').read_text(encoding='ascii').splitlines()
assert len(lines) == 2 * len(new)
for i, row in enumerate(new):
    assert lines[2*i] == '>'+row['prediction_id']+'|'+row['sequence_id']
    assert lines[2*i+1] == row['sequence']
meta_path = next(Path(p) for p in manifest['input_hashes'] if '元数据侧表' in p)
meta = {r['sequence_id']: r for r in read(meta_path)}
relation = lambda rows: [meta[r['sequence_id']]['relation_component_id'] for r in rows if meta[r['sequence_id']]['relation_component_id']]
new_rel = relation(new)
assert len(new_rel) == len(set(new_rel))
assert not (set(new_rel) & set(relation(old + held)))
result = {
    'verified': True, 'new_rows': len(new), 'positive_total_rows': len(total),
    'ledger_rows': len(ledger), 'old_nonpositive_preserved': len(excluded),
    'new_ID50_clusters': len({r['id50_cluster_id'] for r in new}),
    'new_known_relationship_rows': len(new_rel),
    'positive_counts': {t: sum(r[t] == '1' for r in total) for t in TASKS},
    'input_hashes_unchanged': True, 'current_labels_preserved': True,
    'old_structure_references_preserved': True, 'new_predictions_launched': False,
    'limitation': 'Known component isolation only; missing relationship annotations do not establish independence. PDB quality not re-evaluated.'
}
(ROOT/'核查资料/independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
with (ROOT/'核查资料/来源覆盖对比.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['source_id', 'before_fractional_rows', 'added_fractional_rows', 'after_fractional_rows'])
    for source in manifest['stats']['after']['source_ids']:
        writer.writerow([source] + [manifest['stats'][stage]['source_fractional_counts'].get(source, 0) for stage in ['before', 'added', 'after']])
print(json.dumps(result, ensure_ascii=False, indent=2))
