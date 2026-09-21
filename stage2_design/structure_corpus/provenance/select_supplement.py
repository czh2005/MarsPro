from pathlib import Path
import csv
import json
import hashlib
import collections
import math
import statistics

OUT=Path(__file__).resolve().parents[1]
AUDIT=OUT/'核查资料'
DATA=Path('D:/火星蛋白/AAA数据集')
CURRENT=DATA/'AAAA并集来源0_v3_20260910'
OLD=Path('D:/火星蛋白/mpnn数据集/train_with_esmfold2.csv')
META=DATA/'第二阶段来源区片的数据。/MarsLikePro_4万条元数据侧表_v1_20260909.csv'
TASKS=['cold','desiccation','oxidative','perchlorate','radiation','salt']
CONFIG={'seed':20260914,'target_positive_total':10000,'min_length':51,'max_length':1024,'labels':'exact current v3; unknown preserved','group_isolation':'existing 10000 and heldout sequence/id/ID50; known relationship components also excluded','selection':'greedy minimum six-label count variance; source/genus/family/length diversity within label-pattern ties; improving one-record label-pattern swaps','diversity_weights':{'source':4,'genus':1,'pfam':0.5,'coarse':0.4,'length':0.3},'claim':'deterministic heuristic, no global optimality claim'}

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows,fields=None):
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def tokens(s):return tuple(sorted(set(x.strip() for x in s.split('|') if x.strip())))
def mask(r):return tuple(int(r[t]=='1') for t in TASKS)
def anypos(r):return any(r[t]=='1' for t in TASKS)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lengthbin(n):return next((str(x) for x in [100,200,300,400,600,800,1024] if n<=x),'long')

def main():
    outputs=[OUT/'待预测_2021条.csv',OUT/'正例总表_10000条.csv',OUT/'全部结构台账_12021条.csv']
    if any(p.exists() for p in outputs):raise RuntimeError('Outputs exist; do not overwrite a verified batch.')
    meta={r['sequence_id']:r for r in read(META)}
    train=read(CURRENT/'train.csv');idx={r['sequence_id']:r for r in train}
    old=read(OLD);retained=[r for r in old if anypos(r)];excluded=[r for r in old if not anypos(r)]
    need=CONFIG['target_positive_total']-len(retained)
    assert need==len(excluded) and need>0
    assert len(idx)==len(train)
    for r in old:
        assert r['sequence']==idx[r['sequence_id']]['sequence']
        assert all(r[t]==idx[r['sequence_id']][t] for t in TASKS)
    held=read(CURRENT/'validation.csv')+read(CURRENT/'test.csv')
    forbidden=old+held
    exclude_sets={k:{r[k] for r in forbidden} for k in ['sequence_id','sequence','id50_cluster_id']}
    excluded_rel={meta[r['sequence_id']]['relation_component_id'] for r in forbidden}-{''}
    pool=[];reasons=collections.Counter()
    for r in train:
        sid=r['sequence_id'];m=meta[sid]
        why=next((k for k,v in exclude_sets.items() if r[k] in v),'')
        if not why and not anypos(r):why='no_positive'
        if not why and not CONFIG['min_length']<=len(r['sequence'])<=CONFIG['max_length']:why='length'
        if not why and not set(r['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY'):why='nonstandard_residue'
        if not why and m['relation_component_id'] and m['relation_component_id'] in excluded_rel:why='known_relation_overlap'
        if why:reasons[why]+=1
        else:pool.append(r)
    pool.sort(key=lambda r:hashlib.sha256((str(CONFIG['seed'])+r['sequence_id']).encode()).hexdigest())
    assert len(pool)>=need
    features={}
    for r in pool+old:
        sid=r['sequence_id'];m=meta[sid]
        features[sid]={'source':tokens(m['scientific_source_ids']),'genus':tokens(m['genus_names']),'pfam':tokens(m['primary_pfam_accession']),'coarse':(m['coarse_import_buckets'] or 'unknown_source',),'length':(lengthbin(len(r['sequence'])),)}
    counters={k:collections.Counter() for k in CONFIG['diversity_weights']}
    def update(r,sign):
        for key,vals in features[r['sequence_id']].items():
            for v in vals:counters[key][v]+=sign/len(vals)
    for r in retained:update(r,1)
    def score(r):
        f=features[r['sequence_id']]
        return sum(CONFIG['diversity_weights'][k]*sum(1/math.sqrt(max(0,counters[k][v])+1) for v in vals)/len(vals) for k,vals in f.items() if vals)
    groups=collections.defaultdict(list)
    for i,r in enumerate(pool):groups[mask(r)].append(i)
    active=set(range(len(pool)));selected=[];selected_set=set()
    groups_used={'id50_cluster_id':set(),'relation_component_id':set()}
    def keyrel(r):return meta[r['sequence_id']]['relation_component_id']
    def compatible(r):return r['id50_cluster_id'] not in groups_used['id50_cluster_id'] and (not keyrel(r) or keyrel(r) not in groups_used['relation_component_id'])
    counts=[sum(r[t]=='1' for r in retained) for t in TASKS]
    def variance(v):return sum(x*x for x in v)-sum(v)**2/len(v)
    for step in range(need):
        opts=[]
        for pat,indices in groups.items():
            candidates=[i for i in indices if i in active and compatible(pool[i])]
            if candidates:opts.append((variance([x+y for x,y in zip(counts,pat)]),pat,candidates))
        if not opts:raise RuntimeError('No compatible candidates')
        best=min(x[0] for x in opts)
        choices=[i for value,pat,indices in opts if abs(value-best)<1e-7 for i in indices]
        chosen=max(choices,key=lambda i:(score(pool[i]),-i))
        r=pool[chosen];selected.append(chosen);selected_set.add(chosen);active.remove(chosen)
        update(r,1);counts=[x+y for x,y in zip(counts,mask(r))]
        groups_used['id50_cluster_id'].add(r['id50_cluster_id'])
        if keyrel(r):groups_used['relation_component_id'].add(keyrel(r))
    # Local pattern exchanges improve balance without changing sample count.
    swaps=0
    for iteration in range(500):
        selected_groups=collections.defaultdict(list)
        for i in selected_set:selected_groups[mask(pool[i])].append(i)
        available_groups={p:[i for i in inds if i not in selected_set and compatible(pool[i])] for p,inds in groups.items()}
        basevar=variance(counts);bestmove=None
        for a,removed in selected_groups.items():
            for b,added in available_groups.items():
                if not added:continue
                after=[x-y+z for x,y,z in zip(counts,a,b)]
                improvement=basevar-variance(after)
                if improvement>1e-7 and (bestmove is None or improvement>bestmove[0]+1e-7):bestmove=(improvement,a,b,removed,added,after)
        if bestmove is None:break
        _,a,b,removed,added,after=bestmove
        take=max(added,key=lambda i:(score(pool[i]),-i));drop=min(removed,key=lambda i:(score(pool[i]),i))
        for i,sign in [(drop,-1),(take,1)]:update(pool[i],sign)
        selected_set.remove(drop);selected_set.add(take)
        counts=after;swaps+=1
        groups_used={'id50_cluster_id':{pool[i]['id50_cluster_id'] for i in selected_set},'relation_component_id':{keyrel(pool[i]) for i in selected_set}-{''}}
    chosen=[pool[i] for i in sorted(selected_set)]
    # Equal-pattern exchanges can add source coverage without affecting label counts.
    diversity_swaps=0
    for pat,indices in groups.items():
        selected_pattern=[i for i in selected_set if mask(pool[i])==pat]
        for _ in range(min(len(selected_pattern),100)):
            available=[i for i in indices if i not in selected_set and compatible(pool[i])]
            if not available or not selected_pattern:break
            drop=min(selected_pattern,key=lambda i:(score(pool[i]),i))
            take=max(available,key=lambda i:(score(pool[i]),-i))
            if score(pool[take])<=score(pool[drop])*1.05:break
            update(pool[drop],-1);update(pool[take],1)
            selected_set.remove(drop);selected_set.add(take)
            selected_pattern.remove(drop);selected_pattern.append(take)
            groups_used={'id50_cluster_id':{pool[i]['id50_cluster_id'] for i in selected_set},'relation_component_id':{keyrel(pool[i]) for i in selected_set}-{''}}
            diversity_swaps+=1
    chosen=[pool[i] for i in sorted(selected_set)]
    assert len(chosen)==need
    for k in exclude_sets:
        assert len({r[k] for r in chosen})==need
        assert not ({r[k] for r in chosen}&exclude_sets[k])
    relations=[keyrel(r) for r in chosen if keyrel(r)]
    assert len(relations)==len(set(relations)) and not (set(relations)&excluded_rel)
    enrich_keys=['scientific_source_ids','coarse_import_buckets','organism_names','genus_names','taxids','relation_component_id','primary_pfam_accession']
    def enrich(r,role):
        out=dict(r);m=meta[r['sequence_id']]
        for key in enrich_keys:out[key]=m[key]
        out['label_version']='source_zero_union_v1_20260910'
        out['positive_set_role']=role
        out['has_any_positive_label']=str(int(anypos(r)))
        out['label_coverage']=str(sum(r[t] in ('0','1') for t in TASKS))
        for t in TASKS:out[t+'_mask']=str(int(r[t] in ('0','1')))
        return out
    start=max(int(r['prediction_id']) for r in old)+1
    newrows=[]
    for j,r in enumerate(chosen,start):
        n=enrich(r,'supplement_pending');n.update({'prediction_id':str(j),'sequence_length':len(r['sequence']),'selection_batch':'supplement_v3_20260914','prediction_status':'pending','structure_file':'','cif_file':'','plddt_mean_0_1':'','ptm':'','planned_structure_file':f'structures/{j}.pdb'})
        newrows.append(n)
    retained_rows=[enrich(r,'existing_positive') for r in retained]
    excluded_rows=[enrich(r,'retained_nonpositive_not_in_positive_training_set') for r in excluded]
    positive=retained_rows+newrows
    ledger=[enrich(r,'existing_positive' if anypos(r) else 'retained_nonpositive_not_in_positive_training_set') for r in old]+newrows
    assert len(positive)==CONFIG['target_positive_total'] and all(anypos(r) for r in positive)
    write(outputs[0],newrows);write(outputs[1],positive);write(outputs[2],ledger)
    write(AUDIT/'旧候选暂不纳入正例训练集_2021条.csv',excluded_rows)
    with (OUT/'待预测_2021条.fasta').open('w',encoding='ascii') as f:
        for r in newrows:f.write('>'+r['prediction_id']+'|'+r['sequence_id']+'\n'+r['sequence']+'\n')
    def describe(records):
        src=collections.Counter();genera=set();pfam=set();buckets=collections.Counter()
        for r in records:
            m=meta[r['sequence_id']];ss=tokens(m['scientific_source_ids'])
            for s in ss:src[s]+=1/len(ss)
            genera.update(tokens(m['genus_names']));pfam.update(tokens(m['primary_pfam_accession']))
            buckets[m['coarse_import_buckets'] or 'unknown_source']+=1
        lengths=[len(r['sequence']) for r in records]
        return {'rows':len(records),'positive_counts':{t:sum(r[t]=='1' for r in records) for t in TASKS},'unique_ID50':len({r['id50_cluster_id'] for r in records}),'scientific_source_ids':len(src),'source_ids':sorted(src),'unknown_source_rows':sum(not meta[r['sequence_id']]['scientific_source_ids'] for r in records),'source_fractional_counts':dict(src),'source_fractional_hhi':sum((v/sum(src.values()))**2 for v in src.values()) if src else None,'genera':len(genera),'pfam_ids':len(pfam),'coarse_combinations':dict(buckets),'length':{'min':min(lengths),'max':max(lengths),'mean':statistics.mean(lengths),'median':statistics.median(lengths)}}
    stats={'before':describe(retained),'added':describe(newrows),'after':describe(positive),'eligible_pool':describe(pool)}
    table=[{'task':t,'原有正例':stats['before']['positive_counts'][t],'新增正例':stats['added']['positive_counts'][t],'补齐后正例':stats['after']['positive_counts'][t],'候选池可用正例':stats['eligible_pool']['positive_counts'][t]} for t in TASKS]
    write(OUT/'六标签与来源统计.csv',table)
    new_source=sorted(set(stats['after']['source_ids'])-set(stats['before']['source_ids']))
    manifest={'config':CONFIG,'selected_count':need,'pool_filter_exclusions':dict(reasons),'balance_swaps':swaps,'diversity_swaps':diversity_swaps,'stats':stats,'new_source_ids':new_source,'note':'source identifiers are not necessarily independent studies; metadata labels ignored; no sequence modified; no prediction launched','input_hashes':{str(p):digest(p) for p in [OLD,META]+[CURRENT/f'{s}.csv' for s in ['train','validation','test']]},'output_hashes':{p.name:digest(p) for p in outputs},'source_count_unit':'source IDs with fractional allocation for overlapping IDs; not endpoint or study count','replay':'fresh output directory required; fixed seed and deterministic sort'}
    (AUDIT/'selection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selected':need,'pool':len(pool),'stats':{k:{x:v for x,v in value.items() if x not in ['source_ids','source_fractional_counts','coarse_combinations']} for k,value in stats.items()},'new_sources':new_source,'swaps':[swaps,diversity_swaps]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
