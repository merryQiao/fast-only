"""Exact-layout frozen feature extension; no label or model-selection inputs."""
import torch


def snapshot_inputs(snapshot,groups):
    keys=torch.stack([z.key.transpose(1,2).flatten(2) for z in snapshot.layers],dim=1)
    values=torch.stack([z.value.transpose(1,2).flatten(2) for z in snapshot.layers],dim=1)
    group_ids=torch.stack(list(groups.values())).long().argmax(0)[None].expand(len(keys),-1)
    return dict(keys=keys,values=values,positions=snapshot.position_ids,group_ids=group_ids,valid=snapshot.valid_mask)


def check_layout(batch,template):
    for name in ['positions','group_ids','valid']:
        if not torch.equal(batch[name].cpu(),template[name].cpu()):raise ValueError(f'Frozen Reader layout differs: {name}')


def extend_bank(old,rows,fast,reader,baseline,donors):
    tokens=[r['token'] for r in rows];index={t:i for i,t in enumerate(tokens)}
    if (len(index)!=len(rows) or set(tokens)&set(old['tokens']) or set(donors)!=set(tokens)
            or any(r['split']!='val' or r['cohort']!='holdout' for r in rows)
            or len(fast)!=len(rows) or len(reader)!=len(rows) or len(baseline)!=len(rows)):
        raise ValueError('Expected disjoint fixed holdout feature rows')
    by_token={r['token']:r for r in rows}
    if any(s not in index or by_token[t]['route']==by_token[s]['route'] for t,s in donors.items()):raise ValueError('Holdout donors require distinct logs')
    r0=old['reader']['reference'][:1]
    if not torch.equal(old['reader']['reference'],r0.expand_as(old['reader']['reference'])):raise ValueError('Reference source is not constant')
    selected=torch.tensor([index[donors[t]] for t in tokens],device=reader.device)
    new_reader=dict(correct=reader,reference=r0.expand(len(rows),-1,-1),shuffled=reader[selected])
    all_tokens=old['tokens']+tokens
    ref=next(iter(old['sources']['reference'].values()))
    return dict(tokens=all_tokens,index={t:i for i,t in enumerate(all_tokens)},head=old['head'],
        fast=torch.cat((old['fast'],fast)),baseline=torch.cat((old['baseline'],baseline)),
        reader={m:torch.cat((old['reader'][m],v)) for m,v in new_reader.items()},
        sources={m:{**old['sources'][m],**{t:t if m=='correct' else ref if m=='reference' else donors[t] for t in tokens}}
                 for m in ['correct','reference','shuffled']})
