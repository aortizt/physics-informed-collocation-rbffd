"""Independent CGF evidence audit and four monolithic verification replays.

Run from Spyder or Python after extracting the complete package. Original
evidence is read unchanged. Recomputed outputs are written to audit_replay/,
separate from the release analysis/. This is not a new production campaign.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import csv, hashlib, importlib.util, json, math, pathlib, sys, zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'audit_replay'
OUT.mkdir(exist_ok=True)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(name,rows):
    if rows:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

audit=[]
for key in ('collar','xm','vug'):
    directory=ROOT/'evidence'/key
    protocol=json.loads((directory/'protocol.json').read_text())
    frame=pd.read_csv(directory/'campaign_metrics.csv')
    records=list(csv.DictReader((directory/'scientific_manifest_sha256.csv').open()))
    failures=[]
    for row in records:
        rel=row.get('relative_path',row.get('path'))
        p=directory/rel
        size=row.get('size_bytes',row.get('bytes'))
        if not p.is_file() or digest(p)!=row['sha256'] or (size and p.stat().st_size!=int(size)):
            failures.append(rel)
    clean={k:v for k,v in protocol.items() if k!='protocol_hash'}
    computed=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    # Canonical serialization is separately checked using the supplied helper below.
    audit.append(dict(campaign=directory.name,rows=len(frame),cases=len(list((directory/'cases').glob('*.json'))),manifest_entries=len(records),manifest_matches=len(records)-len(failures),failures=failures,protocol_hash=protocol['protocol_hash'],independent_protocol_hash=computed,source_hashes={p.name:digest(p) for p in (directory/'source').glob('*.py')}))

collar=pd.read_csv(ROOT/'evidence/collar/campaign_metrics.csv')
xm=pd.read_csv(ROOT/'evidence/xm/campaign_metrics.csv')
vug=pd.read_csv(ROOT/'evidence/vug/campaign_metrics.csv')
pairs=[]; seedrows=[]
comparisons=[('PROTECTED_FINITE_COLLAR_HALTON','PROTECTED_NEAR_INTERFACE_SHEET'),('PROTECTED_FINITE_COLLAR_HALTON','PROTECTED_FINITE_COLLAR_RANDOM'),('PROTECTED_FINITE_COLLAR_HALTON','HALTON'),('PROTECTED_NEAR_INTERFACE_SHEET','HALTON'),('PROTECTED_GHG_EQUAL','HALTON'),('PROTECTED_GHG_NO_GREEN','PROTECTED_GHG_EQUAL'),('PROTECTED_GHG_NO_GRADIENT','PROTECTED_GHG_EQUAL'),('UNPROTECTED_GHG_EQUAL','PROTECTED_GHG_EQUAL')]
for problem,frame in collar.groupby('problem'):
    for a,b in comparisons:
        joined=frame[frame.method==a].merge(frame[frame.method==b],on=['budget','seed'],suffixes=('_a','_b'),validate='one_to_one')
        for metric in ['full_rmse','collar_rmse','coverage_fill_max','condition_proxy_equilibrated_1norm','charged_seconds']:
            ratio=joined[metric+'_a']/joined[metric+'_b']; difference=joined[metric+'_a']-joined[metric+'_b']
            p=float(wilcoxon(difference,alternative='less').pvalue) if np.any(difference!=0) else 1.
            logratio=np.log(ratio); blocks=joined.assign(logratio=logratio).groupby('seed').logratio.mean()
            seedp=float(wilcoxon(blocks,alternative='less',method='exact').pvalue) if np.any(blocks!=0) else 1.
            pairs.append(dict(problem=problem,challenger=a,reference=b,metric=metric,pairs=len(joined),median_ratio=float(np.median(ratio)),wins=int(np.sum(ratio<1)),pooled_one_sided_p=p,seed_block_one_sided_p=seedp,seed_blocks=len(blocks),seed_blocks_below_one=int(np.sum(blocks<0))))
            for seed,val in blocks.items(): seedrows.append(dict(problem=problem,challenger=a,reference=b,metric=metric,seed=int(seed),mean_log_ratio=float(val),geometric_ratio=float(np.exp(val))))
write_csv('collar_independent_pairs.csv',pairs);write_csv('seed_block_sensitivity.csv',seedrows)
write_csv('xm_gate_summary.csv',[dict(problem=k[0],method=k[1],rows=len(g),admissible=int(g.gate_admissible.sum()),conservation_failures=int((~g.gate_conservation).sum()),range_failures=int((~g.gate_overshoot).sum())) for k,g in xm.groupby(['problem','method'])])

source=ROOT/'evidence/vug/source'
sys.path.insert(0,str(source))
import CGF_Replay_Common_PHS_RBFFD as core
import CGF_Vug_Two_Way_Coupling_Qualification as vc
for item in audit:
    protocol=json.loads((ROOT/'evidence'/item['campaign']/'protocol.json').read_text())
    item['canonical_protocol_hash_matches']=core.protocol_hash({k:v for k,v in protocol.items() if k!='protocol_hash'})==protocol['protocol_hash']

patch=[]
for problem in vc.make_problems():
    n=360; seed=11; nb=max(8,math.ceil(math.sqrt(n)*.72)); ni=max(12,math.ceil(math.sqrt(n)*.9))
    hpts=core.global_halton(problem,n,seed)
    pilot_result=core.solve_monolithic(problem,hpts,nb,ni,81,.08)
    pilot=core.pilot_gradient_from_result(pilot_result)
    for name in vc.ALLOCATIONS:
        points,_=vc.allocation(name,problem,n,seed,pilot)
        result=pilot_result if name=='HALTON' else core.solve_monolithic(problem,points,nb,ni,81,.08)
        nodal_error=np.r_[result.values_minus-problem.exact(result.minus.nodes,np.full(len(result.values_minus),-1)),result.values_plus-problem.exact(result.plus.nodes,np.full(len(result.values_plus),1))]
        original=vug[(vug.problem==problem.name)&(vug.allocation==name)&(vug.budget==n)&(vug.seed==seed)].iloc[0]
        patch.append(dict(problem=problem.name,allocation=name,budget=n,seed=seed,nodal_linf=float(np.max(np.abs(nodal_error))),nodal_rmse=float(np.sqrt(np.mean(nodal_error**2))),linear_reconstruction_full_rmse=result.metrics['full_rmse'],recorded_monolithic_full_rmse=float(original.monolithic_full_rmse),replay_full_difference=result.metrics['full_rmse']-float(original.monolithic_full_rmse)))
        np.savez_compressed(OUT/f'patch_state_{problem.name}_{name}.npz',nodes_minus=result.minus.nodes,nodes_plus=result.plus.nodes,values_minus=result.values_minus,values_plus=result.values_plus,interface_points=result.minus.interface_points,interface_normals=result.minus.interface_normals_out)
write_csv('quadratic_patch_audit.csv',patch)

# Exact-field integration sensitivity for the oblique fixture: this tests the audit, not allocation.
quad=[]
for params in [('X_TRAIN',25.,(.5,.5),1.,50.),('X_HOLDOUT',40.,(.52,.47),1.,20.)]:
    name,angle,center,km,kp=params
    prob=core.oblique_interface_problem(name,angle,center,km,kp)
    theta=math.radians(angle); normal=np.array([math.cos(theta),math.sin(theta)]); tangent=np.array([-normal[1],normal[0]])
    lower,upper=prob.metadata['segment_parameter']; wave=2*np.pi/(upper-lower)
    for q in (30,120,480):
        coord=(np.arange(q)+.5)/q; xx,yy=np.meshgrid(coord,coord,indexing='xy'); pts=np.c_[xx.ravel(),yy.ravel()]
        vol=float(np.mean(prob.forcing(pts)))
        bd=0.
        for points,outward in [(np.c_[coord,coord*0],np.array([0,-1.])),(np.c_[coord*0+1,coord],np.array([1.,0])),(np.c_[coord,coord*0+1],np.array([0,1.])),(np.c_[coord*0,coord],np.array([-1.,0]))]:
            s=(points-np.asarray(center))@tangent
            flux=.7*normal[None,:]-(prob.conductivity(points)*.18*wave*np.sin(wave*(s-lower)))[:,None]*tangent[None,:]
            bd+=float(np.mean(flux@outward))
        quad.append(dict(problem=name,quadrature=q,exact_forcing_integral=vol,exact_outward_conormal_integral=bd,exact_field_relative_balance_defect=abs(vol+bd)/max(abs(vol),abs(bd),1e-12)))
write_csv('oblique_exact_quadrature_audit.csv',quad)
(OUT/'archive_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({'archive_checks': [{k:a[k] for k in ('campaign','rows','manifest_entries','manifest_matches','canonical_protocol_hash_matches')} for a in audit], 'quadratic_replays':len(patch), 'output_directory':str(OUT)},indent=2))
