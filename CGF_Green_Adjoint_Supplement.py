#!/usr/bin/env python3
"""CGF Green/adjoint supplemental experiment, v1.0 (Spyder F5).

Run THIS file. The other three Python files are frozen dependencies.
This is a discrete enriched-defect adjoint heuristic, not a certified DWR
estimator or an explicitly computed continuum Green function. See README.
No exact interior solution values enter node selection. Manufactured exact
values enter prescribed boundary data and independent validation only.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, importlib.util, hashlib, json, time, argparse, shutil, traceback
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

PROFILE = 'production'  # F5 default. 'screen': 30 cases; 'production': 300.
OUTPUT_DIRECTORY = ''   # Empty means a separate results folder beside this file.
HERE = Path(__file__).resolve().parent
BASE_NAME = 'CGF_Adaptive_RBFFD_Frozen_Allocation_Comparison'
BASE_SHA = 'b6194956500a23981a92a8d5805bdb161a9e80d4c566e3b7218cda9a81ca8202'
if hashlib.sha256((HERE/(BASE_NAME+'.py')).read_bytes()).hexdigest() != BASE_SHA:
    raise RuntimeError('Frozen adaptive dependency changed; restore supplied copy.')
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location(BASE_NAME,HERE/(BASE_NAME+'.py'))
b = importlib.util.module_from_spec(spec); sys.modules[BASE_NAME]=b; spec.loader.exec_module(b)
c = b.core
METHODS = ('HALTON','FROZEN_GHG','ADAPT_RESIDUAL','DEFECT_ONLY','ADJOINT','GHG_ADJOINT')
# Fixed before supplemental outcomes. Existing fixtures are NOT unseen holdouts.
# Two added fixtures change material coefficients at the same interface geometry.
def fixtures():
    return [('S', b.gaussian_problem()),
            ('S', c.vertical_resistance_problem('S_TRAIN',.5,1.,100.)),
            ('S', c.vertical_resistance_problem('S_FIXED_GEOMETRY_K30',.5,1.,30.)),
            ('Q', b.nq.problems()[0]),
            ('Q', b.nq.make_problem('Q_FIXED_GEOMETRY_K20',(.5,.5),.24,20.))]

def objective_points(order):
    """Fixed regional mean: x in [.65,.90], y in [.35,.65]. Weights sum to 1.
    Gaussian quadrature normalized by region area; no selection by exact error.
    """
    x,w=np.polynomial.legendre.leggauss(order)
    xx,yy=np.meshgrid(.775+.125*x,.5+.15*x,indexing='ij')
    return np.c_[xx.ravel(),yy.ravel()],np.outer(w,w).ravel()/4

def interpolation(state,xy,side=None):
    side=state.problem.side(xy) if side is None else side
    rr=[];cc=[];dd=[];offset=0
    for cloud in (state.minus,state.plus):
        for i in np.flatnonzero(side==cloud.side):
            ix,w=c.local_phs_weights(cloud,xy[i],'value',stencil_size=b.STENCIL)
            rr.extend([int(i)]*len(ix));cc.extend(offset+ix);dd.extend(w)
        offset+=len(cloud.nodes)
    return coo_matrix((dd,(rr,cc)),shape=(len(xy),offset)).tocsr()

def values(state):return np.r_[state.values_minus,state.values_plus]

def enriched_system(problem,n,seed,family):
    """Independent 2N cloud, doubled-budget anchor rule, equilibrated A.
    No enriched primal solve is needed for the production indicator.
    """
    pts=c.global_halton(problem,2*n,seed+810173)
    nb,ni=b.boundary_counts(family,2*n)
    minus,plus,raw,rhs=c.assemble_monolithic(problem,pts,nb,ni,b.STENCIL)
    norm=np.sqrt(np.asarray(raw.multiply(raw).sum(axis=1)).ravel())
    if np.any(norm<=0) or not np.all(np.isfinite(norm)):raise ArithmeticError('Invalid row')
    A=(diags(1/norm)@raw).tocsc();lu=splu(A)
    return SimpleNamespace(problem=problem,minus=minus,plus=plus,matrix=A,
                           rhs=rhs/norm,lu=lu)

def defect_indicator(problem,pilot,n,seed,family,weighted):
    fine=enriched_system(problem,n,seed,family)
    xy=np.vstack((fine.minus.nodes,fine.plus.nodes))
    sides=np.r_[np.full(len(fine.minus.nodes),-1),np.full(len(fine.plus.nodes),1)]
    T=interpolation(pilot,xy,sides)
    transferred=T@values(pilot);r=fine.rhs-fine.matrix@transferred
    qp,qw=objective_points(12);j=np.asarray(interpolation(fine,qp).T@qw).ravel()
    z=fine.lu.solve(j,trans='T') if weighted else np.ones(len(r))
    eta=np.abs(z*r)
    # Assembly order equals minus node order then plus node order.
    # Interface continuity and flux contributions are summed, then made
    # available to BOTH materials. They are never discarded as zero PDE rows.
    m=len(fine.minus.nodes);im=fine.minus.interface;ip=m+fine.plus.interface
    interface_eta=eta[im]+eta[ip];eta[im]=interface_eta;eta[ip]=interface_eta
    info=dict(defect_points=xy,defect_sides=sides,scaled_defect=r,adjoint=z,
              row_indicator=eta,objective_vector=j,transferred_pilot=transferred,
              enriched_rhs=fine.rhs, enriched_matrix_data=fine.matrix.data,
              enriched_matrix_indices=fine.matrix.indices,
              enriched_matrix_indptr=fine.matrix.indptr,
              enriched_matrix_shape=np.array(fine.matrix.shape))
    meta=dict(enriched_dof=len(r),adjoint_solves=int(weighted),
              adjoint_relative_residual=float(np.linalg.norm(fine.matrix.T@z-j)/max(np.linalg.norm(j),1e-15)) if weighted else 0.,
              signed_defect_estimate=float(z@r) if weighted else 0.,
              abs_interface_contribution=float(interface_eta.sum()))
    def score(x):
        out=np.zeros(len(x));sx=problem.side(x)
        for cloud in (fine.minus,fine.plus):
            use=sx==cloud.side;known=sides==cloud.side
            if np.any(use):out[use]=b.shepard(x[use],xy[known],eta[known])
        return out
    return score,meta,info

def normalize(a):
    a=np.maximum(np.asarray(a),0.)
    if not np.all(np.isfinite(a)):raise ArithmeticError('Nonfinite indicator')
    return a/max(float(a.mean()),1e-30) if a.max()>0 else np.ones(len(a))

def allocate_supplement(problem,pilot,n,seed,nb,ni,score,method):
    # Same remesher and candidate pool for all three new arms.
    x=c.global_halton(problem,max(400,4*n),seed+620321)
    influence=normalize(score(x))
    if method=='GHG_ADJOINT':
        gradient=normalize(c.pilot_gradient_from_result(pilot)(x))
        density=.45+.275*influence+.275*gradient
    else:density=.45+.55*influence
    # In 2D, spacing scales as density**(-1/2). The frozen remesher enforces
    # bounded spacing contrast, exact N, anchors and minimum material counts.
    spacing=1/np.sqrt(density)
    points,remesh=b.remesh(problem,n,seed,nb,ni,(x,spacing))
    return points,dict(allocation_probe=x,allocation_density=density,**remesh)

def execute(problem,family,method,n,seed,cfg,folder):
    folder.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();nb,ni=b.boundary_counts(family,n);arrays={};stages=[]
    def solve(points,label):
        state=b.solve(problem,points,nb,ni)
        arrays.update(b.state_arrays(state,label+'_'))
        stages.append(dict(stage=label,cumulative_seconds=time.perf_counter()-start))
        return state
    info=dict(adjoint_solves=0,enriched_dof=0,adjoint_relative_residual=0.,
              signed_defect_estimate=0.,abs_interface_contribution=0.)
    indicator_seconds=0.
    if method=='ADAPT_RESIDUAL':
        points,_=b.remesh(problem,n,seed,nb,ni);state=solve(points,'initial')
        for cycle in range(3):
            t=time.perf_counter();known,hist=b.update_spacing(problem,state,n,seed,cycle,method)
            indicator_seconds+=time.perf_counter()-t
            arrays.update({f'cycle{cycle}_'+k:np.asarray(v) for k,v in hist.items()})
            points,_=b.remesh(problem,n,seed,nb,ni,known);state=solve(points,f'cycle{cycle+1}')
    else:
        points=c.global_halton(problem,n,seed);state=solve(points,'pilot' if method!='HALTON' else 'final')
        if method=='FROZEN_GHG':
            pilot=c.pilot_gradient_from_result(state)
            length=min(.20,2.5/np.sqrt(n)) if family=='S' else .10
            points,_=c.protected_weighted_allocation(problem,n,seed,(1/3,)*3,pilot,
                         skeleton_fraction=.45,green_length=length,candidate_multiplier=18)
            state=solve(points,'final')
        elif method in ('DEFECT_ONLY','ADJOINT','GHG_ADJOINT'):
            t=time.perf_counter()
            score,info,extra=defect_indicator(problem,state,n,seed,family,method!='DEFECT_ONLY')
            arrays.update(extra)
            points,extra=allocate_supplement(problem,state,n,seed,nb,ni,score,method)
            arrays.update(extra);indicator_seconds=time.perf_counter()-t
            state=solve(points,'final')
    required=time.perf_counter()-start
    metrics,extra=b.diagnostics(problem,state,points,cfg);arrays.update(extra)
    # Independent objective quadrature twice the indicator order; exact used
    # here ONLY for validation. Also save quadrature sensitivity at order 36.
    for order in (24,36):
        qp,qw=objective_points(order)
        val=float(qw@b.apply(problem,state,qp,'value'));ref=float(qw@problem.exact(qp))
        metrics[f'J_{order}']=val;metrics[f'J_exact_{order}']=ref
        metrics[f'J_abs_error_{order}']=abs(val-ref)
    metrics['J_abs_error']=metrics['J_abs_error_36']
    metrics['J_quadrature_change']=abs(metrics['J_36']-metrics['J_24'])
    metrics['J_exact_quadrature_change']=abs(metrics['J_exact_36']-metrics['J_exact_24'])
    metrics['screen_objective_quadrature']=bool(max(metrics['J_quadrature_change'],metrics['J_exact_quadrature_change'])<=.1*max(metrics['J_abs_error'],1e-12))
    metrics['screen_supplement']=bool(metrics['screen_joint'] and metrics['screen_objective_quadrature'] and info['adjoint_relative_residual']<=1e-8)
    metrics.update(info,required_solution_seconds=required,
        diagnostic_seconds=time.perf_counter()-start-required,
        indicator_and_remesh_seconds=indicator_seconds,primal_solves=len(stages),
        primary_factorizations=len(stages)+(method in ('DEFECT_ONLY','ADJOINT','GHG_ADJOINT')),
        interior_budget=n,seed=seed,method=method,problem=problem.name)
    arrays.update(interior_points=points,**b.state_arrays(state,'final_'))
    tmp=folder/'state.partial'
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(folder/'state.npz');c.atomic_csv(folder/'stages.csv',stages)
    b.write_json(folder/'metrics.json',metrics)
    return metrics

def preflight(out):
    """Check full discrete identity, transpose solve, and scaling invariance.
    Enriched primal solve is ONLY a preflight verification, never an oracle
    used to tune a density. This is algebraic validation, not an error theorem.
    """
    reports=[]
    for family,p in (fixtures()[0],fixtures()[3]):
        nb,ni=b.boundary_counts(family,120)
        pilot=b.solve(p,c.global_halton(p,120,19),nb,ni)
        fine=enriched_system(p,120,19,family)
        xy=np.vstack((fine.minus.nodes,fine.plus.nodes))
        side=np.r_[np.full(len(fine.minus.nodes),-1),np.full(len(fine.plus.nodes),1)]
        v=interpolation(pilot,xy,side)@values(pilot)
        qp,qw=objective_points(12);j=np.asarray(interpolation(fine,qp).T@qw).ravel()
        z=fine.lu.solve(j,trans='T');r=fine.rhs-fine.matrix@v
        u=fine.lu.solve(fine.rhs);left=float(j@(u-v));right=float(z@r)
        rel=abs(left-right)/max(abs(left),abs(right),1e-12)
        scale=np.exp(np.linspace(-2,2,len(r)))
        lu=splu((diags(scale)@fine.matrix).tocsc())
        zs=lu.solve(j,trans='T')
        scaling=float(np.linalg.norm(zs*(scale*r)-z*r)/max(np.linalg.norm(z*r),1e-12))
        adj=float(np.linalg.norm(fine.matrix.T@z-j)/max(np.linalg.norm(j),1e-12))
        reports.append(dict(problem=p.name,identity_relative=rel,scaling_relative=scaling,adjoint_relative=adj,
                            passed=bool(rel<1e-6 and scaling<1e-6 and adj<1e-8)))
    b.write_json(out/'preflight.json',reports)
    if not all(x['passed'] for x in reports):raise RuntimeError('Adjoint preflight failed; inspect preflight.json')

def summarize(rows,out):
    import pandas as pd
    df=pd.DataFrame(rows);df.to_csv(out/'campaign.csv',index=False)
    metrics=['J_abs_error','full_rmse','collar_rmse','required_solution_seconds','conservation_defect']
    df.groupby(['problem','method','interior_budget'])[metrics].median().to_csv(out/'budget_medians.csv')
    df.groupby(['problem','method'])[['valid_solve','screen_joint','screen_conservation','screen_objective_quadrature','screen_supplement']].agg(['sum','count']).to_csv(out/'acceptance_counts.csv')
    pairs=[]
    for ref in ('HALTON','FROZEN_GHG','ADAPT_RESIDUAL','DEFECT_ONLY','ADJOINT'):
        a=df[df.method=='GHG_ADJOINT'];r=df[df.method==ref]
        m=a.merge(r,on=['problem','interior_budget','seed'],suffixes=('_a','_b'))
        for problem,g in m.groupby('problem'):
            for metric in metrics:
                ratios=g[metric+'_a']/g[metric+'_b'].clip(lower=1e-15)
                # Average logs by seed first: budgets are NOT independent reps.
                logs=np.log(ratios.clip(lower=1e-15)).groupby(g.seed).mean()
                pairs.append(dict(problem=problem,reference=ref,metric=metric,
                    geometric_ratio=float(np.exp(logs.mean())),seed_blocks=len(logs),
                    matched_pairs=len(g),wins=int((ratios<1).sum())))
    c.atomic_csv(out/'paired_comparisons.csv',pairs)
    med=df.groupby(['problem','method','interior_budget'])[metrics].median().reset_index()
    envelope=[]
    for _,a in med[med.method=='GHG_ADJOINT'].iterrows():
        for ref in ('HALTON','ADJOINT'):
            options=med[(med.problem==a.problem)&(med.method==ref)&
                        (med.required_solution_seconds<=a.required_solution_seconds)]
            for metric in ('J_abs_error','full_rmse'):
                if len(options):
                    best=options.loc[options[metric].idxmin()]
                    envelope.append(dict(problem=a.problem,budget=a.interior_budget,reference=ref,
                        metric=metric,reference_budget=best.interior_budget,
                        ratio=a[metric]/max(best[metric],1e-15)))
    c.atomic_csv(out/'measured_cost_envelope.csv',envelope)
    import matplotlib.pyplot as plt
    for problem,g in med.groupby('problem'):
        fig,axs=plt.subplots(1,2,figsize=(12,4.5))
        for method,h in g.groupby('method'):
            h=h.sort_values('required_solution_seconds')
            for ax,metric in zip(axs,('J_abs_error','full_rmse')):
                ax.loglog(h.required_solution_seconds,h[metric].clip(lower=1e-15),'.-',label=method)
        for ax,title in zip(axs,('Regional mean absolute error','Full-field RMSE')):
            ax.set(xlabel='Required solution seconds',ylabel=title);ax.grid(True,alpha=.2)
        axs[0].legend(fontsize=7);fig.suptitle(problem);fig.tight_layout()
        fig.savefig(out/('error_cost_'+problem+'.png'),dpi=160);plt.close(fig)

def main(profile=PROFILE,output=None,stop_after=0,only_problem=None):
    cfg=b.settings('production' if profile=='production' else 'screen')
    budgets=(120,260,360) if profile=='production' else (120,)
    seeds=(11,23,37) if profile=='production' else (11,)
    problems=fixtures()
    if only_problem:problems=[x for x in problems if x[1].name==only_problem]
    if not problems:raise ValueError('Unknown problem')
    out=Path(output or OUTPUT_DIRECTORY or HERE/('results_green_adjoint_'+profile+'_v1')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    sources={p.name:b.digest(p) for p in HERE.glob('*.py')}
    contract=dict(version='1.0',profile=profile,budgets=budgets,seeds=seeds,methods=METHODS,
        problems=[p.name for _,p in problems],cfg=cfg,sources=sources,
        versions=dict(numpy=np.__version__,scipy=b.scipy.__version__,python=sys.version),extra_halton=(520,720) if profile=='production' else (),
        objective='mean on [0.65,0.90] x [0.35,0.65]',enrichment=2,
        inference='discrete adjoint heuristic; no continuum error bound; existing geometries not holdouts')
    pid=c.protocol_hash(contract)
    if (out/'protocol.json').exists():
        if c.load_json(out/'protocol.json')['id']!=pid:raise RuntimeError('Protocol/source changed: choose a fresh output directory.')
    else:b.write_json(out/'protocol.json',dict(id=pid,contract=contract))
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for name in sources:shutil.copy2(HERE/name,snapshot/name)
    b.write_json(out/'environment.json',c.environment_record())
    log=c.OrderedLog(out/'run.log')
    with threadpool_limits(limits=1):
        if not (out/'preflight.json').exists():preflight(out)
        elif not all(r['passed'] for r in json.loads((out/'preflight.json').read_text())):raise RuntimeError('Saved preflight failure')
        tasks=[]
        for family,p in problems:
            for seed in seeds:
                for n in budgets:
                    offset=(seed+n)%len(METHODS);order=METHODS[offset:]+METHODS[:offset]
                    tasks.extend((family,p,method,n,seed) for method in order)
                if profile=='production':tasks.extend((family,p,'HALTON',n,seed) for n in (520,720))
        rows=[];executed=0
        for family,p,method,n,seed in tasks:
            folder=out/'cases'/f'{p.name}__{method}__N{n}__s{seed}'
            receipt=folder/'commit.json'
            if receipt.exists():
                commit=c.load_json(receipt)
                if commit['protocol']!=pid or any(b.digest(folder/f)!=h for f,h in commit['hashes'].items()):
                    raise RuntimeError('Saved case integrity failure: '+str(folder))
                rows.append(c.load_json(folder/'metrics.json'));continue
            log(f'{len(rows)+1}/{len(tasks)} {p.name} {method} N={n} seed={seed}')
            try:
                metrics=execute(p,family,method,n,seed,cfg,folder)
                b.write_json(receipt,dict(protocol=pid,hashes={f:b.digest(folder/f) for f in ('state.npz','stages.csv','metrics.json')}))
                rows.append(metrics);executed+=1
            except Exception:
                (out/'failure.txt').write_text(traceback.format_exc());raise
            if stop_after and executed>=stop_after:break
        summarize(rows,out)
        complete=len(rows)==len(tasks)
        b.write_json(out/'status.json',dict(complete=complete,completed=len(rows),expected=len(tasks),protocol=pid))
        log(f'Completed {len(rows)}/{len(tasks)}. Upload: {out}.zip')
        c.atomic_csv(out/'scientific_manifest_sha256.csv',c.scientific_manifest(out))
        shutil.make_archive(str(out),'zip',out.parent,out.name)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=('screen','production'),default=PROFILE)
    parser.add_argument('--output');parser.add_argument('--stop-after',type=int,default=0)
    parser.add_argument('--only-problem',default=None)
    args,_unknown=parser.parse_known_args()
    main(args.profile,args.output,args.stop_after,args.only_problem)
