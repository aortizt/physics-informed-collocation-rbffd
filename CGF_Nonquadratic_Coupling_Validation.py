"""Next bounded CGF experiment: nonquadratic two-material coupling validation.

Manuscript destination: Section 5.4 and the restricted DDM conclusion.
Reason: the previous circular exact solution belongs to the reproduced quadratic
space. Its displayed field error predominantly measures linear interpolation.
This experiment changes the manufactured state, records separate nodal/PHS/linear
errors, checks independent PDE residuals, records pilot and branch costs, and saves every
iterate. It does not retune the three existing Robin choices.

SPYDER: extract the next-run package; keep this file beside the supplied
CGF_Replay_Common_PHS_RBFFD.py; press Run. No earlier campaign must be run first.
Default production: 2 configurations x 4 budgets x 5 NEW seeds x 2 allocations
x 3 transmissions = 240 cases. Re-running resumes completed matching cases.
CLI implementation check: python CGF_Nonquadratic_Coupling_Validation.py --profile screen
The screen is not manuscript evidence. It writes a separate directory.

The test is prospective. Neither favorable nor unfavorable results should trigger
parameter searches in Paper I. A finite positive result is still not a general
contraction theorem or a many-subdomain/coarse-space validation.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import argparse, dataclasses, hashlib, json, math, shutil, sys, time, traceback
import numpy as np
from scipy.sparse import coo_matrix
import CGF_Replay_Common_PHS_RBFFD as core

# -------------------- TOP-LEVEL FROZEN CONFIGURATION --------------------
PROFILE='production'
OUTPUT_DIRECTORY=''  # Empty means a results directory beside this script.
SEEDS=(101,149,211,269,331)
BUDGETS=(120,180,260,360)
NONQUADRATIC_AMPLITUDE=0.4
RELAXATION=0.7
MAXIMUM_ITERATIONS=20
EVALUATION_RESOLUTION=81
INDEPENDENT_RESIDUAL_RESOLUTION=25
COLLAR_WIDTH=0.08
STENCIL_SIZE=25
ALLOCATIONS=('HALTON','PROTECTED_GHG_EQUAL')
TRANSMISSIONS=('SCALAR_ROBIN_8','DTN_GEOMETRIC','GHG_CONDITIONED_DTN')
GATES=dict(condition_max=1e10,range_violation_max=.05,relative_algebraic_max=1e-8,
           boundary_error_max=1e-8,independent_residual_relative_max=.25,
           full_relative_l2_max=.05,conservation_max=.05,physical_terminal_max=.005,
           q_net_max=.95,q_tail_max=.95,monolithic_error_factor=1.25,
           relative_fill_factor=1.5,required_pass_fraction=.9)
EXPECTED_CORE_SHA256='9d78b46cb00ef32d3546d81365950f0989b2bd1ccd95671f5c73ac8f1f08a381'
VERSION='CGF-NONQUADRATIC-VALIDATION-1.0'
# ----------------------------------------------------------------------

def make_problem(name,center,radius,k_inside):
    base=core.circular_inclusion_problem(name,center,radius,k_inside,1.)
    c=np.asarray(center); eps=NONQUADRATIC_AMPLITUDE
    def exact(points,side):
        dx=points[:,0]-c[0];dy=points[:,1]-c[1]
        h=dx*dx+dy*dy-radius*radius
        s=np.sin(np.pi*points[:,0])*np.sin(np.pi*points[:,1])
        k=np.where(side<0,k_inside,1.)
        return base.exact(points,side)+eps*h*s/k
    def forcing(points,side):
        x,y=points[:,0],points[:,1];dx=x-c[0];dy=y-c[1]
        h=dx*dx+dy*dy-radius*radius
        sx,sy=np.sin(np.pi*x),np.sin(np.pi*y)
        lap=4*sx*sy+4*np.pi*dx*np.cos(np.pi*x)*sy+4*np.pi*dy*sx*np.cos(np.pi*y)-2*np.pi**2*h*sx*sy
        return base.forcing(points,side)-eps*lap
    # h vanishes on the interface; K grad(eps*h*s/K)=eps grad(h*s)
    # agrees on both sides. Dirichlet data and forcing are manufactured inputs.
    return dataclasses.replace(base,exact_function=exact,forcing_function=forcing,
        metadata={**base.metadata,'state':'quadratic_plus_eps_times_r2_minus_R2_times_sinpix_sinpiy_over_K','epsilon':eps})

def problems():
    return (make_problem('NQ_TRAIN',(0.5,0.5),.24,50.),
            make_problem('NQ_HOLDOUT',(0.56,.44),.20,20.))

def grid(resolution,midpoint=False):
    c=(np.arange(resolution)+.5)/resolution if midpoint else np.linspace(0,1,resolution)
    x,y=np.meshgrid(c,c,indexing='xy');return np.c_[x.ravel(),y.ravel()]

def evaluation_operator(problem,cloud,points,operator):
    positions=np.where(problem.side(points)==cloud.side)[0]
    rr=[];cc=[];vv=[]
    for row in positions:
        stencil,weights=core.local_phs_weights(cloud,points[row],operator,stencil_size=STENCIL_SIZE)
        if operator=='laplacian':weights=-cloud.conductivity*weights
        rr.extend([int(row)]*len(stencil));cc.extend(stencil.tolist());vv.extend(weights.tolist())
    return coo_matrix((vv,(rr,cc)),shape=(len(points),len(cloud.nodes))).tocsr()

def prepare_evaluators(problem,mono):
    xy=grid(EVALUATION_RESOLUTION);rq=grid(INDEPENDENT_RESIDUAL_RESOLUTION,True)
    # Avoid evaluating a strong-form derivative at the discontinuity itself.
    rq=rq[np.abs(problem.phi(rq))>1e-4]
    return dict(xy=xy,rq=rq,exact=problem.exact(xy),forcing=problem.forcing(rq),
        Vm=evaluation_operator(problem,mono.minus,xy,'value'),Vp=evaluation_operator(problem,mono.plus,xy,'value'),
        Lm=evaluation_operator(problem,mono.minus,rq,'laplacian'),Lp=evaluation_operator(problem,mono.plus,rq,'laplacian'))

def state_metrics(problem,mono,um,up,ev):
    phs=np.asarray(ev['Vm']@um+ev['Vp']@up)
    linear=core.evaluate_piecewise(problem,mono.minus,mono.plus,um,up,ev['xy'])
    err=phs-ev['exact'];mask=np.abs(problem.phi(ev['xy']))<=COLLAR_WIDTH
    exactn=np.r_[problem.exact(mono.minus.nodes,np.full(len(um),-1)),problem.exact(mono.plus.nodes,np.full(len(up),1))]
    nodal=np.r_[um,up]-exactn
    rr=np.asarray(ev['Lm']@um+ev['Lp']@up)-ev['forcing']
    fscale=max(np.linalg.norm(ev['forcing']),1e-12)
    span=max(float(np.ptp(ev['exact'])),1e-12)
    range_violation=max(0.,float(phs.max()-ev['exact'].max()),float(ev['exact'].min()-phs.min()))/span
    return dict(full_rmse=float(np.sqrt(np.mean(err*err))),collar_rmse=float(np.sqrt(np.mean(err[mask]**2))),
        full_relative_l2=float(np.linalg.norm(err)/max(np.linalg.norm(ev['exact']),1e-12)),
        nodal_rmse=float(np.sqrt(np.mean(nodal*nodal))),nodal_linf=float(np.max(np.abs(nodal))),
        linear_full_rmse=float(np.sqrt(np.mean((linear-ev['exact'])**2))),
        independent_residual_relative=float(np.linalg.norm(rr)/fscale),range_violation=range_violation),phs

def allocation(name,problem,budget,seed,pilot):
    if name=='HALTON':return core.global_halton(problem,budget,seed)
    return core.protected_weighted_allocation(problem,budget,seed,(1/3,1/3,1/3),pilot,
        skeleton_fraction=.45,green_length=.10,candidate_multiplier=18)[0]

def impedance(name,problem,xy,pilot):
    if name=='SCALAR_ROBIN_8':return np.full(len(xy),8.),np.full(len(xy),8.)
    radius=problem.metadata['radius'];a=problem.k_plus/(.5-radius);b=problem.k_minus/radius
    factor=core.normalized_indicator(pilot(xy))*.7+.65 if name=='GHG_CONDITIONED_DTN' else np.ones(len(xy))
    return a*factor,b*factor

def branch(problem,points,nb,ni,lm,lp,mono,ev,baseline,halton_fill):
    start=time.perf_counter()
    sm=core.assemble_local_robin_system(problem,mono.minus,lm,STENCIL_SIZE,True)
    sp=core.assemble_local_robin_system(problem,mono.plus,lp,STENCIL_SIZE,True)
    setup=time.perf_counter()-start;start=time.perf_counter()
    um=sm.solve(np.zeros(ni));up=sp.solve(np.zeros(ni))
    ums=[];ups=[];tracesm=[];tracesp=[];fluxesm=[];fluxesp=[];history=[]
    previous_m=um.copy();previous_p=up.copy()
    for iteration in range(MAXIMUM_ITERATIONS+1):
        tm,tp=sm.trace(um),sp.trace(up);fm,fp=sm.flux(um),sp.flux(up)
        trace=float(np.sqrt(np.mean((tm-tp)**2)));flux=float(np.sqrt(np.mean((fm+fp)**2)))
        history.append(dict(iteration=iteration,trace_rms=trace,flux_rms=flux,physical_mismatch=math.hypot(trace,flux),
            update_rms=float(np.sqrt(np.mean((um-previous_m)**2)+np.mean((up-previous_p)**2)))))
        ums.append(um.copy());ups.append(up.copy());tracesm.append(tm.copy());tracesp.append(tp.copy());fluxesm.append(fm.copy());fluxesp.append(fp.copy())
        if iteration==MAXIMUM_ITERATIONS:break
        rawm=sm.solve(-fp+lm*tp);rawp=sp.solve(-fm+lp*tm)
        previous_m=um.copy();previous_p=up.copy()
        um=(1-RELAXATION)*um+RELAXATION*rawm;up=(1-RELAXATION)*up+RELAXATION*rawp
    iteration_seconds=time.perf_counter()-start;start=time.perf_counter()
    metrics,phs=state_metrics(problem,mono,um,up,ev)
    # Test fixed interior/Dirichlet equations separately from Robin exchange.
    algebraic=[];boundary=[]
    for system,value in [(sm,um),(sp,up)]:
        nfixed=len(system.cloud.interior)+len(system.cloud.boundary)
        res=system.matrix[:nfixed]@value-system.rhs_base[:nfixed]
        algebraic.append(float(np.linalg.norm(res)/max(np.linalg.norm(system.rhs_base[:nfixed]),1e-12)))
        idx=system.cloud.boundary
        boundary.append(float(np.max(np.abs(value[idx]-problem.exact(system.cloud.nodes[idx],np.full(len(idx),system.cloud.side))))) if len(idx) else 0.)
    residual=np.maximum(np.array([r['physical_mismatch'] for r in history]),1e-300)
    metrics.update(q_net=float((residual[-1]/residual[0])**(1/MAXIMUM_ITERATIONS)),
        q_tail=float(np.median(residual[-5:]/residual[-6:-1])),physical_mismatch_terminal=float(residual[-1]),
        condition_proxy_max=max(sm.condition_proxy,sp.condition_proxy),local_algebraic_relative=max(algebraic),boundary_error_max=max(boundary),
        conservation_defect=core.conservation_defect(problem,mono.minus,mono.plus,um,up,STENCIL_SIZE,quadrature_per_edge=60),
        fill_max=core.coverage_metrics(points)['coverage_fill_max'])
    finite=all(np.isfinite(v) for v in metrics.values())
    gates=dict(gate_allocation=finite and len(points)==len(mono.minus.interior)+len(mono.plus.interior) and metrics['fill_max']<=GATES['relative_fill_factor']*halton_fill,
        gate_local=finite and metrics['condition_proxy_max']<=GATES['condition_max'] and metrics['range_violation']<=GATES['range_violation_max']
        and metrics['local_algebraic_relative']<=GATES['relative_algebraic_max'] and metrics['boundary_error_max']<=GATES['boundary_error_max']
        and metrics['independent_residual_relative']<=GATES['independent_residual_relative_max'],
        gate_physical=finite and metrics['q_net']<GATES['q_net_max'] and metrics['q_tail']<GATES['q_tail_max'] and metrics['physical_mismatch_terminal']<=GATES['physical_terminal_max'],
        gate_assembly=finite and metrics['conservation_defect']<=GATES['conservation_max'] and metrics['full_relative_l2']<=GATES['full_relative_l2_max']
        and all(metrics[k]<=GATES['monolithic_error_factor']*max(baseline[k],1e-14) for k in ('full_rmse','collar_rmse')))
    gates['gate_all']=all(gates.values());metrics.update(gates)
    metrics.update(ddm_setup_seconds=setup,iteration_seconds=iteration_seconds,diagnostic_seconds=time.perf_counter()-start)
    arrays=dict(interior_points=points,nodes_minus=mono.minus.nodes,nodes_plus=mono.plus.nodes,interface_points=mono.minus.interface_points,
        normals_minus=mono.minus.interface_normals_out,normals_plus=mono.plus.interface_normals_out,
        interior_indices_minus=mono.minus.interior,interior_indices_plus=mono.plus.interior,
        boundary_indices_minus=mono.minus.boundary,boundary_indices_plus=mono.plus.boundary,
        interface_indices_minus=mono.minus.interface,interface_indices_plus=mono.plus.interface,
        values_minus=np.asarray(ums),values_plus=np.asarray(ups),trace_minus=np.asarray(tracesm),trace_plus=np.asarray(tracesp),
        flux_minus=np.asarray(fluxesm),flux_plus=np.asarray(fluxesp),lambda_minus=lm,lambda_plus=lp,
        evaluation_points=ev['xy'],evaluation_exact=ev['exact'],evaluation_phs_final=phs,
        evaluation_collar_mask=np.abs(problem.phi(ev['xy']))<=COLLAR_WIDTH,
        residual_points=ev['rq'],forcing_at_residual_points=ev['forcing'],
        monolithic_minus=mono.values_minus,monolithic_plus=mono.values_plus,
        local_matrix_minus_data=sm.matrix.data,local_matrix_minus_indices=sm.matrix.indices,local_matrix_minus_indptr=sm.matrix.indptr,
        local_matrix_plus_data=sp.matrix.data,local_matrix_plus_indices=sp.matrix.indices,local_matrix_plus_indptr=sp.matrix.indptr,
        rhs_minus=sm.rhs_base,rhs_plus=sp.rhs_base)
    return metrics,arrays,history

def validate_manufacture(problem):
    xy,normals=problem.sample_interface((np.arange(17)+.5)/17)
    trace=float(np.max(np.abs(problem.exact(xy,np.full(17,-1))-problem.exact(xy,np.full(17,1)))))
    h=1e-5
    flux=[]
    for side,k in [(-1,problem.k_minus),(1,problem.k_plus)]:
        deriv=(problem.exact(xy+h*normals,np.full(17,side))-problem.exact(xy-h*normals,np.full(17,side)))/(2*h)
        flux.append(k*deriv)
    jump=float(np.max(np.abs(flux[0]-flux[1])))
    pts=np.array([[.19,.21],[.31,.67],[.75,.78]]);side=problem.side(pts);lap=np.zeros(len(pts))
    for e in np.eye(2)*1e-4:lap+=(problem.exact(pts+e,side)-2*problem.exact(pts,side)+problem.exact(pts-e,side))/1e-8
    force=float(np.max(np.abs(-problem.conductivity(pts)*lap-problem.forcing(pts))))
    if trace>1e-10 or jump>1e-5 or force>1e-4:raise RuntimeError('Manufactured trace/flux/forcing identity failed')
    return dict(problem=problem.name,trace_identity_error=trace,flux_identity_fd_error=jump,forcing_identity_fd_error=force)

def main(profile=PROFILE):
    here=Path(__file__).resolve().parent
    if core.sha256_file(here/'CGF_Replay_Common_PHS_RBFFD.py')!=EXPECTED_CORE_SHA256:
        raise RuntimeError('The supplied frozen common-core bytes do not match this protocol.')
    seeds=SEEDS if profile=='production' else SEEDS[:1]
    budgets=BUDGETS if profile=='production' else BUDGETS[:1]
    out=Path(OUTPUT_DIRECTORY).expanduser().resolve() if OUTPUT_DIRECTORY else here/f'results_nonquadratic_{profile}_v1'
    out.mkdir(parents=True,exist_ok=True)
    contract=dict(version=VERSION,profile=profile,seeds=seeds,budgets=budgets,epsilon=NONQUADRATIC_AMPLITUDE,relaxation=RELAXATION,
        maximum_iterations=MAXIMUM_ITERATIONS,evaluation_resolution=EVALUATION_RESOLUTION,residual_resolution=INDEPENDENT_RESIDUAL_RESOLUTION,
        collar_width=COLLAR_WIDTH,stencil_size=STENCIL_SIZE,gates=GATES,allocations=ALLOCATIONS,transmissions=TRANSMISSIONS,
        source_sha256=core.sha256_file(Path(__file__)),common_core_sha256=EXPECTED_CORE_SHA256,
        selection='Training only: eligible >=90% joint passes; rank joint count then terminal mismatch then full RMSE. Freeze allocation+transmission and inspect unchanged holdout.',
        cost='Method-component charge = required pilot + allocation + Robin setup/iteration + diagnostic evaluation. Cloud construction, monolithic reference/evaluator setup, and state I/O are outside this sum; no end-to-end speedup claim.',
        holdout_independence='New seeds and nonquadratic state; geometry parameters reused from the earlier fixtures; no post-holdout tuning.')
    contract['protocol_hash']=core.protocol_hash(contract)
    pp=out/'protocol.json'
    if pp.exists() and core.load_json(pp)['protocol_hash']!=contract['protocol_hash']:
        raise RuntimeError('Existing output has a different protocol. Choose a new OUTPUT_DIRECTORY; do not overwrite evidence.')
    core.atomic_json(pp,contract);core.atomic_json(out/'environment.json',core.environment_record())
    source=out/'source';source.mkdir(exist_ok=True)
    for f in [Path(__file__),here/'CGF_Replay_Common_PHS_RBFFD.py']:shutil.copy2(f,source/f.name)
    log=core.OrderedLog(out/'run.log');log(f'START {VERSION}; {profile}; protocol={contract["protocol_hash"]}')
    core.atomic_json(out/'manufacture_validation.json',{'checks':[validate_manufacture(p) for p in problems()]})
    rows=[]
    for problem in problems():
        for n in budgets:
            nb=max(8,math.ceil(math.sqrt(n)*.72));ni=max(12,math.ceil(math.sqrt(n)*.9))
            for seed in seeds:
                # Save and charge required numerical-pilot work once, then attribute
                # it to each method that actually uses it; do not divide by branches.
                start=time.perf_counter();hp=core.global_halton(problem,n,seed)
                hm=core.solve_monolithic(problem,hp,nb,ni,EVALUATION_RESOLUTION,COLLAR_WIDTH,STENCIL_SIZE,True)
                pilot=core.pilot_gradient_from_result(hm);pilot_seconds=time.perf_counter()-start
                hfill=core.coverage_metrics(hp)['coverage_fill_max']
                for a in ALLOCATIONS:
                    start=time.perf_counter();points=allocation(a,problem,n,seed,pilot);allocation_seconds=time.perf_counter()-start
                    start=time.perf_counter();mono=hm if a=='HALTON' else core.solve_monolithic(problem,points,nb,ni,EVALUATION_RESOLUTION,COLLAR_WIDTH,STENCIL_SIZE,True)
                    ev=prepare_evaluators(problem,mono);baseline,_=state_metrics(problem,mono,mono.values_minus,mono.values_plus,ev)
                    reference_seconds=time.perf_counter()-start+(pilot_seconds if a=='HALTON' else 0.)
                    for t in TRANSMISSIONS:
                        stem=core.case_stem(problem.name,a+'_'+t,n,seed);folder=out/'cases'/stem;folder.mkdir(parents=True,exist_ok=True)
                        record=folder/'metrics.json';statefile=folder/'states.npz'
                        if record.exists():
                            old=core.load_json(record)
                            if old.get('protocol_hash')!=contract['protocol_hash']:raise RuntimeError('Case protocol mismatch: '+stem)
                            if old.get('status')=='failed' or (statefile.exists() and core.sha256_file(statefile)==old.get('state_sha256')):
                                rows.append(old);log('RESUME '+stem);continue
                            raise RuntimeError('State archive missing or changed: '+stem)
                        row=dict(problem=problem.name,configuration='TRAIN' if 'TRAIN' in problem.name else 'HOLDOUT',budget=n,seed=seed,allocation=a,transmission=t,profile=profile,protocol_hash=contract['protocol_hash'])
                        try:
                            start=time.perf_counter();lm,lp=impedance(t,problem,mono.minus.interface_points,pilot);impedance_seconds=time.perf_counter()-start
                            metrics,arrays,history=branch(problem,points,nb,ni,lm,lp,mono,ev,baseline,hfill)
                            needpilot=(a!='HALTON' or t=='GHG_CONDITIONED_DTN')
                            charge=(pilot_seconds if needpilot else 0.)+allocation_seconds+impedance_seconds+metrics['ddm_setup_seconds']+metrics['iteration_seconds']+metrics['diagnostic_seconds']
                            row.update(metrics,status='completed',pilot_seconds=pilot_seconds,pilot_charged_seconds=pilot_seconds if needpilot else 0.,allocation_seconds=allocation_seconds,impedance_seconds=impedance_seconds,reference_evaluator_seconds=reference_seconds,charged_total_seconds=charge,monolithic_full_rmse=baseline['full_rmse'],monolithic_collar_rmse=baseline['collar_rmse'],monolithic_nodal_rmse=baseline['nodal_rmse'])
                            arrays.update(pilot_nodes_minus=hm.minus.nodes,pilot_nodes_plus=hm.plus.nodes,pilot_values_minus=hm.values_minus,pilot_values_plus=hm.values_plus)
                            with (folder/'states.partial').open('wb') as f:np.savez_compressed(f,**arrays)
                            (folder/'states.partial').replace(statefile);row['state_sha256']=core.sha256_file(statefile)
                            core.atomic_csv(folder/'history.csv',history)
                        except Exception as error:
                            row.update(status='failed',gate_all=False,error=repr(error));(folder/'error.txt').write_text(traceback.format_exc())
                        core.atomic_json(record,row);rows.append(row);log(f'{row["status"].upper()} {stem}; joint={row["gate_all"]}')
    fields=sorted(set().union(*(r.keys() for r in rows)));core.atomic_csv(out/'campaign_metrics.csv',rows,fields)
    expected=len(budgets)*len(seeds);required=math.ceil(expected*GATES['required_pass_fraction']);summaries=[]
    for config in ('TRAIN','HOLDOUT'):
        for a in ALLOCATIONS:
            for t in TRANSMISSIONS:
                sub=[r for r in rows if r['configuration']==config and r['allocation']==a and r['transmission']==t]
                ok=[r for r in sub if r['status']=='completed']
                summaries.append(dict(configuration=config,allocation=a,transmission=t,rows=len(sub),joint_passes=sum(bool(r['gate_all']) for r in sub),median_terminal=float(np.median([r['physical_mismatch_terminal'] for r in ok])) if ok else 1e300,median_full_rmse=float(np.median([r['full_rmse'] for r in ok])) if ok else 1e300))
    eligible=[r for r in summaries if r['configuration']=='TRAIN' and r['joint_passes']>=required]
    winner=min(eligible,key=lambda r:(-r['joint_passes'],r['median_terminal'],r['median_full_rmse'])) if eligible else None
    holdout=next((r for r in summaries if winner and r['configuration']=='HOLDOUT' and r['allocation']==winner['allocation'] and r['transmission']==winner['transmission']),None)
    decision=dict(profile=profile,production_evidence=profile=='production',required_passes=required,training_selected=winner,selected_holdout=holdout,
        selected_branch_qualifies=bool(winner and holdout['joint_passes']>=required) if profile=='production' else None,
        no_retuning=True,scope='Finite nonquadratic two-subdomain test; no general contraction, coarse-space, or speedup claim.')
    core.atomic_csv(out/'branch_summary.csv',summaries);core.atomic_json(out/'decision.json',decision)
    core.atomic_json(out/('production_complete.json' if profile=='production' else 'screen_complete.json'),dict(profile=profile,cases=len(rows),expected_cases=2*len(seeds)*len(budgets)*6,failed_cases=sum(r['status']=='failed' for r in rows),protocol_hash=contract['protocol_hash']))
    log(f'COMPLETE {len(rows)} cases; production evidence={profile=="production"}')
    core.atomic_csv(out/'scientific_manifest_sha256.csv',core.scientific_manifest(out))
    print('Upload the complete output directory as a ZIP. Do not select only favorable rows.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',choices=['screen','production'],default=PROFILE)
    args,_unknown=parser.parse_known_args();main(args.profile)
