"""CGF Paper I: bounded, frozen adaptive-allocation comparison (Spyder entry point).

Extract the entire package, open THIS script in a fresh Spyder console, press F5.
No previous campaigns or input downloads are needed. Default: production.
Four fixtures/configurations x four budgets x five seeds x five methods = 400
final cases. Adaptive cases contain a pilot and three fixed remeshing cycles.
A small separate Gaussian-Poisson implementation/adequacy check runs first.

Outputs resume at completed CASE boundaries, verified against source, protocol,
environment and payload hashes. Interrupted cases restart; completed failures
remain recorded failures. No tuning, best-iterate selection or oracle allocation.
The existing transfer configurations have already been inspected in prior work.

Primary comparison: identical simultaneous PHS-RBF-FD interface equations.
Secondary Q comparison: frozen unmodulated cross-side Robin, 20 iterations.
All method-required pilots are independently run and fully charged. Timed primary
solve, primary audit, and full pipeline (including state write) are distinct.

Adaptive controls are disclosed local implementations, NOT exact reproductions
of an author's code: SK-style solution-variation spacing update (Slak & Kosec,
2019, doi:10.2495/BE420131, Sec.3, Eqs.7-8) and a residual-indicator variant.
We adapt that remeshing idea to exact budgets using weighted farthest-point
selection, fixed boundary/interface nodes, same-side Shepard interpolation,
quantile thresholds, and three cycles. See PROTOCOL.md for all departures.
Do not label a win against these controls as superiority over all adaptive RBF-FD.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
from pathlib import Path
import argparse, dataclasses, hashlib, importlib.util, itertools, json, math
import shutil, sys, time, traceback, platform
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits, threadpool_info
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# F5 defaults. Use CLI --profile screen for a short implementation run.
PROFILE = 'production'
OUTPUT_DIRECTORY = ''  # Empty: results beside this script; each profile separated.
VERSION = 'CGF-ADAPTIVE-COMPARISON-1.0'
METHODS = ('HALTON','FROZEN_GHG','QUASI_UNIFORM','ADAPT_VARIATION','ADAPT_RESIDUAL')
BUDGETS = (120,180,260,360)
STATIC_SEEDS = (11,23,37,53,71)
CIRCULAR_SEEDS = (101,149,211,269,331)
ADAPT_CYCLES = 3
STENCIL = 25
POOL_MULTIPLIER = 32
NEIGHBORS = 15
REFINE_FACTOR = 2.0
COARSEN_FACTOR = 1.5
LOW_QUANTILE = 0.30
HIGH_QUANTILE = 0.70
COLLAR = 0.08
CORE_HASH = '9d78b46cb00ef32d3546d81365950f0989b2bd1ccd95671f5c73ac8f1f08a381'
Q_HASH = '768405d4d74258c56b1195dd2041c971d310eea0ecc784d4ab0eea8776f48d67'
HERE = Path(__file__).resolve().parent

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_frozen(name, expected):
    p=HERE/(name+'.py')
    if not p.is_file() or digest(p)!=expected:
        raise RuntimeError('Missing or changed supplied dependency: '+str(p))
    spec=importlib.util.spec_from_file_location(name,p)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

# Explicit reload avoids a stale cached module in an interactive Spyder session.
core=load_frozen('CGF_Replay_Common_PHS_RBFFD',CORE_HASH)
nq=load_frozen('CGF_Nonquadratic_Coupling_Validation',Q_HASH)

def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,(np.bool_,)):return bool(obj)
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(np.floating,float)):return float(obj) if np.isfinite(obj) else None
    return obj

def write_json(p,obj):
    core.atomic_json(Path(p),clean(obj))

def settings(profile):
    return dict(profile=profile,budgets=BUDGETS if profile=='production' else (120,),
                static_seeds=STATIC_SEEDS if profile=='production' else STATIC_SEEDS[:1],
                circular_seeds=CIRCULAR_SEEDS if profile=='production' else CIRCULAR_SEEDS[:1],
                eval_resolution=81 if profile=='production' else 31,residual_resolution=25,
                adaptive_cycles=ADAPT_CYCLES)

def fixtures():
    return [('S','TRAIN',core.vertical_resistance_problem('S_TRAIN',.50,1.,100.)),
            ('Q','TRAIN',nq.problems()[0]),
            ('S','TRANSFER',core.vertical_resistance_problem('S_TRANSFER',.37,1.,30.)),
            ('Q','TRANSFER',nq.problems()[1])]

def boundary_counts(family,n):
    return ((max(8,math.ceil(math.sqrt(n)*.70)),max(10,math.ceil(math.sqrt(n)*.85)))
            if family=='S' else (max(8,math.ceil(math.sqrt(n)*.72)),max(12,math.ceil(math.sqrt(n)*.9))))

def solve(problem,points,nb,ni):
    """Frozen equations/weights, without unused pilot manufactured-error work."""
    start=time.perf_counter()
    minus,plus,raw,rhs=core.assemble_monolithic(problem,points,nb,ni,STENCIL)
    norm=np.sqrt(np.asarray(raw.multiply(raw).sum(axis=1)).ravel())
    if not np.all(np.isfinite(norm)) or np.any(norm<=0):raise ArithmeticError('Invalid matrix row')
    matrix=(diags(1/norm)@raw).tocsc();b=rhs/norm
    assembled=time.perf_counter();lu=splu(matrix);u=lu.solve(b);ended=time.perf_counter()
    if not np.all(np.isfinite(u)):raise ArithmeticError('Nonfinite solution')
    m=len(minus.nodes)
    return SimpleNamespace(problem=problem,minus=minus,plus=plus,values_minus=u[:m],values_plus=u[m:],
        matrix=matrix.tocsr(),matrix_raw=raw,rhs=b,lu=lu,
        assembly_seconds=assembled-start,factor_solve_seconds=ended-assembled)

def apply(problem,state,xy,op):
    out=np.empty(len(xy));sides=problem.side(xy)
    for cloud,u in [(state.minus,state.values_minus),(state.plus,state.values_plus)]:
        for i in np.flatnonzero(sides==cloud.side):
            ix,w=core.local_phs_weights(cloud,xy[i],op,stencil_size=STENCIL)
            out[i]=w@u[ix]
    return out

def shepard(xy,x,h):
    """Positive compact-neighbor inverse-square Shepard interpolation."""
    dist,ix=cKDTree(x).query(xy,k=min(NEIGHBORS,len(x)))
    if dist.ndim==1:dist=dist[:,None];ix=ix[:,None]
    weights=1/np.maximum(dist,1e-12)**2
    return np.sum(weights*h[ix],axis=1)/np.sum(weights,axis=1)

def interpolate_sides(problem,x,h,xy):
    ans=np.empty(len(xy));sx=problem.side(x);sq=problem.side(xy)
    for sign in (-1,1):
        use=sq==sign;known=sx==sign
        if np.any(use):
            if not np.any(known):raise RuntimeError('Spacing data absent in a material')
            ans[use]=shepard(xy[use],x[known],h[known])
    return ans

def remesh(problem,n,seed,nb,ni,known=None):
    """Exact-N weighted maximin selection with fixed boundary/interface anchors.

    Candidate score=min_j 2|x-x_j|/(h(x)+h(x_j)); positive target spacing
    h, bounded density contrast. Min 10 interior nodes/material is the core's
    solvability floor, enforced identically for regular and adaptive controls.
    This finite-candidate algorithm is NOT the Slak-Kosec advancing-front code.
    """
    pool=core.global_halton(problem,POOL_MULTIPLIER*n,seed+530923)
    spacing=np.ones(len(pool)) if known is None else interpolate_sides(problem,*known,pool)
    spacing=np.clip(spacing,np.median(spacing)/4,np.median(spacing)*4)
    edge=core._outer_boundary_points(nb)
    edge=edge[np.abs(problem.phi(edge))>1e-10]
    param=(np.arange(ni)+.5)/ni if problem.metadata['geometry']=='circular_inclusion' else (np.arange(ni)+1)/(ni+1)
    inter,_=problem.sample_interface(param);anchors=np.vstack((edge,inter))
    ah=np.ones(len(anchors)) if known is None else interpolate_sides(problem,*known,anchors)
    score=np.full(len(pool),np.inf)
    for x,h in zip(anchors,ah):score=np.minimum(score,2*np.linalg.norm(pool-x,axis=1)/(spacing+h))
    side=problem.side(pool);selected=[];counts={-1:0,1:0}
    for j in range(n):
        needed=[s for s in (-1,1) if counts[s]<10]
        target=needed[j%len(needed)] if needed else None
        candidate_score=np.where(side==target,score,-np.inf) if target is not None else score
        ix=int(np.argmax(candidate_score))
        if not np.isfinite(candidate_score[ix]) or candidate_score[ix]<=0:raise RuntimeError('Candidate pool exhausted')
        selected.append(ix);counts[int(side[ix])]+=1
        score=np.minimum(score,2*np.linalg.norm(pool-pool[ix],axis=1)/(spacing+spacing[ix]))
        score[np.asarray(selected)]=-np.inf
    ix=np.asarray(selected);return pool[ix],dict(candidate_points=pool,target_spacing=spacing,selected_indices=ix)

def update_spacing(problem,state,n,seed,cycle,method):
    """Uses only numerical state, geometry and prescribed forcing (no exact u)."""
    allx=[];allh=[];eta_all=[]
    for cloud,u in [(state.minus,state.values_minus),(state.plus,state.values_plus)]:
        x=cloud.nodes[cloud.interior]
        # Local spacing is measured against all same-side physical nodes.
        dist,idx=cloud.tree.query(x,k=min(STENCIL,len(cloud.nodes)))
        h=np.mean(dist[:,1:4],axis=1)
        if method=='ADAPT_VARIATION':eta=np.std(u[idx],axis=1)
        else:
            # Shifted low-discrepancy probes are distinct from collocation and
            # final validation grids. Strong residual divided by material K,
            # multiplied by local h^2, supplies a solution-scale indicator.
            probe=core.global_halton(problem,max(400,4*n),seed+907633+101*cycle)
            probe=probe[problem.side(probe)==cloud.side]
            res=apply(problem,state,probe,'laplacian')+problem.forcing(probe)/cloud.conductivity
            hd,_=cloud.tree.query(probe,k=4);hp=np.mean(hd[:,1:4],axis=1)
            eta=shepard(x,probe,np.abs(res)*hp**2)
        allx.append(x);allh.append(h);eta_all.append(eta)
    x=np.vstack(allx);h=np.concatenate(allh);eta=np.concatenate(eta_all)
    if not np.all(np.isfinite(eta)):raise ArithmeticError('Nonfinite adaptive indicator')
    low,high=np.quantile(eta,[LOW_QUANTILE,HIGH_QUANTILE]);mn,mx=float(eta.min()),float(eta.max())
    factor=np.ones(len(eta))
    if low>mn+1e-15:
        mask=eta<low;factor[mask]=1+(low-eta[mask])/(low-mn)*(1/COARSEN_FACTOR-1)
    if mx>high+1e-15:
        mask=eta>high;factor[mask]=1+(eta[mask]-high)/(mx-high)*(REFINE_FACTOR-1)
    hh=h/factor
    hh=np.clip(hh,np.median(hh)/4,np.median(hh)*4)
    return (x,hh),dict(indicator_points=x,indicator=eta,old_spacing=h,new_spacing=hh,
                       refine_threshold=high,coarsen_threshold=low)

def quadrature(problem,order):
    z,w=np.polynomial.legendre.leggauss(order);z=(z+1)/2;w=w/2
    intervals=[(0.,1.)]
    if problem.metadata['geometry']=='vertical_line':
        xi=problem.metadata['xi'];intervals=[(0.,xi),(xi,1.)]
    volume=0.;bp=[];bw=[];bn=[]
    for a,b in intervals:
        x=a+(b-a)*z;xx,yy=np.meshgrid(x,z,indexing='ij')
        volume+=np.sum(problem.forcing(np.c_[xx.ravel(),yy.ravel()]).reshape(order,order)*np.outer(w*(b-a),w))
        for y,normal in [(0.,[0.,-1.]),(1.,[0.,1.])]:
            bp.extend(np.c_[x,np.full(order,y)]);bw.extend(w*(b-a));bn.extend([normal]*order)
    for x,normal in [(0.,[-1.,0.]),(1.,[1.,0.])]:
        bp.extend(np.c_[np.full(order,x),z]);bw.extend(w);bn.extend([normal]*order)
    return float(volume),np.asarray(bp),np.asarray(bw),np.asarray(bn)

def balance(problem,state,order):
    volume,xy,w,normals=quadrature(problem,order);flux=np.empty(len(xy));side=problem.side(xy)
    for cloud,u in [(state.minus,state.values_minus),(state.plus,state.values_plus)]:
        for i in np.flatnonzero(side==cloud.side):
            ix,weights=core.local_phs_weights(cloud,xy[i],'directional',direction=normals[i],stencil_size=STENCIL)
            flux[i]=cloud.conductivity*(weights@u[ix])
    boundary=float(w@flux);return abs(volume+boundary)/max(abs(volume),abs(boundary),1e-12),volume,boundary

def diagnostics(problem,state,points,cfg):
    xy=nq.grid(cfg['eval_resolution']);exact=problem.exact(xy)
    uh=apply(problem,state,xy,'value');err=uh-exact;mask=np.abs(problem.phi(xy))<=COLLAR
    linear=core.evaluate_piecewise(problem,state.minus,state.plus,state.values_minus,state.values_plus,xy)
    rp=nq.grid(cfg['residual_resolution'],True);rp=rp[np.abs(problem.phi(rp))>1e-4]
    residual=-problem.conductivity(rp)*apply(problem,state,rp,'laplacian')-problem.forcing(rp)
    normres=np.linalg.norm(residual)/max(np.linalg.norm(problem.forcing(rp)),1e-12)
    values=np.r_[state.values_minus,state.values_plus]
    en=np.r_[problem.exact(state.minus.nodes,np.full(len(state.values_minus),-1)),
             problem.exact(state.plus.nodes,np.full(len(state.values_plus),1))]
    cond=core._equilibrated_condition_proxy(state.matrix,state.lu)
    algebraic=float(np.linalg.norm(state.matrix@values-state.rhs)/max(np.linalg.norm(state.rhs),1e-12))
    bal,vol,flux=balance(problem,state,48);bal2,_,_=balance(problem,state,72)
    spacing=cKDTree(points).query(points,k=2)[0][:,1]
    metrics=dict(full_rmse=float(np.sqrt(np.mean(err**2))),collar_rmse=float(np.sqrt(np.mean(err[mask]**2))),
        full_relative_l2=float(np.linalg.norm(err)/max(np.linalg.norm(exact),1e-12)),
        nodal_rmse=float(np.sqrt(np.mean((values-en)**2))),linear_full_rmse=float(np.sqrt(np.mean((linear-exact)**2))),
        independent_residual_relative=float(normres),condition_proxy=float(cond),algebraic_relative=algebraic,
        conservation_defect=bal2,conservation_quadrature_change=abs(bal2-bal),forcing_integral=vol,
        outward_Kgrad_integral=flux,separation_min=float(spacing.min()),separation_median=float(np.median(spacing)),
        dof=len(values),interior_minus=len(state.minus.interior),interior_plus=len(state.plus.interior),
        **core.coverage_metrics(points),**core.interface_diagnostics(problem,state.minus,state.plus,state.values_minus,state.values_plus,STENCIL))
    metrics['mesh_ratio_proxy']=metrics['coverage_fill_max']/(.5*metrics['separation_min'])
    metrics['valid_solve']=bool(all(np.isfinite(v) for v in metrics.values()) and algebraic<=1e-8)
    # Absolute, transparent screens: neither fitted to outcomes nor acceptance tests.
    metrics['screen_local']=bool(metrics['valid_solve'] and cond<=1e10 and normres<=.25)
    metrics['screen_conservation']=bool(bal2<=.05 and abs(bal2-bal)<=.01)
    metrics['screen_accuracy']=bool(metrics['full_relative_l2']<=.05)
    metrics['screen_joint']=bool(metrics['screen_local'] and metrics['screen_conservation'] and metrics['screen_accuracy'])
    return metrics,dict(evaluation_points=xy,evaluation_exact=exact,evaluation_phs=uh,residual_points=rp,residual=residual)

def robin(problem,state):
    """Secondary: fixed Q cross-side rule, no gradient modulation or retuning."""
    radius=problem.metadata['radius'];ni=len(state.minus.interface)
    lm=np.full(ni,problem.k_plus/(.5-radius));lp=np.full(ni,problem.k_minus/radius)
    sm=core.assemble_local_robin_system(problem,state.minus,lm,STENCIL,True)
    sp=core.assemble_local_robin_system(problem,state.plus,lp,STENCIL,True)
    um=sm.solve(np.zeros(ni));up=sp.solve(np.zeros(ni));history=[];ums=[];ups=[]
    for iteration in range(21):
        tm,tp=sm.trace(um),sp.trace(up);fm,fp=sm.flux(um),sp.flux(up)
        tr=float(np.sqrt(np.mean((tm-tp)**2)));fl=float(np.sqrt(np.mean((fm+fp)**2)))
        history.append(dict(iteration=iteration,trace_rms=tr,flux_rms=fl,physical_mismatch=math.hypot(tr,fl)))
        ums.append(um.copy());ups.append(up.copy())
        if iteration<20:
            newm=sm.solve(-fp+lm*tp);newp=sp.solve(-fm+lp*tm)
            um=.3*um+.7*newm;up=.3*up+.7*newp
    residual=np.maximum([r['physical_mismatch'] for r in history],1e-300)
    local=[]
    for system,u in ((sm,um),(sp,up)):
        nf=len(system.cloud.interior)+len(system.cloud.boundary)
        local.append(float(np.linalg.norm(system.matrix[:nf]@u-system.rhs_base[:nf])/max(np.linalg.norm(system.rhs_base[:nf]),1e-12)))
    result=SimpleNamespace(**vars(state));result.values_minus=um;result.values_plus=up
    return result,dict(robin_local_algebraic_relative=max(local),robin_condition_proxy=max(sm.condition_proxy,sp.condition_proxy),robin_q_net=float((residual[-1]/residual[0])**.05),
        robin_q_tail=float(np.median(residual[-5:]/residual[-6:-1])),robin_terminal=float(residual[-1])), dict(robin_values_minus=np.asarray(ums),robin_values_plus=np.asarray(ups),lambda_minus=lm,lambda_plus=lp),history

def state_arrays(state,prefix):
    return {prefix+'nodes_minus':state.minus.nodes,prefix+'nodes_plus':state.plus.nodes,
            prefix+'values_minus':state.values_minus,prefix+'values_plus':state.values_plus}

def execute_case(problem,family,method,n,seed,cfg,folder):
    """Each method starts from scratch; no cross-method timing subsidies."""
    nb,ni=boundary_counts(family,n);arrays={};stages=[]
    wall_start=time.perf_counter();times=dict(allocation_seconds=0.,assembly_seconds=0.,factor_solve_seconds=0.,indicator_seconds=0.)
    def allocate(fun):
        t=time.perf_counter();answer=fun();times['allocation_seconds']+=time.perf_counter()-t;return answer
    def solve_stage(points,label):
        state=solve(problem,points,nb,ni)
        times['assembly_seconds']+=state.assembly_seconds;times['factor_solve_seconds']+=state.factor_solve_seconds
        arrays.update(state_arrays(state,label+'_'))
        stages.append(dict(stage=label,cumulative_method_seconds=time.perf_counter()-wall_start,
            assembly_seconds=state.assembly_seconds,factor_solve_seconds=state.factor_solve_seconds))
        return state
    pilot_seconds=0.
    if method in ('HALTON','FROZEN_GHG'):
        points=allocate(lambda:core.global_halton(problem,n,seed));state=solve_stage(points,'halton' if method=='HALTON' else 'pilot')
        if method=='FROZEN_GHG':
            t=time.perf_counter();pilot=core.pilot_gradient_from_result(state);times['indicator_seconds']+=time.perf_counter()-t
            pilot_seconds=time.perf_counter()-wall_start
            length=min(.20,2.5/math.sqrt(n)) if family=='S' else .10
            points=allocate(lambda:core.protected_weighted_allocation(problem,n,seed,(1/3,)*3,pilot,
                skeleton_fraction=.45,green_length=length,candidate_multiplier=18)[0])
            state=solve_stage(points,'final')
    else:
        points,info=allocate(lambda:remesh(problem,n,seed,nb,ni));state=solve_stage(points,'uniform')
        if method.startswith('ADAPT_'):
            pilot_seconds=time.perf_counter()-wall_start
            for cycle in range(cfg['adaptive_cycles']):
                t=time.perf_counter();known,info=update_spacing(problem,state,n,seed,cycle,method)
                times['indicator_seconds']+=time.perf_counter()-t
                arrays.update({'adapt'+str(cycle)+'_'+k:np.asarray(v) for k,v in info.items()})
                points,info=allocate(lambda:remesh(problem,n,seed,nb,ni,known))
                arrays.update({'remesh'+str(cycle)+'_'+k:v for k,v in info.items()})
                state=solve_stage(points,'adapt'+str(cycle+1))
    primary_seconds=time.perf_counter()-wall_start
    metrics,diag=diagnostics(problem,state,points,cfg);arrays.update(diag)
    primary_audit_seconds=time.perf_counter()-wall_start
    metrics.update(times,pilot_seconds=pilot_seconds,wall_primary_solution_seconds=primary_seconds,
        wall_primary_audited_seconds=primary_audit_seconds,
        primary_diagnostics_seconds=primary_audit_seconds-primary_seconds,number_of_solves=len(stages))
    arrays.update(interior_points=points,**state_arrays(state,'final_'),matrix_data=state.matrix.data,
        matrix_indices=state.matrix.indices,matrix_indptr=state.matrix.indptr,matrix_shape=np.asarray(state.matrix.shape),rhs=state.rhs)
    history=[]
    if family=='Q':
        rt=time.perf_counter();rs,rm,ra,history=robin(problem,state)
        rm['robin_setup_iteration_seconds']=time.perf_counter()-rt
        # Cost already includes primary simultaneous work; do not call this the
        # standalone Robin solver time. Assembly algebraic defect is meaningful.
        rdiag,rd=diagnostics(problem,rs,points,cfg)
        rm.update({'robin_'+k:v for k,v in rdiag.items() if k not in ('condition_proxy','algebraic_relative','valid_solve','screen_local','screen_joint')})
        rm['robin_simultaneous_equation_residual']=rdiag['algebraic_relative']
        rm['robin_incremental_audited_seconds']=time.perf_counter()-rt
        rm['robin_screen_physical']=bool(rm['robin_q_net']<.95 and rm['robin_q_tail']<.95 and rm['robin_terminal']<=.005)
        rm['robin_error_ratio_to_simultaneous']=rdiag['full_rmse']/max(metrics['full_rmse'],1e-15)
        rm['robin_screen_local']=bool(rm['robin_condition_proxy']<=1e10 and rm['robin_local_algebraic_relative']<=1e-8 and rdiag['independent_residual_relative']<=.25)
        rm['robin_screen_joint']=bool(rm['robin_screen_physical'] and rm['robin_screen_local'] and rdiag['screen_conservation'] and rdiag['screen_accuracy'] and rm['robin_error_ratio_to_simultaneous']<=1.25 and rdiag['collar_rmse']<=1.25*max(metrics['collar_rmse'],1e-15))
        metrics.update(rm);arrays.update(ra);arrays.update({'robin_'+k:v for k,v in rd.items()})
    folder.mkdir(parents=True,exist_ok=True)
    # State I/O is measured, and intermediate adaptive states are preserved.
    io=time.perf_counter();tmp=folder/'state.partial'
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(folder/'state.npz');core.atomic_csv(folder/'stages.csv',stages)
    if history:core.atomic_csv(folder/'robin_history.csv',history)
    metrics['state_io_seconds']=time.perf_counter()-io
    metrics['wall_pipeline_to_state_write_seconds']=time.perf_counter()-wall_start
    metrics['state_sha256']=digest(folder/'state.npz')
    return metrics

def gaussian_problem():
    base=core.vertical_resistance_problem('GAUSSIAN_PREFLIGHT',.5,1.,1.)
    # Standard sharp Gaussian manufactured-Poisson family; NOT a reproduction
    # of the L-shaped or Helmholtz benchmark in Slak-Kosec's paper.
    a=100.;cx=.32;cy=.53
    def exact(x,side):return np.exp(-a*((x[:,0]-cx)**2+(x[:,1]-cy)**2))
    def forcing(x,side):
        r2=(x[:,0]-cx)**2+(x[:,1]-cy)**2
        return (4*a-4*a*a*r2)*np.exp(-a*r2)
    return dataclasses.replace(base,exact_function=exact,forcing_function=forcing,
                              metadata={**base.metadata,'validation_state':'Gaussian','a':a,'cx':cx,'cy':cy})

def verify_case(folder,protocol_id):
    p=folder/'metrics.json'
    if not p.exists():return None
    receipt=folder/'commit.json'
    if not receipt.exists():return None  # Interrupted before atomic commit: restart this case.
    if core.load_json(receipt).get('metrics_sha256')!=digest(p):raise RuntimeError('Case metrics changed: '+str(p))
    row=core.load_json(p)
    if row['protocol_hash']!=protocol_id:raise RuntimeError('Case protocol differs: '+str(folder))
    for name,sha in row.get('payload_sha256',{}).items():
        if not (folder/name).is_file() or digest(folder/name)!=sha:raise RuntimeError('Case payload changed: '+str(folder/name))
    if row['status']=='completed' and not row.get('payload_sha256'):raise RuntimeError('Missing payload provenance')
    return row

def run_record(problem,family,config,method,n,seed,cfg,folder,protocol_id,log):
    old=verify_case(folder,protocol_id)
    if old is not None:log('RESUME '+folder.name);return old
    row=dict(problem=problem.name,family=family,configuration=config,method=method,budget=n,seed=seed,
             protocol_hash=protocol_id,profile=cfg['profile'])
    folder.mkdir(parents=True,exist_ok=True);log('START '+folder.name)
    try:
        # Reproducible randomized condition estimation; method order cannot
        # contaminate the node generator or this random stream.
        np.random.seed(seed)
        row.update(execute_case(problem,family,method,n,seed,cfg,folder),status='completed')
    except Exception as error:
        row.update(status='failed',screen_joint=False,valid_solve=False,error=repr(error))
        (folder/'error.txt').write_text(traceback.format_exc())
    row['payload_sha256']={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name not in ('metrics.json','commit.json') and not p.name.endswith('.tmp')}
    write_json(folder/'metrics.json',row)
    write_json(folder/'commit.json',dict(metrics_sha256=digest(folder/'metrics.json'),protocol_hash=protocol_id))
    log(row['status'].upper()+' '+folder.name+'; error='+str(row.get('full_rmse','NA')))
    return row

def preflight(out,cfg,protocol_id,log):
    """No training or transfer outcomes are used to tune/validate the baseline."""
    p=gaussian_problem();pcfg={**cfg,'profile':'verification_only','eval_resolution':41}
    rows=[]
    for n in (120,240):
        for seed in (8081,8087):
            for method in ('QUASI_UNIFORM','ADAPT_VARIATION','ADAPT_RESIDUAL'):
                stem=f'GAUSSIAN_{method}_N{n}_S{seed}'
                rows.append(run_record(p,'S','PREFLIGHT',method,n,seed,pcfg,out/'preflight'/stem,protocol_id,log))
    d=pd.DataFrame(rows);d.to_csv(out/'preflight_metrics.csv',index=False)
    result={}
    for method in ('ADAPT_VARIATION','ADAPT_RESIDUAL'):
        ratios=[]
        for seed in (8081,8087):
            a=d[(d.method==method)&(d.seed==seed)&(d.budget==240)].iloc[0]
            b=d[(d.method=='QUASI_UNIFORM')&(d.seed==seed)&(d.budget==240)].iloc[0]
            ratios.append(float(a.get('full_rmse',np.inf))/float(b.get('full_rmse',1.)))
        result[method]=dict(fine_budget_median_error_ratio=float(np.median(ratios)),
            adequate=bool(np.all(np.isfinite(ratios)) and np.median(ratios)<1.0))
    finite=all(r.get('valid_solve',False) for r in rows)
    result['passed']=bool(finite and result['ADAPT_RESIDUAL']['adequate'])
    result['primary_adaptive_comparator']='ADAPT_RESIDUAL'
    result['variation_role']='Secondary diagnostic only: did not pass the fine-budget adequacy check during development. Never use its underperformance alone as evidence of GHG superiority.'
    result['development_decision']='Before any interface comparison: the initial requirement that both adaptive variants improve the fine-budget Gaussian failed. Residual passed; variation retained for transparency and excluded from primary superiority assessment. No thresholds or adaptive parameters were tuned after that check.'
    result['meaning']='Implementation/adequacy check on independent Gaussian Poisson family, not published-code reproduction or manuscript evidence.'
    write_json(out/'preflight_decision.json',result)
    return result

def sign_flip_p(logratios):
    x=np.asarray(logratios)
    if not len(x):return None
    observed=float(x.mean())
    means=np.asarray([np.mean(x*np.asarray(s)) for s in itertools.product((-1,1),repeat=len(x))])
    return float(np.mean(means<=observed+1e-15))

def summarize(rows,out,cfg,complete):
    """All failures remain in counts; paired ratios require both valid solves."""
    d=pd.DataFrame(rows);d.to_csv(out/'campaign_metrics.csv',index=False)
    if d.empty:return
    summaries=[];paired=[];seedrows=[]
    keys=['family','configuration','budget','seed']
    for group,g in d.groupby(['family','configuration','method','budget']):
        good=g[g.status=='completed']
        r=dict(zip(['family','configuration','method','budget'],group),attempted=len(g),
               completed=len(good),failed=len(g)-len(good),joint_passes=int(g.get('screen_joint',pd.Series(False,index=g.index)).fillna(False).sum()))
        for metric in ('full_rmse','collar_rmse','wall_primary_solution_seconds','wall_primary_audited_seconds',
                       'wall_pipeline_to_state_write_seconds','conservation_defect','independent_residual_relative'):
            r['median_'+metric]=float(good[metric].median()) if metric in good and len(good) else None
        summaries.append(r)
    pd.DataFrame(summaries).to_csv(out/'summary_by_budget.csv',index=False)
    for method in METHODS[1:]:
        a=d[d.method==method];b=d[d.method=='HALTON']
        merged=a.merge(b,on=keys,suffixes=('_challenger','_halton'))
        for _,r in merged.iterrows():
            valid=(r.status_challenger=='completed' and r.status_halton=='completed' and bool(r.get('valid_solve_challenger',False)) and bool(r.get('valid_solve_halton',False)))
            row={k:r[k] for k in keys};row.update(method=method,paired_valid=valid)
            for metric in ('full_rmse','collar_rmse','wall_primary_solution_seconds','wall_primary_audited_seconds'):
                x=r.get(metric+'_challenger',np.nan);y=r.get(metric+'_halton',np.nan)
                row[metric+'_ratio']=float(x/y) if valid and np.isfinite(x) and np.isfinite(y) and y>0 else None
            paired.append(row)
    pd.DataFrame(paired).to_csv(out/'paired_to_halton.csv',index=False)
    # Direct GHG versus each adaptive comparator; 4 budgets form ONE seed block.
    comparisons=[]
    for other in ('QUASI_UNIFORM','ADAPT_VARIATION','ADAPT_RESIDUAL'):
        for family in ('S','Q'):
            for config in ('TRAIN','TRANSFER'):
                sub=d[(d.family==family)&(d.configuration==config)]
                a=sub[sub.method=='FROZEN_GHG'];b=sub[sub.method==other]
                merged=a.merge(b,on=keys,suffixes=('_ghg','_base'))
                for metric in ('full_rmse','collar_rmse','wall_primary_solution_seconds','wall_primary_audited_seconds'):
                    blocks=[]
                    for seed,g in merged.groupby('seed'):
                        if len(g)!=len(cfg['budgets']):continue
                        if not all((g.status_ghg=='completed')&(g.status_base=='completed')):continue
                        if not all(g.valid_solve_ghg.fillna(False)&g.valid_solve_base.fillna(False)):continue
                        x=g[metric+'_ghg'].to_numpy(float);y=g[metric+'_base'].to_numpy(float)
                        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(x>0) and np.all(y>0):
                            lr=float(np.mean(np.log(x/y)));blocks.append(lr)
                            seedrows.append(dict(family=family,configuration=config,reference=other,metric=metric,seed=seed,mean_log_ratio=lr))
                    comparisons.append(dict(family=family,configuration=config,reference=other,metric=metric,
                        complete_seed_blocks=len(blocks),geometric_mean_seed_ratio=float(np.exp(np.mean(blocks))) if blocks else None,
                        descriptive_sign_flip_p=sign_flip_p(blocks),scope='exploratory; no multiplicity correction; positive log ratio favors comparator'))
    pd.DataFrame(comparisons).to_csv(out/'ghg_vs_adaptive_seed_blocks.csv',index=False)
    pd.DataFrame(seedrows).to_csv(out/'seed_block_values.csv',index=False)
    # Equal-error work: raw nondominated envelopes, no extrapolation or timing
    # interpolation. Can inspect what error is actually reached within each cost.
    front=[]
    for key,g in d[d.status=='completed'].groupby(['family','configuration','method','seed']):
        for _,r in g.iterrows():
            t=r.get('wall_primary_solution_seconds',np.nan);e=r.get('full_rmse',np.nan)
            dominated=any((g.wall_primary_solution_seconds<=t)&(g.full_rmse<=e)&((g.wall_primary_solution_seconds<t)|(g.full_rmse<e)))
            front.append(dict(zip(['family','configuration','method','seed'],key),budget=r.budget,time=t,error=e,pareto_nondominated=not dominated))
    pd.DataFrame(front).to_csv(out/'error_cost_pareto.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,(family,config) in zip(axs.flat,itertools.product(('S','Q'),('TRAIN','TRANSFER'))):
        for method in METHODS:
            g=d[(d.family==family)&(d.configuration==config)&(d.method==method)&(d.status=='completed')]
            if not len(g):continue
            med=g.groupby('budget')[['wall_primary_solution_seconds','full_rmse']].median()
            ax.loglog(med.wall_primary_solution_seconds,med.full_rmse,'o-',label=method)
        ax.set_title(family+' | '+config);ax.set_xlabel('Complete method-to-solution wall time (s)');ax.set_ylabel('PHS reconstructed full RMSE');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7);fig.savefig(out/'error_cost.png',dpi=160);plt.close(fig)
    write_json(out/'decision_scope.json',dict(complete=complete,production_evidence=bool(complete and cfg['profile']=='production'),
        automatically_selected_winner=None,reason='No retuning or pooled-row automatic superiority declaration. Inspect absolute failures, seed-block comparisons and measured cost envelopes.',
        transfer_status='Previously inspected geometries; descriptive transfer, not a pristine holdout of a newly tuned selector.'))

def main(profile=PROFILE,output=None,stop_after=0,preflight_only=False):
    cfg=settings(profile)
    target=output or OUTPUT_DIRECTORY
    out=Path(target).expanduser().resolve() if target else HERE/f'results_adaptive_comparison_{profile}_v1'
    out.mkdir(parents=True,exist_ok=True)
    # OS-held lock releases on interruption/crash. No stale lock-file deletion.
    import fcntl
    with (out/'run.lock').open('a') as lock, threadpool_limits(limits=1):
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process is running in this output folder')
        sources={p.name:digest(p) for p in (Path(__file__),HERE/'CGF_Replay_Common_PHS_RBFFD.py',HERE/'CGF_Nonquadratic_Coupling_Validation.py')}
        env=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
                 pandas=pd.__version__,matplotlib=matplotlib.__version__,threadpools=threadpool_info())
        contract=dict(version=VERSION,settings=cfg,methods=METHODS,source_sha256=sources,
            stencil=STENCIL,phs_power=3,polynomial_degree=2,collar=COLLAR,pool_multiplier=POOL_MULTIPLIER,
            low_quantile=LOW_QUANTILE,high_quantile=HIGH_QUANTILE,refine_factor=REFINE_FACTOR,coarsen_factor=COARSEN_FACTOR,
            spacing_interpolation_neighbors=NEIGHBORS,spacing_clamp_relative_median=[.25,4.],
            fixtures=[dict(family=f,configuration=c,name=p.name,k_minus=p.k_minus,k_plus=p.k_plus,metadata=p.metadata) for f,c,p in fixtures()],
            required_environment={k:env[k] for k in ('python','platform','numpy','scipy','pandas','matplotlib','threadpools')},
            timing='Each method independently runs all required work. Primary solution includes every allocation, pilot and adaptation solve. Audited primary adds diagnostics. Q pipeline adds fixed Robin and its diagnostics. Pipeline-to-state-write includes state persistence; imports, preflight, logs, final metadata commit, aggregation, ZIP are outside case timing.',
            comparisons='Fixed rules; no selector tuning or best-iterate choice. TRAIN precedes previously inspected TRANSFER geometries.',
            baseline='Local SK-style spacing/variation and residual variants; exact-N maximin remeshing, not author software reproduction.')
        # Strip process-specific library paths from identity: package versions and
        # BLAS implementation/version/thread counts are what matter for resume.
        contract['required_environment']['threadpools']=[{k:v for k,v in r.items() if k not in ('filepath','prefix')} for r in env['threadpools']]
        protocol_id=core.protocol_hash(contract);contract['protocol_hash']=protocol_id
        if (out/'protocol.json').exists():
            if core.load_json(out/'protocol.json')['protocol_hash']!=protocol_id:
                raise RuntimeError('Protocol/source/environment changed. Use a NEW output directory; do not mix evidence.')
        write_json(out/'protocol.json',contract);write_json(out/'environment.json',env)
        src=out/'source';src.mkdir(exist_ok=True)
        for name in sources:shutil.copy2(HERE/name,src/name)
        log=core.OrderedLog(out/'run.log');log('BEGIN '+VERSION+' '+profile+' protocol='+protocol_id)
        adequacy=preflight(out,cfg,protocol_id,log)
        if not adequacy['passed']:
            write_json(out/'needs_comparator_review.json',dict(reason='Independent comparator adequacy check did not pass; production not started.',details=adequacy))
            raise RuntimeError('Comparator adequacy check failed. Upload this output folder for diagnosis; do not retune on transfer outcomes.')
        if preflight_only:log('PREFLIGHT COMPLETE; no production requested');return out
        rows=[];expected=4*len(cfg['budgets'])*len(cfg['static_seeds'])*len(METHODS)
        count=0
        for family,config,problem in fixtures():
            for n in cfg['budgets']:
                seeds=cfg['static_seeds'] if family=='S' else cfg['circular_seeds']
                for seed in seeds:
                    # Counterbalance execution order; never select order using errors.
                    offset=(seed+n)%len(METHODS);order=METHODS[offset:]+METHODS[:offset]
                    for method in order:
                        stem=f'{family}_{config}_{method}_N{n}_S{seed}'
                        rows.append(run_record(problem,family,config,method,n,seed,cfg,out/'cases'/stem,protocol_id,log));count+=1
                        if stop_after and count>=stop_after:
                            summarize(rows,out,cfg,False);log('PAUSED by --stop-after; rerun without it to resume');return out
            summarize(rows,out,cfg,False)
        summarize(rows,out,cfg,True)
        write_json(out/(profile+'_complete.json'),dict(expected_cases=expected,recorded_cases=len(rows),failed_cases=sum(r['status']=='failed' for r in rows),
            all_planned_cases_attempted=len(rows)==expected,all_cases_succeeded=all(r['status']=='completed' for r in rows),protocol_hash=protocol_id))
        log('COMPLETE '+str(len(rows))+' cases. Include unsuccessful outcomes when uploading.')
        core.atomic_csv(out/'scientific_manifest_sha256.csv',core.scientific_manifest(out,exclusions=('scientific_manifest_sha256.csv','run.lock')))
    # Full results ZIP beside directory. Timing excludes packaging.
    shutil.make_archive(str(out),'zip',root_dir=out.parent,base_dir=out.name)
    print('Upload:',str(out)+'.zip')
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',choices=('screen','production'),default=PROFILE)
    ap.add_argument('--output',default=None)
    ap.add_argument('--stop-after',type=int,default=0,help='Intentional pause for verification; not part of scientific protocol')
    ap.add_argument('--preflight-only',action='store_true')
    args,_=ap.parse_known_args()
    main(args.profile,args.output,args.stop_after,args.preflight_only)
