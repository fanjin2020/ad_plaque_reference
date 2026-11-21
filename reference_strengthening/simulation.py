"""Known-estimand simulations; failure to estimate is reported, never discarded silently."""
import itertools
import math
import numpy as np
from scipy.special import expit
from scipy.stats import t,binom

from paper4.reference_sensitivity.analysis import random_generator


def coefficients(config):
    panels=config["reference_indices"]
    targets=config["target_indices"]
    size=len(next(iter(config["truths"].values())))
    matrix=np.zeros((size,len(panels)))
    for j,panel in enumerate(panels):
        if set(panel)&set(targets) or len(set(panel))!=len(panel):
            raise ValueError("Invalid synthetic panel")
        matrix[targets,j]=1/len(targets)
        matrix[panel,j]=-1/len(panel)
    np.testing.assert_allclose(matrix.sum(axis=0),0,atol=1e-12)
    return matrix


def bernoulli_summary(events):
    n=len(events)
    if not n:
        return dict(rate=None,mcse=None,mc_low=None,mc_high=None)
    p=float(np.mean(events)); z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(rate=p,mcse=math.sqrt(p*(1-p)/n),mc_low=max(0.,center-half),mc_high=min(1.,center+half))


def correlated_changes(rng, shape, means, sd, rho):
    return means + sd*(math.sqrt(rho)*rng.normal(size=(*shape,1))
                       + math.sqrt(1-rho)*rng.normal(size=(*shape,len(means))))


def calibrate_mnar(config, means, rho, rate, seed, truth_name):
    rng=random_generator(seed,"mnar_calibration",truth_name,rho,rate)
    delta=correlated_changes(rng,(config["calibration_draws"],),means,config["sd_regional_change"],rho)
    # Centered plaque intensity = centered adjacent intensity + regional change.
    latent_plaque=rng.normal(0,config["sd_adjacent_intensity"],size=delta.shape)+delta
    lo,hi=-30.,30.
    for _ in range(45):
        middle=(lo+hi)/2
        probability=expit(middle-config["mnar_logistic_slope"]*latent_plaque)
        if probability.mean()<rate:
            lo=middle
        else:
            hi=middle
    intercept=(lo+hi)/2
    expected=float(expit(intercept-config["mnar_logistic_slope"]*latent_plaque).mean())
    return intercept,expected


def fit_replicates(delta, observed, matrix, minimum):
    common=observed.all(axis=-1)
    counts=common.sum(axis=1)
    usable=counts>=minimum
    scores=delta @ matrix
    denominator=np.maximum(counts,1)
    estimates=(scores*common[:,:,None]).sum(axis=1)/denominator[:,None]
    squares=(((scores-estimates[:,None,:])**2)*common[:,:,None]).sum(axis=1)
    se=np.sqrt(squares/np.maximum(counts-1,1)[:,None]/denominator[:,None])
    usable &= np.isfinite(se).all(axis=1) & (se>0).all(axis=1)
    means=estimates[usable]; se=se[usable]; df=counts[usable]-1
    pvalues=2*t.sf(np.abs(means/se),df[:,None])
    nominal=t.ppf(.975,df)[:,None]*se
    bonferroni=t.ppf(1-.05/(2*matrix.shape[1]),df)[:,None]*se
    return counts,usable,means,se,pvalues,nominal,bonferroni


def performance_rows(base, means, truth, pvalues, nominal, bonferroni):
    n,m=means.shape
    choice=np.argmin(pvalues,axis=1) if n else np.zeros(0,dtype=int)
    index=np.arange(n)
    errors=means-truth[None,:]
    methods=[
        ("fixed_reference_nominal",errors[:,0:1],nominal[:,0:1],pvalues[:,0]<.05,"selected_interval"),
        ("selected_reference_nominal",errors[index,choice,None],nominal[index,choice,None],
         pvalues[index,choice]<.05,"selected_interval"),
        ("all_reference_nominal",errors,nominal,(pvalues<.05).any(axis=1),"joint_family"),
        ("all_reference_bonferroni",errors,bonferroni,(pvalues<.05/m).any(axis=1),"joint_family")]
    rows=[]
    for method,error,width,rejected,coverage_scope in methods:
        covered=(np.abs(error)<=width).all(axis=1)
        row=dict(**base,method=method,n_estimable=n,coverage_scope=coverage_scope,
            simulation_precision_note="fewer_than_100_estimable_replicates" if n<100 else "",
            bias=float(error.mean()) if n else None,
            rmse=float(np.sqrt(np.mean(error**2))) if n else None,
            mean_absolute_error=float(np.abs(error).mean()) if n else None,
            mean_interval_width=float(2*width.mean()) if n else None,
            bias_mcse=float(error.mean(axis=1).std(ddof=1)/math.sqrt(n)) if n>1 else None,
            success_and_coverage_rate=float(covered.sum()/base["repetitions"]))
        row.update({"coverage_"+k:v for k,v in bernoulli_summary(covered).items()})
        row.update({"any_rejection_"+k:v for k,v in bernoulli_summary(rejected).items()})
        row["false_rejection_rate_all_null"]=row["any_rejection_rate"] if base["truth"]=="all_null" else None
        rows.append(row)
    per_reference=[]
    for j in range(m):
        row=dict(**base,reference="R"+str(j+1),true_effect=float(truth[j]),n_estimable=n,
            simulation_precision_note="fewer_than_100_estimable_replicates" if n<100 else "",
            bias=float(errors[:,j].mean()) if n else None,
            rmse=float(np.sqrt(np.mean(errors[:,j]**2))) if n else None,
            empirical_sd=float(means[:,j].std(ddof=1)) if n>1 else None)
        row.update({"nominal_coverage_"+k:v for k,v in bernoulli_summary(np.abs(errors[:,j])<=nominal[:,j]).items()})
        row.update({"bonferroni_marginal_coverage_"+k:v for k,v in bernoulli_summary(np.abs(errors[:,j])<=bonferroni[:,j]).items()})
        row.update({"selection_"+k:v for k,v in bernoulli_summary(choice==j).items()})
        per_reference.append(row)
    if n:
        assert np.array_equal(pvalues[index,choice]<.05,(pvalues<.05).any(axis=1))
        assert np.all(bonferroni>=nominal)
    return rows,per_reference


def simulate(config,repetitions,seed):
    matrix=coefficients(config)
    performance,details,availability,calibrations,truths=[],[],[],[],[]
    calibrated={}
    for truth_name, mean_list in config["truths"].items():
        means=np.array(mean_list,dtype=float)
        for j,effect in enumerate(means @ matrix):
            truths.append(dict(truth=truth_name,reference="R"+str(j+1),true_effect=float(effect),
                targets="|".join(map(str,config["target_indices"])),
                reference_indices="|".join(map(str,config["reference_indices"][j]))))
        for rho in config["correlations"]:
            for mechanism,rate in config["missingness"]:
                if mechanism!="MNAR":
                    continue
                intercept,expected=calibrate_mnar(config,means,rho,rate,seed,truth_name)
                calibrated[truth_name,rho,rate]=intercept
                calibrations.append(dict(truth=truth_name,rho=rho,requested_missing_fraction=rate,
                    calibrated_missing_fraction=expected,logistic_intercept=intercept,
                    logistic_slope=config["mnar_logistic_slope"],calibration_draws=config["calibration_draws"]))
    scenarios=list(itertools.product(config["truths"],config["sample_sizes"],config["correlations"],config["missingness"]))
    for number,(truth_name,n,rho,(mechanism,rate)) in enumerate(scenarios,1):
        means=np.array(config["truths"][truth_name],dtype=float)
        rng=random_generator(seed,"performance_simulation",truth_name,n,rho,mechanism,rate,repetitions)
        delta=correlated_changes(rng,(repetitions,n),means,config["sd_regional_change"],rho)
        if mechanism=="complete":
            observed=np.ones(delta.shape,dtype=bool)
        elif mechanism=="MCAR":
            observed=rng.random(delta.shape)>=rate
        elif mechanism=="MNAR":
            plaque=rng.normal(0,config["sd_adjacent_intensity"],size=delta.shape)+delta
            probability=expit(calibrated[truth_name,rho,rate]-config["mnar_logistic_slope"]*plaque)
            observed=rng.random(delta.shape)>=probability
        else:
            raise ValueError("Unknown missingness mechanism")
        counts,usable,est,se,p,nom,bonf=fit_replicates(delta,observed,matrix,config["minimum_complete_donors"])
        base=dict(truth=truth_name,n=n,rho=rho,missingness=mechanism,
            requested_missing_fraction=rate,repetitions=repetitions)
        a,b=performance_rows(base,est,means @ matrix,p,nom,bonf)
        performance.extend(a); details.extend(b)
        expected = (1. if mechanism=="complete" else
            float(binom.sf(config["minimum_complete_donors"]-1,n,(1-rate)**len(means))) if mechanism=="MCAR" else None)
        availability.append(dict(**base,n_estimable=int(usable.sum()),
            n_unavailable=int((~usable).sum()),estimable_fraction=float(usable.mean()),
            simulation_precision_note="fewer_than_100_estimable_replicates" if usable.sum()<100 else "",
            analytic_estimable_fraction=expected,
            analytic_expected_estimable_count=expected*repetitions if expected is not None else None,
            missing_fraction_observed=float((~observed).mean()),
            mean_complete_donors=float(counts.mean()),median_complete_donors=float(np.median(counts)),
            minimum_complete_donors_required=config["minimum_complete_donors"],
            **{"failure_"+k:v for k,v in bernoulli_summary(~usable).items()}))
        if number%10==0 or number==len(scenarios):
            print(f"Simulation {number}/{len(scenarios)}: {truth_name}, n={n}, rho={rho}, {mechanism} {rate}",flush=True)
    return {"03_simulation_performance":performance,"03_simulation_per_reference":details,
        "03_simulation_availability":availability,"03_mnar_calibration":calibrations,"03_known_truth":truths}
