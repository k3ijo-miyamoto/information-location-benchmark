"""Development-only corpus-paired INR and prospective SD-only allocation."""
from math import sqrt
from statistics import mean, median, stdev
from .generator_v13_learner_plan_v01 import required_corpora, R_GRID, CRITICAL, SD_INFLATION
from .native_stage_v_common_v13 import identities, run_id, NS


def describe(values):
    return dict(values=values, mean=mean(values), median=median(values),
                sample_sd=stdev(values), min=min(values), max=max(values))


def analyze(records):
    expected = {run_id(i) for i in identities()}
    if len(records) != len(expected) or {r['run_id'] for r in records} != expected:
        raise ValueError('incomplete/duplicate Stage V; allocation blocked')
    if any(r['result']['status'] != 'success' for r in records):
        raise ValueError('failed Stage V; allocation blocked')
    values = {}
    for row in records:
        i = row['identity']
        if run_id(i) != row['run_id']:
            raise ValueError('identity mismatch')
        values[(i['cell'], i['corpus'], i['init'], i['family'], i['N'])] = row['metadata']['evaluation']['test']['normalized_regret']
    inr = {}
    for cell, corpus, init, family, _ in values:
        key = (cell, corpus, init, family)
        inr[key] = sum(w*values[(*key, n)] for w,n in zip((.5,1,1,.5), NS))/3

    def contrast(cell, left, right):
        paired = [[inr[(cell,c,i,left)]-inr[(cell,c,i,right)] for i in range(3)] for c in range(6)]
        summary = describe([mean(p) for p in paired])
        summary.update(cell=cell, left=left, right=right, paired_initialization_differences=paired,
                       within_corpus_initialization_sd=describe([stdev(p) for p in paired]),
                       independent_units=6, development_only=True)
        return summary

    primary = {}
    for number, regime, left, right in ((1,1,'tabular','static'),(2,2,'static','temporal'),(3,3,'static','temporal')):
        for opportunity in ('O1','O2'):
            primary[f'C{number}-{opportunity}'] = contrast(f'IS{regime}-{opportunity}',left,right)
    controls = {}
    for opportunity in ('O1','O2'):
        for left,right in (('tabular','static'),('static','temporal'),('static_size_control','temporal')):
            row = contrast('IS0-'+opportunity,left,right)
            row['caution_margin'] = .10
            row['caution'] = abs(row['mean']) > .10
            controls[f'{opportunity}:{left}-{right}'] = row
    sds = {name: row['sample_sd'] for name,row in primary.items()}
    selected = required_corpora(list(sds.values()))
    allocation = dict(version='stage-c-allocation-0.1', decision='SCA-A', six_sd_inputs=sds,
                      max_sd=max(sds.values()), sd_inflation=SD_INFLATION,
                      candidate_half_widths={str(r): CRITICAL[r]*max(sds.values())*SD_INFLATION/sqrt(r) for r in R_GRID},
                      selected_R=selected['corpora'], precision_feasible=selected['precision_feasible'],
                      future_corpus_seed_count=8*selected['corpora'], future_init_seed_count=3,
                      stage_c_seeds_reserved=0, stage_c_runs=0, mean_effects_used=False,
                      p_values_used=False, winners_used=False, negative_controls_used=False)
    runtimes = {f'{family}:{n}': median(r['result']['runtime_seconds'] for r in records
                if r['identity']['family']==family and r['identity']['N']==n)
                for family in ('tabular','static','temporal','static_size_control') for n in NS}
    analysis = dict(version='stage-v-analysis-0.1', development_only=True,
                    primary=primary, negative_controls=controls,
                    total_run_runtime_seconds=sum(r['result']['runtime_seconds'] for r in records),
                    median_runtime_seconds_by_family_N=runtimes,
                    total_optimizer_steps=sum(r['metadata']['optimizer_steps'] for r in records),
                    total_example_presentations=sum(r['metadata']['example_presentations'] for r in records),
                    max_allocated_gpu_memory_bytes=max(r['metadata']['max_allocated_gpu_memory_bytes'] for r in records),
                    confirmatory_claim=False, protocol_changes_from_results=False)
    if analysis['total_optimizer_steps'] != 19584000 or analysis['total_example_presentations'] != 626688000:
        raise ValueError('training accounting mismatch')
    return analysis, allocation
