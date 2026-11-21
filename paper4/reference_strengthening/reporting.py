"""Inference families recorded in Table S65."""


def inference_map():
    return [dict(analysis="historical_manuscript",interval="nominal t or paired bootstrap",multiplicity="unchanged; historical exploratory scope"),
        dict(analysis="fixed_size_ten_panels",interval="nominal bootstrap plus Bonferroni t",multiplicity="Holm p and Bonferroni intervals within ten panels per group"),
        dict(analysis="harmonized_target_references",interval="nominal bootstrap plus Bonferroni t",multiplicity="within reference family per group, scope and definition; not across all specifications"),
        dict(analysis="background_margin_fractions",interval="pointwise paired donor percentile bootstrap",multiplicity="descriptive fixed-background proportions; no protein-level discovery control"),
        dict(analysis="simulation",interval="Wilson Monte Carlo intervals",multiplicity="Bonferroni strategy across five references; availability and conditional performance separate")]
