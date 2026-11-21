"""Fixed identities and matched-panel sampling shared by the plaque analyses."""

import numpy as np


NUM = ["CTSD", "TPP1"]
DEN = [
    "ATP6V0A1",
    "ATP6V1A",
    "ATP6V1B2",
    "ATP6V1C1",
    "ATP6V1D",
    "ATP6V1E1",
    "ATP6V1G2",
    "ATP6V1H",
]
ACCESSIONS = {
    "CTSD": "P07339",
    "TPP1": "O14773",
    "ATP6V0A1": "Q93050",
    "ATP6V1A": "P38606",
    "ATP6V1B2": "P21281",
    "ATP6V1C1": "P21283",
    "ATP6V1D": "Q9Y5K8",
    "ATP6V1E1": "P36543",
    "ATP6V1G2": "O95670",
    "ATP6V1H": "Q9UI12",
}


def draw_panels(pools, count, rng):
    """Sample uniformly per slot, conditional on having ten distinct genes."""
    if min(map(len, pools)) == 0 or len(np.unique(np.concatenate(pools))) < len(pools):
        raise ValueError("Not enough eligible genes to draw distinct matched panels")
    batches = []
    accepted = proposed = 0
    budget = count * 100
    while accepted < count and proposed < budget:
        batch_size = min(max(count - accepted, 256), 8192, budget - proposed)
        draws = np.column_stack([rng.choice(pool, size=batch_size) for pool in pools])
        unique = (np.diff(np.sort(draws, axis=1), axis=1) != 0).all(axis=1)
        batches.append(draws[unique][: count - accepted])
        accepted += len(batches[-1])
        proposed += batch_size
    if accepted < count:
        raise ValueError(
            f"Only {accepted}/{count} distinct-within-panel draws after {proposed} attempts"
        )
    return np.concatenate(batches)
