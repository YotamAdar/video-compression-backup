# Baseline implementation notes

Inspected local checkout: `~/Turbo-DDCM`, HEAD
`77353e47383a6c10b00587629f65cc73c1464266`. Inspection and project setup do not
change that checkout.

## Compression

`turbo_ddcm/turbo_ddcm.py` provides `TurboDDCM` and a stateful step-by-step API.
The source image is encoded using the VAE posterior mode. A shared seed generates
the initial latent. For each encoding step, the model predicts a clean latent;
the difference from the reference latent is the selection residual. The codebook
is regenerated with seed `seed + step + 1`.

Selection computes correlations with all K Gaussian atoms, chooses M largest
absolute correlations, uses their signs as coefficients (C=1), and divides the
sum by its standard deviation. The reverse update uses eta=1. Later steps use
eta=0 without transmitted indices, including the final step. The no-bit-step
count comes from a resolution-dependent bitrate heuristic in `utils.py`.

The codebook is allocated densely in GPU memory. No attempt is made here to
optimize it or change the underlying algorithm. The adapter uses inference mode
to avoid autograd bookkeeping, and selects the current CUDA device because
upstream sets `self.device = 'cuda'` even when a device argument is supplied.

## Diffusion schedule

`ddpm.py` loads a Stable Diffusion pipeline and DDIM scheduler, then replaces
timesteps with a subsample of the 162-entry `utils.SCHEDULER` list. It implements
its own prediction conversion, previous-timestep lookup, variance, and reverse
update. The final previous timestep is zero. These choices are part of the
baseline; calling a stock scheduler step would not be equivalent.

`flux.py` uses packed latents and converts between flow-matching and DDPM-like
quantities. It shares the schedule list but uses its own reverse updates. Keep
that path distinct from experiments with the Stable Diffusion schedule.

## Bitstream

`bit_stream_lex_enc.py` sorts the selected atom indices and reorders their signs.
Each step stores `ceil(log2(comb(K, M)))` bits for a lexicographic combination rank
and M sign bits. `utils.save_as_binary` writes big-endian bytes with leading zero
padding (despite a misleading little-endian comment). The decoder strips padding
using the known fixed step length. It requires more than eight bits per step.

The payload has no header, dimensions, model identity, seed, or step count. Keep
the sidecar and exact implementation available. The shared model and seed are
essential to regenerating noise; these are not a conventional entropy-coded
image residual.

When a combination table is absent, construction writes it into the upstream
`combs_files` directory. The adapter refuses that case before importing the
model. Existing tables are reused; it never generates a missing one upstream.
New research encoders should place generated tables under an ignored project
cache. Precomputed tables are trusted local pickle files.

## Local modifications and interpretation

Ignoring line-ending differences, the checkout differs from HEAD in two places:

1. `compress.py` chooses `Manojb/stable-diffusion-2-1-base` for the 512 path.
2. `utils.get_no_bits_steps` sets `nbs=23` for M=40, T=30, H=W=512,
   giving six encoding steps, regardless of the normal heuristic.

The tracked default is M=30. Changing to M=40 retains this local override.
Do not describe the local checkout as an unmodified published baseline. Runs save
source hashes and diffs in addition to the Git commit. The M=40 table is currently
untracked upstream and is hashed separately in run metadata.

The original entry scripts have implicit resize/model selection and incomplete
handling at some size boundaries. The adapter calls the class directly with an
explicit model ID and common preprocessing. It does not promise identical CLI
behavior at those boundaries. Priority maps are not exposed in this first adapter.

## Research extension points

- Atom selection and coefficient quantization: `algorithms/`.
- Encoding/no-bit-step allocation and reverse updates: `schedules/`.
- Rank coding, headers, and temporal side information: `bitstreams/`.
- Frame dependencies and latent state: a new temporal codec with an explicit
  sequence contract when needed.

Validate new methods by decoding persisted bytes and comparing at matched actual
rates, using the same input manifest, preprocessing, model weights, seeds, and
hardware where applicable. Run GPU equivalence and roundtrip checks only after
an experiment is authorized.
