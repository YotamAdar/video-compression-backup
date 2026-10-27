# Video compression research

A separate research workspace that reuses `~/Turbo-DDCM` as an external baseline.
The baseline currently compresses images independently; this project does not yet
implement a temporal video codec. Extracted PNG frames are the initial input.

## Environment and paths

Reuse the existing `turbo_ddcm` Conda environment and shared model cache. Nothing
needs to be installed to use the source layout:

```bash
cd ~/video-compression
conda activate turbo_ddcm
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cp configs/paths.example.json configs/paths.local.json
```

Edit `configs/paths.local.json` to point at existing PNG frames and a run directory.
It is ignored by Git. JSON paths should be absolute or begin with `~`; shell
variables such as `$HOME` inside JSON are not expanded. Do not copy datasets,
weights, or environments into this repository. Keep using the existing Hugging
Face cache; the adapter forces offline mode, so absent weights cause an error
instead of a download.

Inspect the source identity and prerequisite tables without importing models:

```bash
python -m video_compression inspect
```

## Layout

- `configs/`: tracked experiment parameters; ignored machine-specific paths.
- `src/video_compression/codecs/`: common interface and external baseline adapter.
- `src/video_compression/algorithms/`: new compression methods.
- `src/video_compression/schedules/`: experimental timesteps and reverse updates.
- `src/video_compression/bitstreams/`: serialization and native baseline byte packing.
- `src/video_compression/evaluate.py`: shared roundtrip evaluation.
- `scripts/slurm/`: allocation template, with cluster settings supplied at submission.
- `tests/`: small CPU checks that do not load diffusion models.
- `runs/`: generated artifacts, ignored by Git.

## Experiments on SLURM

The template requests one GPU. Supply the partition, account, and QOS used on your
Gipdeep allocation; those values are intentionally not guessed. Initialize Conda
in the submitting shell so `conda run` is available in the batch job.

After selecting the allocation settings and approving an experiment, submit:

```bash
sbatch scripts/slurm/evaluate.sbatch configs/baseline.json configs/paths.local.json baseline-001
```

For an existing interactive GPU allocation, the equivalent command is:

```bash
python -m video_compression evaluate --name baseline-001
```

Both commands run GPU inference. VS Code Remote Tunnels provides access to the
workspace; a GPU allocation still comes from SLURM. Use `cuda:0` for the first
GPU exposed by the allocation. No experiment has been submitted during setup.

The run name must be new. Each run saves:

- `metadata.json`: resolved settings, source hashes and diffs, environment,
  input hashes, effective schedule, GPU details, and completion/failure status.
- `decoder_config.json`: shared settings needed with the native payload files.
- `bitstreams/` and `reconstructions/`: saved payloads and decoded PNG frames.
- `metrics.json` and `summary.json`: actual bytes, bitrate, MSE, PSNR, and timing.

The entire run directory is ignored. Curate small result summaries into a tracked
directory when they are ready to accompany a research result. Commit code and
experiment configurations before running comparisons. Source hashes also identify
uncommitted Python code, but a clean commit is easier to reproduce.

## Developing a method

Implement `encode(image) -> bytes`, `decode(payload) -> image`, and `metadata()`
following `codecs/base.py`. Return RGB tensors in `[-1, 1]` with shape
`[1, 3, H, W]`. Metadata currently needs `width` and `height`. Add the codec to
`build_codec()` in `evaluate.py`, then create a tracked configuration.

Start with direct functions and small classes. Reuse the existing model wrappers
where helpful, and put changed selection logic, schedules, and formats in this
repository. Avoid changing the baseline or globally monkey-patching its methods.
Document any reused code and preserve its upstream license attribution if code
is later copied.

The first evaluation runner deliberately targets Turbo-DDCM's native configuration
and `.turbo_ddcm` payloads. A different codec will also need its format/sidecar
handling and baseline-specific provenance factored out of `evaluate()`; the
`Codec` protocol does not assume a shared bitstream format. Introduce a sequence
interface when the first temporal method needs explicit frame dependencies.

## Comparison conventions and limits

All PNG inputs are sorted and resized to the explicitly selected model size using
the baseline's RGB/Pillow preprocessing. Metrics compare decoded, quantized RGB
against the resized input, not against an original-resolution image. Original and
evaluated sizes are recorded. There is no implicit model selection from input size.

Rate uses bytes read from saved payloads. A second rate includes one shared decoder
sidecar amortized across the run; it excludes the research provenance report,
pre-shared model weights, and source code. This is not a self-contained container.

Timing uses synchronized wall-clock measurements. Baseline encode time includes
its mandatory reconstruction and serialization, unlike the original script's
partial CPU-process-time measurement. Decode time excludes file IO. Model loading
is excluded; there is no warmup, so first-frame overhead is retained. Use matched
hardware and timing scope for comparisons; add repeated trials for timing studies.

Model commit metadata can be null if the loaded pipeline does not expose its
revision. Exact weights must then be identified separately before claiming a
fully reproducible result. No cross-device or cross-version bitwise determinism
is assumed. GPU roundtrips, the FLUX path, and optional float32 mode have not been
validated during setup.

## CPU checks

No packages are needed for these checks beyond Python and the external source:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
bash -n scripts/slurm/evaluate.sbatch
```

See [docs/baseline.md](docs/baseline.md) for the implementation study and baseline
restrictions.
