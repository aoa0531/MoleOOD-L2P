# MoleOOD + length-one additive prompts (supervised IIL)

This implementation keeps the local PyG model's **complete graph + mean
substructure** fusion. It does not restore the paper's attention pooling.
It adapts L2P's retrieval mechanism, not its Transformer token concatenation.

## Data flow

1. D0 trains the original environment inference and invariant main-model losses.
2. Freeze both encoders (including BatchNorm statistics and dropout).
3. Encode each clean molecule without prompts under `no_grad` to obtain a query.
4. Cosine-match the query against independently learned keys; select Top-1.
5. Add its `[d]` prompt to every atom embedding belonging to that molecule,
   before the first message-passing layer. Reuse the same frozen encoder, but
   keep autograd enabled so classification gradients reach the prompt.
6. Add the mean substructure representation, then apply the trainable predictor.
7. Incremental loss = class-weighted cross entropy + weighted key matching loss.

Prompt and key arrays both have shape `[pool_size, emb_dim]`. Both use separate
uniform draws in `[-prompt_init_scale, +prompt_init_scale]`. This explicit range
is a graph-adaptation choice, not a claim about Flax's default initializer range.
There are no masks, atom decoders, dynamically allocated prompts, task IDs, or
test-time adaptation. The environment models are only trained/used in D0.

## Run

Run commands from this directory. Config-relative data/output paths resolve
against this directory, regardless of the shell working directory.

The local `.venv-iil` test environment inherits Torch/scikit-learn from the
existing `pytorch` Conda environment and adds PyG/RDKit. Alternatively use your
own environment containing torch, torch-geometric, rdkit, numpy, scikit-learn.
The verified package versions are listed in `requirements-iil.txt`.

```powershell
# Only needed for a new manifest (the real scaffold manifest is already supplied).
.venv-iil/Scripts/python.exe prepare_iil_split.py --data data/ec50/lbap_core_ec50_scaffold_brics.json --output splits/new_split.json

# Full experiment; CPU is the safe default. Set device to cuda:0 if appropriate.
.venv-iil/Scripts/python.exe train_iil.py --config configs/iil_scaffold.json

# Resume only at a saved stage boundary; config and manifest must match exactly.
.venv-iil/Scripts/python.exe train_iil.py --config configs/iil_scaffold.json --resume log/iil_scaffold_seed42/stage_2.pt

# Trusted local checkpoints only (optimizer/RNG state requires pickle loading).
.venv-iil/Scripts/python.exe evaluate.py --iil-checkpoint log/iil_scaffold_seed42/stage_4.pt --splits iid_test ood_test

.venv-iil/Scripts/python.exe -m unittest discover -s tests -v
```

The default 40/15/15/15/15 split is shuffled, approximately label-stratified,
and grouped by **exact SMILES strings**, not by scaffold or assay. The manifest
stores SHA256 of the input file and covers each train record once. Equivalent
molecules with different SMILES are not canonicalized; assess this before
claiming structure-level deduplication. Conflicting labels are reported and
kept together, not silently removed. Validation/test samples never enter training.

Each increment uses only its own samples (multiple epochs, no replay). D0-only
class frequencies set fixed weights for incremental classification. Initial
MoleOOD losses retain their original unweighted form. Training uses fixed epoch
budgets; test metrics never select checkpoints. Hyperparameters should be chosen
on validation splits in separate runs, with final test reporting kept separate.

## Outputs and interpretation

- `initial_unprompted.pt`: trained D0 baseline, no prompt pool.
- `stage_0.pt`: the same backbone/head plus newly randomized prompts and keys.
- `stage_N.pt`: after increment N, including optimizer, RNG, config and manifest.
- `metrics.json`: metrics after each stage, plus both D0 modes, and prompt counts.

Evaluation reports ROC-AUC, average precision (`pr_auc_ap`, not trapezoidal
PR-AUC), balanced accuracy and accuracy. Undefined binary ranking metrics are
JSON null for single-class evaluation sets. OOD results are supplementary
scaffold generalization results. The source file's `iid_test` name alone does
not establish strict IID sampling or shared scaffolds. Learning curves on a
fixed test set are not by themselves a measure of forgetting.

Inactive prompt/key rows and their row-valued Adam moments are restored after
each update; Adam's scalar step counter is shared. Prediction-head parameters
update normally. Resume is supported at completed stage boundaries, not mid-epoch.

The existing `main.py` and legacy evaluation path are retained. Legacy
`evaluate.py` still expects its old DrugOOD/MMCV interfaces; use
`--iil-checkpoint` for the new PyG workflow. Existing feature clamping and GNN
edge processing are inherited from the local baseline, not a verified
reproduction of the original paper's featurization.

## Verification performed

Seven CPU tests passed: zero-prompt equivalence to the existing fusion,
node-to-prompt assignment, prompt/key gradients and frozen encoder state,
inactive-row/momentum preservation, state-dict roundtrip, deterministic grouped
splits, and synthetic end-to-end training with exact stage-boundary resumption.
The latter also exercises checkpoint evaluation. A four-molecule batch from
the real scaffold JSON passed forward/backward and an optimizer update with
variable substructure counts (3, 13, 8, 7). No full-data training or accuracy
benchmark has been performed. Synthetic test metrics are not research results.
