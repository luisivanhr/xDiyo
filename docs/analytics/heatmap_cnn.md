# Optional heatmap CNN

The native CNN supports two uses: a spatial edge encoder trained jointly with
the temporal GAT, and a standalone classification or regression model over
historical grid features. Both use the existing training, weighting, validation,
recipe and saved-model interfaces. These implementation checks do not establish
a predictive advantage.

Install the optional neural runtime with `pip install -e ".[graphs,training]"`.
Importing `xdiyo_analytics.neural` does not import PyTorch. Device and thread
selection use the runner's `ExecutionPolicy`, as for the GAT.

## Two independent options

- `TemporalGraphConfig(heatmaps=None)` is the unchanged scalar GAT default.
  It does not load or require heatmap tables.
- `HeatmapCNNConfig(reconstruction=None)` is the default in both CNN uses.
  It creates no decoder and adds no auxiliary loss.
- `Reconstruction(weight=0.1, hidden_width=64, loss="mse")` enables an auxiliary branch.
  Its weight must be positive; disable it with `None`.

```text
observed maps + availability planes
                 |
          shared CNN encoder
                 |
           spatial vector z
            /           \
prediction branch       optional MLP decoder
GAT or standalone       reconstructed input maps
fixture head                     |
       |                  masked reconstruction loss
prediction loss
```

The decoder takes the same flattened, projected representation used for
prediction. It is a fully connected decoder, not another convolutional encoder:
`Linear(embedding_dim, hidden_width) -> activation -> Linear(hidden_width, map_cells)`.
Reshaping its output reconstructs the map channels. Availability planes are not
reconstruction targets. Both losses propagate into the shared encoder. The
decoder is not invoked for prediction or embedding extraction.

The data objective is `prediction_loss + reconstruction.weight * reconstruction_loss`.
The GAT retains its existing prior penalty and AdamW weight decay; the standalone
model uses AdamW weight decay. The selected loss is computed for each **observed**
map, then averaged over the observed maps for that item.
Wholly absent maps contribute neither an error nor a denominator. An all-missing
population has zero reconstruction loss, not NaN.

### Reconstruction loss choices

- **`loss="mse"` (default):** mean squared error across cells. The decoder's
  output remains unrestricted, and the original MSE calculation is unchanged.
- **`loss="kl"`:** forward KL divergence `D_KL(target || reconstruction)`.
  Decoder outputs are logits; `log_softmax` normalizes over the spatial cells of
  each map separately. The resulting reconstructed distribution is nonnegative
  and sums to one (subject to floating-point roundoff).

For KL, `p` is the observed grid divided by its total mass and `q` is the decoder
distribution. The per-map loss is `sum_cells p * (log(p) - log(q))`, in nats.
It **sums** over cells, then averages observed maps; there is no extra division
by the grid's number of cells. Zero target cells contribute zero without adding
pseudocounts. Stable log-softmax avoids taking the logarithm of an underflowed
predicted probability.

Default mass-normalized inputs already define probability masses. Other
nonnegative count/density grids are normalized only for the KL target; inputs
to the encoder retain their configured values. Negative or zero-total observed
maps cannot define a KL target and raise an error. Native empty/zero-mass sources
remain unavailable and are excluded, rather than becoming uniform targets.

Use `Reconstruction(loss="kl", weight=0.1)` in either standalone or GAT mode.
MSE and KL have different numerical scales, so the same reconstruction weight
does not imply the same auxiliary influence. The weight remains configurable.

For GAT training, this loss is averaged across historical directed edges with
at least one observed map at each issuance, then weighted by that issuance's
share of fitting-fixture observation weight. For standalone training, it is
averaged using row weights over rows with an observed map. Repeated historical
edges at different issuances therefore participate at each eligible issuance.

`train_loss` and `validation_loss` remain prediction losses. Optional extra
metrics are `train_reconstruction_loss`, `validation_reconstruction_loss`,
`train_objective` and `validation_objective`. The GAT training objective includes
its explicit prior penalty; validation excludes that penalty. As before, training
loss is evaluated before the optimizer step and validation after it. The default
early-stopping monitor remains validation prediction loss (training prediction
loss when there is no validation population). An explicit `TrainingControl.monitor`
can select an auxiliary or combined metric.

## GAT integration

```python
from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalGraphModel, GraphHeatmaps
from xdiyo_analytics.neural import HeatmapCNNConfig, Reconstruction
from xdiyo_analytics.training import TrainingControl

spec = TemporalGraphModel(
    config=TemporalGraphConfig(
        timezone="Europe/Berlin",
        heatmaps=GraphHeatmaps(
            encoder=HeatmapCNNConfig(reconstruction=Reconstruction(weight=0.1)),
        ),
    ),
    control=TrainingControl(max_steps=100, seed=17),
)
store = spec.event_store(history, team_seasons, heatmaps=heatmap_points)
model = spec.build(store)
model.fit(fit_context)
predictions = model.predict(prediction_context)
```

`GraphHeatmaps()` uses native `Heatmap(grid_size=10, normalization="mass",
side="both", kinds=("player",), orientation="team")`. Each historical match
supplies individual maps, not pooled rolling point clouds. The receiving team's
own map is in its canonical frame; the opponent's map is rotated 180 degrees
using the existing `features.spatial` implementation. The reverse edge uses the
opposite receiver perspective. Pitch coordinates retain native normalized
coordinate units. Goalkeeper points are excluded by default.

The default shared encoder is:

```text
4 channels: own, opponent, own-available, opponent-available
Conv2d(4,8,3,padding=1) -> ReLU
Conv2d(8,8,3,padding=1) -> ReLU
AvgPool2d(2) -> flatten 200 -> Linear(200,16)
```

This is 4,096 trainable encoder parameters including biases, before the optional
decoder. The 16 coordinates and two explicit availability flags are concatenated
with scalar edge features. The enriched vector feeds both attention logits and
message values in every GAT layer. One encoder is shared across teams, leagues,
historical matches, directions and layers. There are no frozen/pretrained CNN
outputs. Gradients from the fixture prediction loss train the encoder even when
reconstruction is disabled.

Only deterministic native grids are prepared once into the immutable event store.
Its fingerprint includes these grid values and their configuration. Learned
encodings are recomputed during optimization, once per issuance forward pass and
reused across its GAT layers. Existing By Round and One Day Ahead cutoff and
result-availability rules select historical edges; forecast-target/future maps
never enter that snapshot. No xG is required. Changing grid construction settings
requires new recipe preparation; encoder settings can vary over the same grids.

## Missingness and validation

Each entire unavailable map becomes zero-filled only at the neural boundary,
with availability zero. Its observed counterpart is retained and keeps availability
one, including its zero-valued cells. Native absent, empty-selected-kind and
zero-mass maps remain unavailable. If every input map is absent, the encoder
representation is explicitly zero. Convolution padding is separate from missing
map placeholders.

Partial NaNs, infinities, float32 overflow, malformed coordinates, duplicate map
identities/point ordering, or conflicting source versions raise errors. They are
not converted into missing observations. Grid normalization and validation use
the existing native spatial implementation.

## Standalone classification and regression

Prepare safe historical grids using `Lag`, `RollingMean` or `EMA`, then select
`HeatmapCNNModel`. The adapter accepts native complete grid groups in `dataset.X`.
It never reads current-fixture point clouds. Feature construction retains native
cutoff, missing-window-slot, orientation and predictor-safety rules.

```python
from xdiyo_analytics.features import Heatmap, RollingMean
from xdiyo_analytics.neural import HeatmapCNNModel, HeatmapCNNConfig
from xdiyo_analytics.training import TrainingControl, TrainingRunner, ValidationTail

features = {
    "recent_map": RollingMean(
        Heatmap(grid_size=10, kinds=("player",), orientation="team"),
        window=20, min_periods=1,
    ),
}
# Evaluate and assemble features through the ordinary native pipeline.
spec = HeatmapCNNModel(
    task="regression",  # or classification; classes are inferred from fitting y
    encoder=HeatmapCNNConfig(),
    control=TrainingControl(max_steps=100, seed=17),
)
runner = TrainingRunner(spec.build, validation=ValidationTail(0.2))
# result = runner.run(dataset, split_plan)
```

Empty `grid_features` discovers all native grid groups. To choose a subset,
provide their full prefixes without `_yXX_xXX`, for example
`home::recent_map::heatmap_for_mass_10` (inspect the actual assembled column names).
All selected grids must have the same resolution and every cell. Do not use a
column selector that removes individual cells. Optional `numeric_features`
select additional prepared scalar predictors; their scaling is fitted on fitting
rows only and missing values get explicit masks. Do not wrap these adapters in
generic imputers or scalers that would destroy map missingness.

Standalone inputs concatenate their selected maps and masks as CNN channels. The
prediction head uses `[spatial vector, map flags, scaled numeric values, numeric
missing flags]`. Classification supports one target and native `(target, class)`
probability columns; regression supports one or more numeric targets. Native
observation/class weights and reusable probability calibration work downstream.
`model.encode(prediction_context)` extracts the frozen per-fixture spatial vector.
These are fixture representations, distinct from GAT team embeddings.

## Configurable and inferred settings

UI and recipes expose convolution channel widths, odd kernel size, pooling size,
embedding size, activation, reconstruction loss/weight/hidden width, batch size,
standalone head width, learning rate, weight decay, and native training controls.
Grid resolution/method/normalization/kinds are native `Heatmap` parameters.
GAT grid side/frame remain `both`/`team` to preserve receiver semantics.

Input channel count, flattened width, output class count/order and regression
target count are inferred from the prepared inputs and fitting labels. Flattening
preserves coarse spatial position. A nondefault grid resolution changes the
inferred projection and decoder sizes.

## UI, persistence and diagnostics

In **Model & training**, choose **Temporal team GAT**, open its configuration
and enable **Heatmaps**. Under **Encoder**, enable **Reconstruction** only when
desired, then choose **Loss → mse / kl**. Use **Use native graph labels and preprocessing** for native outcome
labels and masked graph scaling. The builder loads `heatmap_points` automatically
when heatmaps are enabled.

For standalone use, add a historical Heatmap feature in **Features & ratings**,
choose **Heatmap CNN**, select **Task**, and click **Use native CNN preprocessing**.
Set the corresponding labels explicitly. Both paths retain the existing training
control/validation and UI export workflow. JSON and Python exports round-trip the
nested optional settings.

`save_model` automatically selects the native serializer. GAT artifacts include
CNN configuration/weights and deterministic grids in the embedded event store;
standalone artifacts include input group order, fitting-only preprocessing, class
order, CNN/decoder/head weights and training history. Loading restores prediction
and embedding values. These are prediction artifacts, not optimizer-resume files.
Each fold/restart gets freshly initialized weights. Scalar-only GAT artifact
fingerprints remain compatible with earlier configurations.

GAT team embeddings and attention still use `EmbeddingReporter` and
`GraphAttentionWeightReporter`. Optional reconstruction/objective curves are
selectable in `GATLossReporter`; standalone losses use `LearningCurveReporter`.
No reconstruction visualization or team interpretation is implied by a standalone
fixture embedding.
