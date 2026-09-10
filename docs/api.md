# API reference

## JSON utilities

```{eval-rst}
.. automodule:: sigmf_zarr.json
   :members:
   :undoc-members:
```

## Store

```{eval-rst}
.. automodule:: sigmf_zarr.store
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
```

## Resolved signals

```{eval-rst}
.. autoclass:: sigmf_zarr.signals.SignalView
   :members:
```

## Index utilities

```{eval-rst}
.. automodule:: sigmf_zarr.indexes
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.splits
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.sources
   :members: source_groups
```

## RFML validation

```{eval-rst}
.. automodule:: sigmf_zarr.rfml
   :members:
   :undoc-members:
```

## Integrity

```{eval-rst}
.. automodule:: sigmf_zarr.integrity
   :members:
   :undoc-members:
```

```{eval-rst}
.. automodule:: sigmf_zarr.provenance
   :members: capture_inputs, verify_inputs, file_identity
```

## Validation

```{eval-rst}
.. automodule:: sigmf_zarr.validation
   :members:
   :undoc-members:
```

## Read-only views

```{eval-rst}
.. automodule:: sigmf_zarr.readonly
   :members:
   :undoc-members:
```

## SigMF import and export

```{eval-rst}
.. automodule:: sigmf_zarr.sigmf
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.export_plan
   :members:
   :undoc-members:
```

## RadioML

```{eval-rst}
.. automodule:: sigmf_zarr.radioml2016
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.radioml2018
   :members:
   :undoc-members:
```

## Panoradio HF dataset

```{eval-rst}
.. automodule:: sigmf_zarr.panoradio
   :members:
   :undoc-members:
```

## CSPB datasets and TIM files

```{eval-rst}
.. automodule:: sigmf_zarr.tim
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.cspb
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.cli.import_cspb
   :members:
   :undoc-members:
```

## CLI

```{eval-rst}
.. automodule:: sigmf_zarr.cli.command
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.cli.import_command
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.cli.import_panoradio
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.cli.group
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.cli.output
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.cli.context
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.cli.sigmf
   :members:
   :undoc-members:

.. automodule:: sigmf_zarr.cli.store
   :members:
   :undoc-members:

```

## Optional PyTorch adapter

`sigmf_zarr.pytorch.RecordingDataset` provides dense map-style loading, and
`RecordingItem` defines its typed item mapping. See [PyTorch datasets](pytorch.md)
for constructor options, transforms, worker handling, and validation.

## Experiment manifests

```{eval-rst}
.. automodule:: sigmf_zarr.experiments
   :members: dataset_manifest
```
