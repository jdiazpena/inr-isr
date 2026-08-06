# Configuration

This folder is reserved for explicit, reviewable experiment configuration.

`training_defaults.json` records the shared SIREN, optimizer, regularization,
diagnostic, and reconstruction-grid defaults. Trainers accept it explicitly with:

```bash
python src/synthetic_train_3d.py --config config/training_defaults.json [other arguments]
```

Explicit command-line arguments override values from the JSON file. Machine-specific
paths and one-off output locations should remain command-line arguments.

`velocity_integration_benchmark.json` is the active declaration of benchmark scopes,
including geometries, velocities, integration products, seeds, flow-reversal seeds,
and regularization modes. Editing it changes the scientific experiment matrix and
should therefore be reviewed together with the generated manifest.

Candidate configurations include:

- alternative synthetic morphology suites;
- dense reconstruction grid settings;
- real-radar window definitions.
