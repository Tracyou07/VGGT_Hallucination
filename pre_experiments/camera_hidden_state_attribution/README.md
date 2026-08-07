# Hidden-State Attribution Utilities

These modules are retained as dependencies for replaying Camera Head tokens and
reading frozen translation-preferred unit features. The original attribution runner,
publisher, results, and experiment documentation are intentionally excluded from the
training branch.

Do not refit unit rankings on validation or test scenes. Every training run that uses
selected units must record the source manifest, run ID, Camera Head iteration, unit
indices, and manifest digest.
