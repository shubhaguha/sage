Notes
===

Setup
---

```shell
# Create env
python3 -m venv sage-env
source sage-env/bin/activate
pip install -U pip

# Install sage-importance lib
pip install .

# Run example notebooks
sage-env/bin/jupyter notebook  # to make sure you run Jupyter in the right environment

# Run credit example with CatBoost model
python3 test_unfairness_loss_with_permutation_estimator.py
```
