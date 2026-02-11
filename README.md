# labeled-growth

## Top-Level Directory Structure

- `package_name` (Violet and Frannie pick the name)
  - `__init__.py`
  - `algorithms`
    - `__init__.py`
    - `sem.py`
    - `gd.y`
    - `sem.py`
    - ...
- `data` (*untouched* data files. Anything that has been cleaned is now `throughput`. 
- `experiments` (scripts and notebooks that produce figures, tables, or throughputs).
  - `experiments` should be able to do things like this: `from package_name.algorithms.sem import StochasticExpectationMaximization`. 
- `throughput` (anything which is the *output* of a computational process and which will be an *input* to another computational process (e.g. cleaned data, simulations that will later be visualized, etc). 
- `fig` (includes both figures and tables)
- `paper` (primarily `.tex` files, macros, bibliography)

## For 2/18

- Finish populating the `package_name` dir and pick a name. 
- Reconcile Violet + Frannie's version of `model.py` so that it works with other scripts.
- Verify that everything runs. 
  
