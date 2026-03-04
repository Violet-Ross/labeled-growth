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
  
## For 2/25

- Violet will log in to Ada, Phil to help if not possible.
- Violet to reproduce experiments and move towards producing equivalent versions of figures on new data set. 
- Phil to buy Boulder plane tix and schedule some work time with Violet.
- Frannie to work on optimization for community detection: no Float64 booleans!
- Frannie to give his awesome talk and tell us how awesome it was. 

## For 3/4

- Phil to request VPN and Ada access for Violet.
- Violet to run experiments.
- Violet to implement an SEM convergence criterion based on the parameter vector.
  - Already implemented. 
- Violet to try constructing subsets of the coauthorship data.
   - For Frannie's purposes: 30K hyperedges and 300 nodes (Senate Bills) is the upper bound. More nodes probably ok if fewer edges. 
- Generate likelihood heatmaps with revised likelihood calculation
- Phil to bring in his own scripts and structure the relative imports in the same way that Violet did.
- Frannie to try two more ideas for making gradient descent more tractable.

## For 3/11

- Violet to try a percentage-based convergence criterion rather than an absolute error.
- Violet to rerun viz related to likelihood, including the heatmap (now that likelihood is updated).
- Violet to continue corresponding with helpdesk on Ada access.
- Frannie to try some more with simulated annealing on edge-size limited senate bills.
- Frannie to try on new data set.
- Frannie to try SEM inferred parameters.
- Frannie to try importing local senate bills in the same way that Violet did on `senate_plots.py`.
  - Important to read nodetype as `int`. 

### Issues

- Label initialization on senate bills data set.
  

## For Later
