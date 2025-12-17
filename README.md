# TABF: Template-Based Automatic Fragmentation Program

Repo for the publication: [A Template‑Based Automatic Fragmentation Algorithm for Complex and Large Systems in the Generalized Energy‑Based Fragmentation Framework](https://chemrxiv.org/engage/chemrxiv/article-details/68ea0bab5dd091524fec1595)
## Set up environment

Key dependencies are:

- RDKit
- NetworkX
- scipy
- numpy

## To run
For common organic molecules, run
```
python getfrg.py file.xyz --charge charge
```
For ionic liquids, run
```
python getfrg_pair.py file.xyz --charge charge

```
