# ProEvo-NAS:
## Dependency Install

```text
pytorch==1.12.0
CUDA ==11.6
dgl ==1.1.0+cu116
ray==2.10.0
networkx==3.1
```

## Usage

```python
# Search process
cd /search_algorithm/NAS
python search_algorithm.py
```
Note: Running search_algorithm.py will automatically generate two directories required for the prediction step:

results/: Stores the detailed prediction outputs.

save_models/: Saves the identified architecture weights and pre-trained models.
```
#Test on DPFunc and Deepfri
python pred.py
```
## Datasets

The experimental data used in this repository are organized as follows:

* **`deepfri_data/`**: Contains the protein IDs and related dataset splits utilized in the [DeepFRI](https://www.nature.com/articles/s41467-021-23303-9) paper.
* **`DPFunc/`**: Contains configurations and data related to the DPFunc dataset. For a comprehensive description of the dataset characteristics and original data acquisition, please refer to the original [DPFunc](https://www.nature.com/articles/s41467-024-54816-8)) publication.
