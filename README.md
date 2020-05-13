# Exploring Structure and Texture in Clothing Embeddings

The structure of a garment–the silhouette, the decorative details, how the seams fit together–is integral to the style and the construction of a piece of clothing. Work in deep learning for fashion focuses disproportionately on textural attributes of clothing, to the detriment of other design details. The goal of this project is to use a neural network to learn a texture-invariant, structure-sensitive embedding for images of clothing. To do so, I gather a new dataset consisting of both photographs and detailed design sketches and train a network to map both types of data to a shared embedding space. The resulting embedding is less sensitive to color and more sensitive to design details than one trained on photographs alone.

[View Project Write-Up](https://github.com/a-newman/clothing-embedding/blob/master/Write-up.pdf)

## Data Files

Large data files (model checkpoints, precomputed nearest-neighbor results, etc.) can be downloaded [here](https://drive.google.com/open?id=1RTqBixnRtM7G52X-zBr-YiPW_UU7WMxC). Unzip the file \<folder_name.zip\> into the folder \<folder_name\>. 
- `ckpt.zip`: model weights 
- `preds.zip`: intermediate data files, including embeddings, nearest neighbor predictions, visualization files, etc
- `scripts.zip`: indexing files required for loading deepfashion data

## Code 

### Files and folders

- `model.py`: implementation of Siamese and dual-branch models
- `data_loader.py`: contrastive data loaders for DeepFashion2 and Simplicity
- `predict.py`: code for running analysis: generating embedding vectors, calculating nearest neighbors, running pca
- `train.py`: code for training model
- `config.py`: configuration variables 
- `visualize.py`: visualization functions
- `view_*.ipynb`: visualize analysis results
- `scripts/data_prep.py`: organize the DeepFashion-Duplicates data files

### Commands

#### Training a Model 

`python train.py --run_id <run_id> --dataset_name <dataset_name> --dset_mode <dset_mode> --model_type <model_type> `

Arguments:
- `run_id`: string, name to use to identify this training run 
- `dataset_name`: string, "deepfashion" or "simplicity"
- `dset_mode`: string. For DeepFashion2, can be "color_mask_crop", "color_crop", or "grayscale_mask". For Simplicity, can be "natural_image" or "line_drawing" (for simplicity, mode does not matter during training)
- `model_type`: string, "siamese" or "dual"

To train on DeepFashion2: 
`python train.py --run_id deepfashion_embedding --dataset_name deepfashion --dset_mode grayscale_mask --model_type siamese`


#### Shortcut to run all analysis steps: 
`python predict.py all <kwargs>`

For DeepFashion2: 
`python predict.py all --run_id deepfashion_embedding --dataset_name deepfashion --dset_mode grayscale_mask --dset_split validation`

#### Generating the embedding vectors

`python predict.py gen_embedding --run_id <run_id> --analysis_tag <analysis_tag> --dataset_name <dataset_name> --dset_mode <dset_mode> --dset_split <dset_split> --principal_encoder <principal_encoder`
- `run_id`: string to identify a prior training run to use the ckpt from 
- `analysis_tag`: string (optional) id to identify this round of analysis
- `dataset_name`: string, "deepfashion" or "simplicity"
- `dset_mode`: string. For DeepFashion2, can be "color_mask_crop", "color_crop", or "grayscale_mask". For Simplicity, can be "natural_image" or "line_drawing" 
- `dset_split`: string, "validation" or "train"
- `principal_encoder`: int (optional), 1 or 2. For a dual model, choose which encoder to pass the images through. For Simplicity dataset, 1 processes natural images, 2 line drawings

To generate an embedding for DeepFashion2: 
`python predict.py gen_embedding --run_id deepfashion_embedding --dataset_name deepfashion --dset_mode grayscale_mask --dset_split validation`

#### Calculate nearest neighbors for an embedding

`python predict.py nearest_neighbors --run_id <run_id> analysis_tag <analysis_tag> --dset_split <dset_split> --subset_n <subset_n>`

- `run_id`: string to identify a prior training run to use the ckpt from 
- `analysis_tag`: string (optional) id to identify this round of analysis; should match an embedding that was previously computed
- `dset_split`: string, "validation" or "train"
- `subset_n`: int optional, defaults to 100), number of items from the embedding to calculate nearest neighbors for

For DeepFashion2: 
`python predict.py nearest_neighbors --run_id deepfashion_embedding --dset_split validation`

#### Run PCA on an embedding

`python predict.py pca --run_id <run_id> analysis_tag <analysis_tag> --dset_split <dset_split> --subset_n <subset_n>`
- `run_id`: string to identify a prior training run to use the ckpt from 
- `analysis_tag`: string (optional) id to identify this round of analysis; should match an embedding that was previously computed
- `dset_split`: string, "validation" or "train"
- `subset_n`: int, number of items from the embedding to calculate nearest neighbors for

For DeepFashion2: 
`python predict.py nearest_neighbors --run_id deepfashion_embedding --dset_split validation`

#### Visualize results

See visualization notebooks

