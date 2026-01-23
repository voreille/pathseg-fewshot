# Few-Shot Semantic Segmentation Benchmark for Computational Pathology vFMs

This repository provides a benchmark for few-shot semantic segmentation in histopathology, with a focus on evaluating vision foundation models (vFMs) under episodic training and evaluation.

The benchmark is designed to:

* Support multiple heterogeneous segmentation datasets
* Standardize preprocessing across scanners and magnifications
* Enable fast episodic sampling without repeated mask I/O
* Make it easy to add new datasets with minimal boilerplate

---

## Project structure

### 1. Datasets and preprocessing

For each dataset included in the benchmark, a dedicated preprocessing CLI is provided under:

pathseg/preprocessing/<dataset_name>/prepare.py

Each CLI takes as input the raw data downloaded from the original source and outputs a standardized directory structure compatible with the benchmark.

Currently, preprocessing scripts accept:

* --target-magnification (e.g. 10x or 20x)

A global download-and-prepare script is intentionally not provided yet, as datasets are handled individually during early development.

---

### 2. Magnification and MPP handling

Different scanners may report different microns-per-pixel (MPP) values for the same nominal magnification.

We assume the following conventions:

* 20x ≈ 0.5 MPP
* 10x ≈ 1.0 MPP

During preprocessing, images are resampled to the closest achievable target MPP using integer downsampling factors when possible.

Example:
An image reported as 20x with MPP = 0.24 will be downsampled by a factor of 2 to reach approximately MPP = 0.48.

Original and target MPP values are stored in the dataset metadata.

---

### 3. Preprocessed data layout

The preprocessed data directory can be located anywhere on disk. In the examples below, we assume it is stored under repo_root/data/.
```
preprocessed_data_rootdir/
  datasets.csv
  datasets_index.parquet <- contains (dataset_id[str], valid_labels[int], maybe label_map[dict]) 
  class_index.parquet <- concatenation of each class_index (dataset_id, class_id, list of candidates for that class with sufficient min area containing the path of the image and mask, ...) 
  dataset_name/
    label_map.json
    metadata.csv
    class_index.parquet
    images/
      sample_001.png
      sample_002.png
    masks_semantic/
      sample_001.png
      sample_002.png

```
---

### 4. Annotations format
Annotations are expected to be stored as grayscale images uint8 as png or jpg.
A label_map.json (str to int) is provided to map the int values to a string values.

The label "Background" is reserverd and always mapped to 0.
The label "Ignore" is reserved and always mapped to 255 (usually unnatoted pixels as it is the case in IGNITE are mapped to this)

### 5. Dataset metadata

metadata.csv (one row per image)

This file is the primary entry point for dataset loading and episode construction. It is designed to be human-readable and easy to inspect.

Required fields:

* dataset_id: identifier of the dataset (typically the folder name)
* sample_id: unique identifier within the dataset
* image_relpath: relative path to the image file
* mask_relpath: relative path to the semantic mask
* width: image width in pixels
* height: image height in pixels
* mpp_x, mpp_y: microns per pixel
* magnification: nominal magnification (e.g. 10, 20)
* split: dataset split (e.g. train / val / test)

Additional dataset-specific columns are allowed.

---

### 6. Per-class annotations (derived)

To enable fast episode construction without repeatedly reading segmentation masks, per-image-per-class statistics are stored in a separate file.

annotations.parquet (one row per image x class)

Typical fields include:

* dataset_id
* sample_id
* dataset_class_id
* present
* area_ratio
* bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax

This file is generated during preprocessing and treated as a cache.

---

## Episodic training and evaluation

### Episode definition

An episode is always constructed from a single dataset.

The default episode construction for a C-way K-shot setting (adapted for semantic segmentation) follows these steps:

1. Dataset sampling
   Randomly sample one dataset from the available datasets.

2. Class sampling
   Sample C class (C <= number of classes of that dataset, for now on the dataset I want to use the minimal number of classes is 5)  
   Derive the set of semantic classes present in the query images.
   I guess like this it resampled naturally the underepresented classes

3. Query sampling
   For each class draw 1 or more representative patches of that class using the index class_index.parquet
   and random crop patch from these patches 

4. Support sampling
   Same as 3. but K representative to have 

5. Training / evaluation
   The episode is processed using the dataset’s native label space.
   Segmentation is treated as a standard multiclass problem within the episode.

This design avoids cross-dataset label ambiguity and allows each dataset to define its own semantic segmentation task.

---

## Getting started

### 1. Download datasets

Downloading datasets is optional and depends on which benchmarks you intend to run.

### 2. Environment setup

conda create -n pathseg-benchmark python=3.10 -y
conda activate pathseg-benchmark

python -m pip install --upgrade pip

### 3. Install Pytorch

pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu123

Replace cu123 with your CUDA version (e.g. cu121, cu118).

### 3. Install this repo

pip install -e ".[parquet]"

or in dev mode

pip install -e ".[dev,parquet]"

---

## Training and evaluation

TODO
Episodic training and evaluation scripts will be added.

---

## Reproducing results

TODO
This section will document how to reproduce the experiments reported in the paper.

## Notes to keep in mind
### About the background class

I will start with setting every class not sampled as background causes it looks more like the way I want to use the model in the end. 
I think I also need to make the number of ways vary so the model is not lost at inference.

Maybe consider training for only foreground vs background and then do postprocessing to handle multiple class.

### building tile candidates for selection in episode sampling 
So the idea is to tile with a fixed grid each image by the input size of the model
with the CLI pathseg_fewshot/preprocessing/build_indexes.py, and compute for each
tile the ration of the class present and select tiles based on the area of the classes.

to mitigate the issue arising from fixed grid, namely loose of data augmentation 
by random crop and handling of border tiles. for the first we add jittering during training and for the second we define the border tile as ending on the end of the image so it lead to some overlap but we assume it is ok for now.


there are mainly two types of episode sampler stateless and stateful, used for training and validation respectively.

Currently for validation I chose to sample 2 episodes per datasets with the same
spec as for training, except that between episode sample_ids are non-overlapping and defined the sample_ids selecteds in these episodes as validation. 
and I chose the ANORAK dataset as a test set, so not used for training.

Border tiles are defined as ending exactly on the end of the image so it leads to overlap between the border tiles and the one before last tile, so maybe watch out cause for small images the overlapping pixels may be overrepresented.
To avoid this for the validation I filter out border tiles before sampling episodes.

sample_ids are also non-overlapping also between support and query set for any episodes