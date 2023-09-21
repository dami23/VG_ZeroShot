This project includes the PyTorch implementation codes for our work “A Context-Aware One-Stage Framework for Zero-Shot Visual Grounding”， which has been submitted to ICRA 2024.

The experiments are conducted on Ubuntu 20.04 :
python = 3.7
PyTorch = 1.13.1

### Data Preparation
1. run "sh data/download_ann.sh" to download the csv files for Flickr30K, Flickr30K-Split-0, and Flickr30K-Split-1.
2. run "python data/prepare_refer.py" to generate csv files for RefCOCO, RefCOCO+, and RefCOCOg.
3. run "python data.ocid_csv_generate.py" to obtain csv files for OCID-Ref.

### Pre-trained CLIP Models
cd third_party \
cd modified_CLIP \
pip install -e .

### Training and Evaluation
1. training: python codes/model_train.py 'dataset_experiment'
dataset: the name of benchmark datasets, such as Flickr30K, Flickr30K-Split-0, Flickr30K-Split-1, RefCOCO, RefCOCO+, RefCOCOg, OCID-Ref
experiment: a specific name to save a model for the current model, such as 'final' for the whole framework.

2.evaluation: python codes/model_train.py 'refcoco_final' --resume=True --only_test=True
'refcoco_final': saved model name

### Trained Models
The trained models on each dataset can be downloaded from: [here](https://drive.google.com/drive/folders/183BmPhVlt8NYfZdWq5LGYB5XAG6ohI0S?usp=share_link).

### Acknowledgement
We thank
1. the repository on retina-net (https://github.com/yhenon/pytorch-retinanet).
2. the repository on CLIP and pre-trained models (https://github.com/openai/CLIP).
