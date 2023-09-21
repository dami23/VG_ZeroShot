This projects includes the PyTorch implentation codes for our work “A Context-Aware One-Stage Framework for Zero-Shot Visual Grounding”， which has been submitted to ICRA 2024.

The experiments are conducted on Ubuntu 20.04 :
python = 3.7
pytorch = 1.13.1

### Data Preparation
1. run sh data/download_ann.sh to download the csv files for Flickr30K, Flickr30K-Split-0, and Flickr30K-Split-1.
2. run python data/prepare_refer.py to generate csv files for RefCOCO, RefCOCO+, and RefCOCOg.
3. csv files for OCID-Ref can be found in data/ocid/ directory.

