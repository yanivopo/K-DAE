K-AUTOENCODERS DEEP CLUSTERING
---------------------------------------------------------------
This repo contains the source code for our paper:

[**K-AUTOENCODERS DEEP CLUSTERING**](http://www.eng.biu.ac.il/goldbej/files/2020/02/ICASSP_2020_Yaniv.pdf) 
<br>
Yaniv Opochinsky, Shlomo E. Chazan, Sharon Gannot and Jacob Goldberger





---------------------------------------------------------------  
#### How to Use?
* clone the repo
* cd to the cloned dir 
* conda create -n k_dae python=3.6 
* conda activate k_dae 
* pip install -r path\to\requirements.txt 


### Run example: 
* `python main.py -dn mnist` 

#### Optional Args: 

* `----dataset_name` - The name of dataset [mnist / fashion / usps] default: mnist
* `--save_dir` - path to output folder. (contains logs and model.)
