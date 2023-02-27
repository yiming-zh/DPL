# Towards-Unifying-the-Label-Space-for-Aspect--and-Sentence-based-Sentiment-Analysis
---
Code repository for paper [Towards Unifying the Label Space for Aspect- and Sentence-based Sentiment Analysis](https://aclanthology.org/2022.findings-acl.3/).



<!-- <p align="center"><img width="85%" src="imgs/shaping_main.png" /></p> -->



### Set Up
Use the following commands to clone and install this package. 

```
# environment
Ubuntu 20.04; cpu x86_64; 
GPU 2080ti; cuda V10.1;

# python environment
python == 3.6.12
torch == 1.2.0
transformers == 2.9.1
```
#### scripts
```
conda create -n DPL python=3.6
conda activate DPL
pip install torch==1.2.0
pip install -r requirements.txt
```


### Example

Download prepared shaped SemEval 2014 task4 dataset and our auxiliary dataset in the ```dataset/``` directory: https://drive.google.com/file/d/1jYlkuBLxdQfk746o07E-ryagQ_qJ6zAU/view?usp=sharing  
```
dataset
└── Biaffine
    └── glove
        ├── Laptops
        └── Restaurants
```

Download prepared pre-trained model in the ```pre_model/``` directory: https://drive.google.com/file/d/1UdXFD88YaE9aGr1zaR7uN4X8wphmcabY/view?usp=sharing
These models are published by [Rietzler](https://github.com/deepopinion/domain-adapted-atsc). You can also download them from Rietzler's url: [Laptops](https://drive.google.com/file/d/1I2hOyi120Fwn2cApfVwjaOw782IGjWS8/view) & [Restaurants](https://drive.google.com/file/d/1DmVrhKQx74p1U5c7oq6qCTVxGIpgvp1c/view).
```
pre_model
├── laptop
│   ├── added_tokens.json
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   └── vocab.txt
└── rest
    ├── added_tokens.json
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    └── vocab.txt
```

The final file list
```
.
├── README.md
├── __init__.py
├── dataset
├── experiment
├── img
├── layer
├── loader.py
├── main.py
├── model.py
├── pre_model
├── requirements.txt
├── train.py
├── utils
├── visual
└── vocabulary
```

Run:
```
cd experiment
bash run_resturant.sh $gpu
```


### Citation
```
@inproceedings{zhang-etal-2022-towards,
    title = "Towards Unifying the Label Space for Aspect- and Sentence-based Sentiment Analysis",
    author = "Zhang, Yiming  and
      Zhang, Min  and
      Wu, Sai  and
      Zhao, Junbo",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.3",
    doi = "10.18653/v1/2022.findings-acl.3",
    pages = "20--30",
}
```

