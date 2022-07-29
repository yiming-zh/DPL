# Towards-Unifying-the-Label-Space-for-Aspect--and-Sentence-based-Sentiment-Analysis
---
[Towards Unifying the Label Space for Aspect- and Sentence-based Sentiment Analysis](https://aclanthology.org/2022.findings-acl.3/)

<!-- <p align="center"><img width="85%" src="imgs/shaping_main.png" /></p> -->



### Set Up
Use the following commands to clone and install this package. 

```
# environment
python == 3.6.12
torch == 1.2.0
transformers == 2.9.1

# scripts

pip install -e requirements.txt

```


### Example use

Download prepared shaped data for the FewRel task here and place the data in the ```data/``` directory: https://drive.google.com/drive/folders/1tEuhAukhvwhW_7_tO-kG1plql2rSyp84?usp=sharing  

Run:
```
bash run_fewrel.sh
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

