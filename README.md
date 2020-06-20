# ARGF_multimodal_fusion
Codes for: "Modality to Modality Translation: An Adversarial Representation Learning and Graph Fusion Network for Multimodal Fusion" AAAI-20

Pdf is available at: https://www.aaai.org/ojs/index.php/AAAI/article/view/5347


Some of the codes are borrowed from https://github.com/Justin1904/Low-rank-Multimodal-Fusion. We thank very much for their sharing.


The raw data are released in https://github.com/A2Zadeh/CMU-MultimodalSDK and  https://github.com/soujanyaporia/multimodal-sentiment-analysis. If you need to use these data, please cite their corresponding papers. For raw datasets, please download them from: https://github.com/soujanyaporia/multimodal-sentiment-analysis/tree/master/dataset (you need to place the downloaded data in the "dataset" folder). We have placed the processed data in pickle format in the main folder.

To run the code: 

For mosi dataset: python train_mosi_graph.py  
For mosei dataset: python train_mosei_graph.py     
For iemocap dataset: python train_iemocap_graph.py  

We test the code with python2, and the framework is Pytorch. You can change the defaulted target modality in the code.

If you need to use the codes, please cite our paper:

Mai, Sijie, Haifeng Hu, and Songlong Xing. "Modality to Modality Translation: An Adversarial Representation Learning and Graph Fusion Network for Multimodal Fusion." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 01. 2020.
