# ARGF_multimodal_fusion
Codes for: "Modality to Modality Translation: An Adversarial Representation Learning and Graph Fusion Network for Multimodal Fusion" AAAI-20

Pdf is available at: https://www.aaai.org/ojs/index.php/AAAI/article/view/5347


Some of the codes are borrowed from https://github.com/Justin1904/Low-rank-Multimodal-Fusion. We thank very much for their sharing.


The raw data are released in https://github.com/A2Zadeh/CMU-MultimodalSDK and  https://github.com/soujanyaporia/multimodal-sentiment-analysis. If you need to use these data, please cite their corresponding papers. For raw datasets, please download them from: https://github.com/soujanyaporia/multimodal-sentiment-analysis/tree/master/dataset (you need to place the downloaded data in the "dataset" folder). We have placed the processed data in pickle format in the main folder (PLEASE UNRAR THE ).

To run the code: 

For mosi dataset: python train_mosi_graph.py  
For mosei dataset: python train_mosei_graph.py     
For iemocap dataset: python train_iemocap_graph.py  

We test the code with python2, and the framework is Pytorch. You can change the defaulted target modality in the code. The code has not been cleaned yet, and we will continue to update it.

Re-evaluating the codes:

Since we lose the best hyperparameters, we are now re-evaluating the model with target modality set to language. We found that the best hyperparameter setting for iemocap dataset is:

ahid = vhid = thid = 100, adr = vdr = tdr = 0.1, lr = 0.001, batch_size = 32, decay = 0.01, alpha = 0.001

Now the best acc and f1 for iemocap is 61.18 and 60.92, respectively (see iemocap_setting.png). You might need to run the codes with different random seeds to obtain the best results.

The best hyperparameter setting for MOSEI is:

ahid = vhid = thid = 150, adr = vdr = tdr = 0.2, lr = 0.01, batch_size = 16, decay = 0, alpha = 0.1

Now the best acc and f1 for MOSEI is 60.97 and 58.92, respectively (see MOSEI_parameter.png). Remember to set the random seeds as:

np.random_seed(20210820)

torch.manual_seed(20210820)

If you need to use the codes, please cite our paper:

Mai, Sijie, Haifeng Hu, and Songlong Xing. "Modality to Modality Translation: An Adversarial Representation Learning and Graph Fusion Network for Multimodal Fusion." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 01. 2020.
