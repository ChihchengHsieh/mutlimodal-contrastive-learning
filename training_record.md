Test:  [ 0/28]  eta: 0:04:06  model_time: 0.0877 (0.0877)  evaluator_time: 0.0000 (0.0000)  loss: 2.8706 (2.8706)  cl-loss: 2.8706 (2.8706)  time: 8.8170  data: 8.6831  max mem: 1068
Test:  [27/28]  eta: 0:00:08  model_time: 0.0877 (0.0858)  evaluator_time: 0.0000 (0.0000)  loss: 3.0799 (2.9944)  cl-loss: 3.0799 (2.9944)  time: 8.2141  data: 8.1033  max mem: 1068
Test: Total time: 0:03:53 (8.3495 s / it)
Averaged stats: model_time: 0.0877 (0.0858)  evaluator_time: 0.0000 (0.0000)  loss: 3.0799 (2.9944)  cl-loss: 3.0799 (2.9944)
========================================For Training [MCL]========================================
MultimodalContrastiveLearningArgs(name='MCL', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, cl_pj_dim=128, clinical_cat_emb_dim=16, early_stopping_patience=None, warmup_epoch=0)
PhysioNetClinicalDatasetArgs(image_size=128, clinical_num=['age', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'acuity'], clinical_cat=['gender'], categorical_col_maps={'gender': 2}, normalise_clinical_num=True, use_aug=True)
MCLModelArgs(name='resnet50', clinical_emb_dims=16, clinical_out_channels=1000, cl_m1_pool=None, cl_m2_pool=None, cl_lambda_0=0.5, cl_temperature=0.1, cl_pj_pooled_dim=1000, cl_pj_embedding_dim=1000, cl_pj_dim=128)
==================================================================================================

Best model has been saved to: [MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25]
The final model has been saved to: [MCL_resnet50_accuracy_0_1152_epoch100_10-10-2023 20-07-44]

==================================================================================================
{'accuracy': 0.14260970056056976}



# MCL_resnet50_accuracy_0_1413_epoch21_10-03-2023 23-04-59
# Best model has been saved to: [MCL_resnet50_accuracy_0_1369_epoch27_10-07-2023 17-52-30]
# The final model has been saved to: [MCL_resnet50_accuracy_0_1369_epoch27_10-07-2023 17-52-30]


========================================For Training [chexpert]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='CL_Fix5_best', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=0)
=======================================================================================================

Best model has been saved to: [chexpert_CL_Fix5_best_f1_0_3720_precision_0_6507_accuracy_0_8776_recall_0_2604_auc_0_6189_epoch52_10-11-2023 22-50-18]
The final model has been saved to: [chexpert_CL_Fix5_best_f1_0_3704_precision_0_6628_accuracy_0_8783_recall_0_2570_auc_0_6179_epoch62_10-11-2023 23-07-46]

=======================================================================================================
{'f1': 0.38585209003215437, 'precision': 0.6469002695417789, 'accuracy': 0.8800627943485086, 'recall': 0.27491408934707906, 'auc': 0.6255414543515456}
Test:  [0/4]  eta: 0:00:28  model_time: 0.0099 (0.0099)  evaluator_time: 0.0000 (0.0000)  loss: 0.3061 (0.3061)  classification_loss: 0.3061 (0.3061)  time: 7.2466  data: 7.2277  max mem: 368
Test:  [3/4]  eta: 0:00:06  model_time: 0.0274 (0.0734)  evaluator_time: 0.0000 (0.0000)  loss: 0.2685 (0.2899)  classification_loss: 0.2685 (0.2899)  time: 6.5699  data: 6.4817  max mem: 368
Test: Total time: 0:00:26 (6.5699 s / it)
Averaged stats: model_time: 0.0274 (0.0734)  evaluator_time: 0.0000 (0.0000)  loss: 0.2685 (0.2899)  classification_loss: 0.2685 (0.2899)
========================================For Training [chexpert]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='CL_Fix5_final', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1152_epoch100_10-10-2023 20-07-44', trainable_backbone_layers=0)
=======================================================================================================

Best model has been saved to: [chexpert_CL_Fix5_final_f1_0_2253_precision_0_6471_accuracy_0_8694_recall_0_1364_auc_0_5622_epoch52_10-12-2023 00-38-05]
The final model has been saved to: [chexpert_CL_Fix5_final_f1_0_2399_precision_0_6390_accuracy_0_8697_recall_0_1477_auc_0_5671_epoch62_10-12-2023 00-55-09]

=======================================================================================================
{'f1': 0.266046511627907, 'precision': 0.7079207920792079, 'accuracy': 0.876138147566719, 'recall': 0.16380297823596793, 'auc': 0.5765349255378495}




Test:  [0/4]  eta: 0:00:31  model_time: 0.1160 (0.1160)  evaluator_time: 0.0000 (0.0000)  loss: 0.2949 (0.2949)  classification_loss: 0.2949 (0.2949)  time: 7.9642  data: 7.8166  max mem: 368
Test:  [3/4]  eta: 0:00:07  model_time: 0.0999 (0.1017)  evaluator_time: 0.0000 (0.0000)  loss: 0.2552 (0.2778)  classification_loss: 0.2552 (0.2778)  time: 7.0586  data: 6.9331  max mem: 368
Test: Total time: 0:00:28 (7.0591 s / it)
Averaged stats: model_time: 0.0999 (0.1017)  evaluator_time: 0.0000 (0.0000)  loss: 0.2552 (0.2778)  classification_loss: 0.2552 (0.2778)
========================================For Training [chexpert - CL_NoFix]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='CL_NoFix', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=5)
==================================================================================================================

Best model has been saved to: [chexpert_CL_NoFix_f1_0_3720_precision_0_6507_accuracy_0_8776_recall_0_2604_auc_0_6189_epoch52_10-12-2023 16-45-44]
The final model has been saved to: [chexpert_CL_NoFix_f1_0_3704_precision_0_6628_accuracy_0_8783_recall_0_2570_auc_0_6179_epoch62_10-12-2023 17-03-10]

==================================================================================================================
{'f1': 0.38585209003215437, 'precision': 0.6469002695417789, 'accuracy': 0.8800627943485086, 'recall': 0.27491408934707906, 'auc': 0.6255414543515456}
Test:  [0/4]  eta: 0:00:29  model_time: 0.1335 (0.1335)  evaluator_time: 0.0000 (0.0000)  loss: 0.2953 (0.2953)  classification_loss: 0.2953 (0.2953)  time: 7.2541  data: 7.0962  max mem: 368
Test:  [3/4]  eta: 0:00:06  model_time: 0.1335 (0.1307)  evaluator_time: 0.0000 (0.0000)  loss: 0.2558 (0.2782)  classification_loss: 0.2558 (0.2782)  time: 6.6758  data: 6.5214  max mem: 368
Test: Total time: 0:00:26 (6.6760 s / it)
Averaged stats: model_time: 0.1335 (0.1307)  evaluator_time: 0.0000 (0.0000)  loss: 0.2558 (0.2782)  classification_loss: 0.2558 (0.2782)
========================================For Training [chexpert - CL_Fix5]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='CL_Fix5', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=0)
=================================================================================================================

Best model has been saved to: [chexpert_CL_Fix5_f1_0_3768_precision_0_6814_accuracy_0_8801_recall_0_2604_auc_0_6204_epoch52_10-12-2023 18-32-07]
The final model has been saved to: [chexpert_CL_Fix5_f1_0_3670_precision_0_6637_accuracy_0_8782_recall_0_2537_auc_0_6164_epoch62_10-12-2023 18-49-09]

=================================================================================================================
{'f1': 0.38442822384428216, 'precision': 0.6583333333333333, 'accuracy': 0.8808477237048665, 'recall': 0.27147766323024053, 'auc': 0.6245509109311108}
Test:  [0/4]  eta: 0:00:29  model_time: 0.1268 (0.1268)  evaluator_time: 0.0000 (0.0000)  loss: 0.2941 (0.2941)  classification_loss: 0.2941 (0.2941)  time: 7.4483  data: 7.2964  max mem: 368
Test:  [3/4]  eta: 0:00:06  model_time: 0.1268 (0.1239)  evaluator_time: 0.0000 (0.0000)  loss: 0.2554 (0.2774)  classification_loss: 0.2554 (0.2774)  time: 6.7590  data: 6.6057  max mem: 368
Test: Total time: 0:00:27 (6.7595 s / it)
Averaged stats: model_time: 0.1268 (0.1239)  evaluator_time: 0.0000 (0.0000)  loss: 0.2554 (0.2774)  classification_loss: 0.2554 (0.2774)
========================================For Training [chexpert - CL_Fix2]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='CL_Fix2', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=3)
=================================================================================================================

Best model has been saved to: [chexpert_CL_Fix2_f1_0_3777_precision_0_6648_accuracy_0_8790_recall_0_2638_auc_0_6211_epoch52_10-12-2023 20-19-13]
The final model has been saved to: [chexpert_CL_Fix2_f1_0_3694_precision_0_6637_accuracy_0_8783_recall_0_2559_auc_0_6175_epoch62_10-12-2023 20-38-02]

=================================================================================================================
{'f1': 0.3951612903225806, 'precision': 0.667574931880109, 'accuracy': 0.8822605965463108, 'recall': 0.2806414662084765, 'auc': 0.6292237711249767}
Test:  [0/4]  eta: 0:00:31  model_time: 0.1691 (0.1691)  evaluator_time: 0.0000 (0.0000)  loss: 0.3113 (0.3113)  classification_loss: 0.3113 (0.3113)  time: 7.7934  data: 7.5953  max mem: 368
Test:  [3/4]  eta: 0:00:06  model_time: 0.1189 (0.1260)  evaluator_time: 0.0000 (0.0000)  loss: 0.2795 (0.2963)  classification_loss: 0.2795 (0.2963)  time: 6.8899  data: 6.7403  max mem: 368
Test: Total time: 0:00:27 (6.8902 s / it)
Averaged stats: model_time: 0.1189 (0.1260)  evaluator_time: 0.0000 (0.0000)  loss: 0.2795 (0.2963)  classification_loss: 0.2795 (0.2963)
========================================For Training [chexpert - imagenet]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='imagenet', weights='imagenet', cl_model_name=None, trainable_backbone_layers=3)
==================================================================================================================

Best model has been saved to: [chexpert_imagenet_f1_0_3462_precision_0_6056_accuracy_0_8725_recall_0_2424_auc_0_6084_epoch22_10-12-2023 21-16-25]
The final model has been saved to: [chexpert_imagenet_f1_0_3889_precision_0_5820_accuracy_0_8722_recall_0_2920_auc_0_6290_epoch32_10-12-2023 21-33-39]

==================================================================================================================
{'f1': 0.3615819209039548, 'precision': 0.6120218579234973, 'accuracy': 0.8758241758241758, 'recall': 0.2565864833906071, 'auc': 0.6153771056210813}
Test:  [0/4]  eta: 0:00:31  model_time: 0.1655 (0.1655)  evaluator_time: 0.0000 (0.0000)  loss: 0.3203 (0.3203)  classification_loss: 0.3203 (0.3203)  time: 7.8903  data: 7.6937  max mem: 368
Test:  [3/4]  eta: 0:00:06  model_time: 0.1105 (0.1313)  evaluator_time: 0.0000 (0.0000)  loss: 0.2953 (0.3109)  classification_loss: 0.2953 (0.3109)  time: 6.9464  data: 6.7902  max mem: 368
Test: Total time: 0:00:27 (6.9466 s / it)
Averaged stats: model_time: 0.1105 (0.1313)  evaluator_time: 0.0000 (0.0000)  loss: 0.2953 (0.3109)  classification_loss: 0.2953 (0.3109)
========================================For Training [chexpert - random]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='random', weights=None, cl_model_name=None, trainable_backbone_layers=5)
================================================================================================================

Best model has been saved to: [chexpert_random_f1_0_1298_precision_0_6465_accuracy_0_8653_recall_0_0722_auc_0_5329_epoch52_10-12-2023 23-04-34]
The final model has been saved to: [chexpert_random_f1_0_1635_precision_0_7069_accuracy_0_8683_recall_0_0924_auc_0_5431_epoch62_10-12-2023 23-22-06]

================================================================================================================
{'f1': 0.11752360965372508, 'precision': 0.7, 'accuracy': 0.8679748822605965, 'recall': 0.06414662084765177, 'auc': 0.5298903015098728}



################### REAL


Test:  [0/4]  eta: 0:00:44  model_time: 0.0875 (0.0875)  evaluator_time: 0.0000 (0.0000)  loss: 0.2908 (0.2908)  classification_loss: 0.2908 (0.2908)  time: 11.1976  data: 11.0771  max mem: 1039
Test:  [3/4]  eta: 0:00:09  model_time: 0.0875 (0.0786)  evaluator_time: 0.0000 (0.0000)  loss: 0.2422 (0.2664)  classification_loss: 0.2422 (0.2664)  time: 9.4695  data: 9.3629  max mem: 1039
Test: Total time: 0:00:37 (9.4700 s / it)
Averaged stats: model_time: 0.0875 (0.0786)  evaluator_time: 0.0000 (0.0000)  loss: 0.2422 (0.2664)  classification_loss: 0.2422 (0.2664)
========================================For Training [chexpert - CL_NoFix]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='CL_NoFix', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=5, release_fixed_weights_after=2)
==================================================================================================================

Best model has been saved to: [chexpert_CL_NoFix_f1_0_4891_precision_0_6828_accuracy_0_8892_recall_0_3811_auc_0_6762_epoch18_10-13-2023 02-04-16]
The final model has been saved to: [chexpert_CL_NoFix_f1_0_5198_precision_0_6403_accuracy_0_8874_recall_0_4374_auc_0_6988_epoch28_10-13-2023 02-21-40]

==================================================================================================================
{'f1': 0.502164502164502, 'precision': 0.6783625730994152, 'accuracy': 0.8916797488226059, 'recall': 0.39862542955326463, 'auc': 0.684304528493205}
Test:  [0/4]  eta: 0:00:40  model_time: 0.0885 (0.0885)  evaluator_time: 0.0000 (0.0000)  loss: 0.2953 (0.2953)  classification_loss: 0.2953 (0.2953)  time: 10.1473  data: 10.0288  max mem: 1039
Test:  [3/4]  eta: 0:00:09  model_time: 0.0885 (0.0794)  evaluator_time: 0.0000 (0.0000)  loss: 0.2558 (0.2782)  classification_loss: 0.2558 (0.2782)  time: 9.0833  data: 8.9763  max mem: 1039
Test: Total time: 0:00:36 (9.0838 s / it)
Averaged stats: model_time: 0.0885 (0.0794)  evaluator_time: 0.0000 (0.0000)  loss: 0.2558 (0.2782)  classification_loss: 0.2558 (0.2782)
========================================For Training [chexpert - CL_Fix5Layers]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='CL_Fix5Layers', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=0, release_fixed_weights_after=None)
=======================================================================================================================

Best model has been saved to: [chexpert_CL_Fix5Layers_f1_0_3768_precision_0_6814_accuracy_0_8801_recall_0_2604_auc_0_6204_epoch52_10-13-2023 03-53-20]
The final model has been saved to: [chexpert_CL_Fix5Layers_f1_0_3670_precision_0_6637_accuracy_0_8782_recall_0_2537_auc_0_6164_epoch62_10-13-2023 04-11-45]

=======================================================================================================================
{'f1': 0.38442822384428216, 'precision': 0.6583333333333333, 'accuracy': 0.8808477237048665, 'recall': 0.27147766323024053, 'auc': 0.6245509109311108}
Test:  [0/4]  eta: 0:00:40  model_time: 0.0875 (0.0875)  evaluator_time: 0.0000 (0.0000)  loss: 0.2855 (0.2855)  classification_loss: 0.2855 (0.2855)  time: 10.1354  data: 10.0158  max mem: 1039
Test:  [3/4]  eta: 0:00:09  model_time: 0.0855 (0.0778)  evaluator_time: 0.0000 (0.0000)  loss: 0.2445 (0.2642)  classification_loss: 0.2445 (0.2642)  time: 9.0843  data: 8.9795  max mem: 1039
Test: Total time: 0:00:36 (9.0846 s / it)
Averaged stats: model_time: 0.0855 (0.0778)  evaluator_time: 0.0000 (0.0000)  loss: 0.2445 (0.2642)  classification_loss: 0.2445 (0.2642)
========================================For Training [chexpert - CL_Fix2Layers]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='CL_Fix2Layers', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=3, release_fixed_weights_after=None)
=======================================================================================================================

Best model has been saved to: [chexpert_CL_Fix2Layers_f1_0_4932_precision_0_6705_accuracy_0_8884_recall_0_3901_auc_0_6795_epoch20_10-13-2023 04-50-40]
The final model has been saved to: [chexpert_CL_Fix2Layers_f1_0_5341_precision_0_6389_accuracy_0_8885_recall_0_4589_auc_0_7085_epoch30_10-13-2023 05-14-19]

=======================================================================================================================
{'f1': 0.5139186295503212, 'precision': 0.6818181818181818, 'accuracy': 0.8930926216640502, 'recall': 0.41237113402061853, 'auc': 0.6909045046126377}
Test:  [0/4]  eta: 0:00:40  model_time: 0.0855 (0.0855)  evaluator_time: 0.0000 (0.0000)  loss: 0.2867 (0.2867)  classification_loss: 0.2867 (0.2867)  time: 10.1422  data: 10.0247  max mem: 1039
Test:  [3/4]  eta: 0:00:09  model_time: 0.0855 (0.0784)  evaluator_time: 0.0000 (0.0000)  loss: 0.2448 (0.2660)  classification_loss: 0.2448 (0.2660)  time: 9.0823  data: 8.9762  max mem: 1039
Test: Total time: 0:00:36 (9.0825 s / it)
Averaged stats: model_time: 0.0855 (0.0784)  evaluator_time: 0.0000 (0.0000)  loss: 0.2448 (0.2660)  classification_loss: 0.2448 (0.2660)
========================================For Training [chexpert - CL_Fix20Epoch]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='CL_Fix20Epoch', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=0, release_fixed_weights_after=20)
=======================================================================================================================

Best model has been saved to: [chexpert_CL_Fix20Epoch_f1_0_5067_precision_0_6774_accuracy_0_8903_recall_0_4047_auc_0_6868_epoch34_10-13-2023 06-34-21]
The final model has been saved to: [chexpert_CL_Fix20Epoch_f1_0_5206_precision_0_6358_accuracy_0_8870_recall_0_4408_auc_0_7000_epoch44_10-13-2023 06-58-12]

=======================================================================================================================
{'f1': 0.5098314606741573, 'precision': 0.6588021778584392, 'accuracy': 0.8904238618524333, 'recall': 0.41580756013745707, 'auc': 0.6908035435760962}
Test:  [0/4]  eta: 0:00:40  model_time: 0.0875 (0.0875)  evaluator_time: 0.0000 (0.0000)  loss: 0.3153 (0.3153)  classification_loss: 0.3153 (0.3153)  time: 10.1433  data: 10.0248  max mem: 1039
Test:  [3/4]  eta: 0:00:09  model_time: 0.0875 (0.0788)  evaluator_time: 0.0000 (0.0000)  loss: 0.2481 (0.2763)  classification_loss: 0.2481 (0.2763)  time: 9.0996  data: 8.9943  max mem: 1039
Test: Total time: 0:00:36 (9.0996 s / it)
Averaged stats: model_time: 0.0875 (0.0788)  evaluator_time: 0.0000 (0.0000)  loss: 0.2481 (0.2763)  classification_loss: 0.2481 (0.2763)
========================================For Training [chexpert - ImageNet]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='ImageNet', weights='imagenet', cl_model_name=None, trainable_backbone_layers=3, release_fixed_weights_after=None)
==================================================================================================================

Best model has been saved to: [chexpert_ImageNet_f1_0_4461_precision_0_6727_accuracy_0_8846_recall_0_3337_auc_0_6537_epoch6_10-13-2023 07-12-30]
The final model has been saved to: [chexpert_ImageNet_f1_0_5082_precision_0_6475_accuracy_0_8873_recall_0_4183_auc_0_6907_epoch16_10-13-2023 07-36-20]

==================================================================================================================
{'f1': 0.4741959611069559, 'precision': 0.6831896551724138, 'accuracy': 0.8896389324960754, 'recall': 0.3631156930126002, 'auc': 0.6681869169083375}
Test:  [0/4]  eta: 0:00:41  model_time: 0.0877 (0.0877)  evaluator_time: 0.0000 (0.0000)  loss: 0.2966 (0.2966)  classification_loss: 0.2966 (0.2966)  time: 10.2973  data: 10.1794  max mem: 1039
Test:  [3/4]  eta: 0:00:09  model_time: 0.0865 (0.0786)  evaluator_time: 0.0000 (0.0000)  loss: 0.2648 (0.2863)  classification_loss: 0.2648 (0.2863)  time: 9.1615  data: 9.0564  max mem: 1039
Test: Total time: 0:00:36 (9.1620 s / it)
Averaged stats: model_time: 0.0865 (0.0786)  evaluator_time: 0.0000 (0.0000)  loss: 0.2648 (0.2863)  classification_loss: 0.2648 (0.2863)
========================================For Training [chexpert - Random]========================================
ImageClassificationArgs(name='chexpert', learning_rate=0.03, sgd_momentum=0.9, batch_size=128, weight_decay=0.0001, early_stopping_patience=10, warmup_epoch=0)
REFLACXCheXpertDatasetArgs(image_size=128, label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'])
ResNetClassifierArgs(name='Random', weights=None, cl_model_name=None, trainable_backbone_layers=5, release_fixed_weights_after=None)
================================================================================================================

Best model has been saved to: [chexpert_Random_f1_0_3160_precision_0_6868_accuracy_0_8763_recall_0_2052_auc_0_5950_epoch9_10-13-2023 07-58-01]
The final model has been saved to: [chexpert_Random_f1_0_4354_precision_0_6303_accuracy_0_8799_recall_0_3326_auc_0_6505_epoch19_10-13-2023 08-21-55]

================================================================================================================
{'f1': 0.3524305555555556, 'precision': 0.7275985663082437, 'accuracy': 0.8828885400313972, 'recall': 0.2325315005727377, 'auc': 0.6093528887255175}



creating index...
index created!
Test:  [  0/114]  eta: 0:00:40  model_time: 0.1685 (0.1685)  evaluator_time: 0.0045 (0.0045)  loss: 0.4398 (0.4398)  loss_classifier: 0.2201 (0.2201)  loss_box_reg: 0.1945 (0.1945)  loss_objectness: 0.0145 (0.0145)  loss_rpn_box_reg: 0.0107 (0.0107)  time: 0.3526  data: 0.1746  max mem: 2636
Test:  [100/114]  eta: 0:00:03  model_time: 0.0709 (0.0686)  evaluator_time: 0.0020 (0.0025)  loss: 0.3258 (0.3157)  loss_classifier: 0.1626 (0.1612)  loss_box_reg: 0.1300 (0.1257)  loss_objectness: 0.0171 (0.0212)  loss_rpn_box_reg: 0.0066 (0.0075)  time: 0.2494  data: 0.1757  max mem: 2636
Test:  [113/114]  eta: 0:00:00  model_time: 0.0700 (0.0687)  evaluator_time: 0.0020 (0.0025)  loss: 0.3258 (0.3177)  loss_classifier: 0.1492 (0.1622)  loss_box_reg: 0.1179 (0.1259)  loss_objectness: 0.0191 (0.0219)  loss_rpn_box_reg: 0.0086 (0.0077)  time: 0.2553  data: 0.1822  max mem: 2636
Test: Total time: 0:00:28 (0.2534 s / it)
Averaged stats: model_time: 0.0700 (0.0687)  evaluator_time: 0.0020 (0.0025)  loss: 0.3258 (0.3177)  loss_classifier: 0.1492 (0.1622)  loss_box_reg: 0.1179 (0.1259)  loss_objectness: 0.0191 (0.0219)  loss_rpn_box_reg: 0.0086 (0.0077)
Accumulating evaluation results...
DONE (t=0.09s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.114
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.503
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.043
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.026
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.053
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.102
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.194
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.198
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.098
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
========================================For Training [lesion_detection - CL_Fix2Layers]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.01, sgd_momentum=0.9, batch_size=4, weight_decay=1e-05, early_stopping_patience=30, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
FasterRCNNArgs(name='CL_Fix2Layers', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=3, release_fixed_weights_after=None)
===============================================================================================================================

Best model has been saved to: [lesion_detection_CL_Fix2Layers_ap_0_1065_ar_0_4989_epoch3_10-14-2023 20-43-49]
The final model has been saved to: [lesion_detection_CL_Fix2Layers_ap_0_0905_ar_0_3189_epoch33_10-14-2023 22-24-02]

===============================================================================================================================
{'ap': 0.11392938279454032, 'ar': 0.5027021183508772}
creating index...
index created!
Test:  [  0/114]  eta: 0:00:38  model_time: 0.1608 (0.1608)  evaluator_time: 0.0030 (0.0030)  loss: 0.4629 (0.4629)  loss_classifier: 0.2354 (0.2354)  loss_box_reg: 0.2053 (0.2053)  loss_objectness: 0.0121 (0.0121)  loss_rpn_box_reg: 0.0099 (0.0099)  time: 0.3417  data: 0.1738  max mem: 2636
Test:  [100/114]  eta: 0:00:03  model_time: 0.0659 (0.0692)  evaluator_time: 0.0020 (0.0024)  loss: 0.3096 (0.3087)  loss_classifier: 0.1560 (0.1568)  loss_box_reg: 0.1310 (0.1262)  loss_objectness: 0.0169 (0.0185)  loss_rpn_box_reg: 0.0065 (0.0071)  time: 0.2458  data: 0.1763  max mem: 2636
Test:  [113/114]  eta: 0:00:00  model_time: 0.0670 (0.0690)  evaluator_time: 0.0020 (0.0024)  loss: 0.3140 (0.3102)  loss_classifier: 0.1480 (0.1575)  loss_box_reg: 0.1212 (0.1266)  loss_objectness: 0.0183 (0.0188)  loss_rpn_box_reg: 0.0079 (0.0073)  time: 0.2536  data: 0.1836  max mem: 2636
Test: Total time: 0:00:29 (0.2552 s / it)
Averaged stats: model_time: 0.0670 (0.0690)  evaluator_time: 0.0020 (0.0024)  loss: 0.3140 (0.3102)  loss_classifier: 0.1480 (0.1575)  loss_box_reg: 0.1212 (0.1266)  loss_objectness: 0.0183 (0.0188)  loss_rpn_box_reg: 0.0079 (0.0073)
Accumulating evaluation results...
DONE (t=0.10s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.115
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.506
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.041
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.026
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.006
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.050
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.106
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.198
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.098
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.242
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
========================================For Training [lesion_detection - ImageNet_Fix2Layers]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.01, sgd_momentum=0.9, batch_size=4, weight_decay=1e-05, early_stopping_patience=30, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
FasterRCNNArgs(name='ImageNet_Fix2Layers', weights='imagenet', cl_model_name=None, trainable_backbone_layers=3, release_fixed_weights_after=None)
=====================================================================================================================================

Best model has been saved to: [lesion_detection_ImageNet_Fix2Layers_ap_0_1142_ar_0_5371_epoch3_10-14-2023 22-34-55]
The final model has been saved to: [lesion_detection_ImageNet_Fix2Layers_ap_0_0907_ar_0_3035_epoch33_10-15-2023 00-19-10]

=====================================================================================================================================
{'ap': 0.11467119354043041, 'ar': 0.5056022177414562}
creating index...
index created!
Test:  [  0/114]  eta: 0:00:42  model_time: 0.1919 (0.1919)  evaluator_time: 0.0020 (0.0020)  loss: 0.4008 (0.4008)  loss_classifier: 0.2086 (0.2086)  loss_box_reg: 0.1566 (0.1566)  loss_objectness: 0.0219 (0.0219)  loss_rpn_box_reg: 0.0136 (0.0136)  time: 0.3696  data: 0.1737  max mem: 2636
Test:  [100/114]  eta: 0:00:03  model_time: 0.0650 (0.0701)  evaluator_time: 0.0020 (0.0023)  loss: 0.2786 (0.2851)  loss_classifier: 0.1474 (0.1463)  loss_box_reg: 0.1069 (0.1077)  loss_objectness: 0.0174 (0.0235)  loss_rpn_box_reg: 0.0065 (0.0076)  time: 0.2440  data: 0.1755  max mem: 2636
Test:  [113/114]  eta: 0:00:00  model_time: 0.0670 (0.0700)  evaluator_time: 0.0025 (0.0023)  loss: 0.2810 (0.2874)  loss_classifier: 0.1406 (0.1476)  loss_box_reg: 0.1033 (0.1081)  loss_objectness: 0.0233 (0.0238)  loss_rpn_box_reg: 0.0089 (0.0078)  time: 0.2544  data: 0.1832  max mem: 2636
Test: Total time: 0:00:29 (0.2545 s / it)
Averaged stats: model_time: 0.0670 (0.0700)  evaluator_time: 0.0025 (0.0023)  loss: 0.2810 (0.2874)  loss_classifier: 0.1406 (0.1476)  loss_box_reg: 0.1033 (0.1081)  loss_objectness: 0.0233 (0.0238)  loss_rpn_box_reg: 0.0089 (0.0078)
Accumulating evaluation results...
DONE (t=0.18s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.106
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.005
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.085
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.182
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
========================================For Training [lesion_detection - random]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.01, sgd_momentum=0.9, batch_size=4, weight_decay=1e-05, early_stopping_patience=30, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
FasterRCNNArgs(name='random', weights=None, cl_model_name=None, trainable_backbone_layers=5, release_fixed_weights_after=None)
========================================================================================================================

Best model has been saved to: [lesion_detection_random_ap_0_1125_ar_0_4268_epoch3_10-15-2023 00-29-39]
The final model has been saved to: [lesion_detection_random_ap_0_0971_ar_0_3755_epoch33_10-15-2023 02-18-42]

========================================================================================================================
{'ap': 0.10555122316240648, 'ar': 0.40414065193561965}

creating index...
index created!
Test:  [  0/114]  eta: 0:01:07  model_time: 0.2859 (0.2859)  evaluator_time: 0.0060 (0.0060)  loss: 0.4472 (0.4472)  loss_classifier: 0.2171 (0.2171)  loss_box_reg: 0.2055 (0.2055)  loss_objectness: 0.0144 (0.0144)  loss_rpn_box_reg: 0.0102 (0.0102)  time: 0.5934  data: 0.2995  max mem: 1535
Test:  [100/114]  eta: 0:00:05  model_time: 0.0925 (0.1040)  evaluator_time: 0.0050 (0.0048)  loss: 0.3208 (0.2926)  loss_classifier: 0.1620 (0.1501)  loss_box_reg: 0.1162 (0.1098)  loss_objectness: 0.0176 (0.0254)  loss_rpn_box_reg: 0.0055 (0.0073)  time: 0.3971  data: 0.2942  max mem: 1535
Test:  [113/114]  eta: 0:00:00  model_time: 0.1061 (0.1047)  evaluator_time: 0.0050 (0.0048)  loss: 0.3101 (0.2941)  loss_classifier: 0.1496 (0.1506)  loss_box_reg: 0.1085 (0.1100)  loss_objectness: 0.0282 (0.0260)  loss_rpn_box_reg: 0.0071 (0.0074)  time: 0.4221  data: 0.3071  max mem: 1535
Test: Total time: 0:00:46 (0.4112 s / it)
Averaged stats: model_time: 0.1061 (0.1047)  evaluator_time: 0.0050 (0.0048)  loss: 0.3101 (0.2941)  loss_classifier: 0.1496 (0.1506)  loss_box_reg: 0.1085 (0.1100)  loss_objectness: 0.0282 (0.0260)  loss_rpn_box_reg: 0.0071 (0.0074)
Accumulating evaluation results...
DONE (t=0.15s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.099
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.033
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.017
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.038
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.090
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.162
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.165
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.196
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
========================================For Training [lesion_detection - CL_Fix5Layers]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.01, sgd_momentum=0.9, batch_size=4, weight_decay=1e-05, early_stopping_patience=30, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
FasterRCNNArgs(name='CL_Fix5Layers', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=0, release_fixed_weights_after=None)
===============================================================================================================================

Best model has been saved to: [lesion_detection_CL_Fix5Layers_ap_0_0787_ar_0_4175_epoch30_10-15-2023 07-05-23]
The final model has been saved to: [lesion_detection_CL_Fix5Layers_ap_0_0664_ar_0_3621_epoch60_10-15-2023 09-40-02]

===============================================================================================================================
{'ap': 0.09863259583147095, 'ar': 0.4694049155536745}
creating index...
index created!
Test:  [  0/114]  eta: 0:01:11  model_time: 0.3201 (0.3201)  evaluator_time: 0.0060 (0.0060)  loss: 0.4176 (0.4176)  loss_classifier: 0.2204 (0.2204)  loss_box_reg: 0.1694 (0.1694)  loss_objectness: 0.0167 (0.0167)  loss_rpn_box_reg: 0.0111 (0.0111)  time: 0.6234  data: 0.2952  max mem: 1535
Test:  [100/114]  eta: 0:00:05  model_time: 0.1010 (0.1121)  evaluator_time: 0.0040 (0.0043)  loss: 0.3063 (0.3012)  loss_classifier: 0.1558 (0.1579)  loss_box_reg: 0.1189 (0.1117)  loss_objectness: 0.0178 (0.0243)  loss_rpn_box_reg: 0.0059 (0.0073)  time: 0.4229  data: 0.2966  max mem: 1535
Test:  [113/114]  eta: 0:00:00  model_time: 0.1045 (0.1127)  evaluator_time: 0.0040 (0.0043)  loss: 0.3023 (0.3029)  loss_classifier: 0.1514 (0.1584)  loss_box_reg: 0.1134 (0.1123)  loss_objectness: 0.0226 (0.0247)  loss_rpn_box_reg: 0.0076 (0.0074)  time: 0.4341  data: 0.3139  max mem: 1535
Test: Total time: 0:00:48 (0.4211 s / it)
Averaged stats: model_time: 0.1045 (0.1127)  evaluator_time: 0.0040 (0.0043)  loss: 0.3023 (0.3029)  loss_classifier: 0.1514 (0.1584)  loss_box_reg: 0.1134 (0.1123)  loss_objectness: 0.0226 (0.0247)  loss_rpn_box_reg: 0.0076 (0.0074)
Accumulating evaluation results...
DONE (t=0.13s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.095
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.018
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.048
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.098
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.161
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.163
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.196
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
========================================For Training [lesion_detection - ImageNet_Fix5layers]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.01, sgd_momentum=0.9, batch_size=4, weight_decay=1e-05, early_stopping_patience=30, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
FasterRCNNArgs(name='ImageNet_Fix5layers', weights='imagenet', cl_model_name=None, trainable_backbone_layers=0, release_fixed_weights_after=None)
=====================================================================================================================================

Best model has been saved to: [lesion_detection_ImageNet_Fix5layers_ap_0_0970_ar_0_4153_epoch22_10-15-2023 11-33-40]
The final model has been saved to: [lesion_detection_ImageNet_Fix5layers_ap_0_0856_ar_0_3633_epoch52_10-15-2023 14-08-17]

=====================================================================================================================================
{'ap': 0.09494137898589142, 'ar': 0.42209643977240036}


### Reduce faster-rcnn size


Test:  [  0/114]  eta: 0:01:04  model_time: 0.3587 (0.3587)  evaluator_time: 0.0010 (0.0010)  loss: 0.4007 (0.4007)  loss_classifier: 0.1908 (0.1908)  loss_box_reg: 0.1787 (0.1787)  loss_objectness: 0.0214 (0.0214)  loss_rpn_box_reg: 0.0097 (0.0097)  time: 0.5660  data: 0.2054  max mem: 2501
Test:  [100/114]  eta: 0:00:05  model_time: 0.1750 (0.1855)  evaluator_time: 0.0011 (0.0009)  loss: 0.2772 (0.2757)  loss_classifier: 0.1475 (0.1375)  loss_box_reg: 0.1089 (0.1060)  loss_objectness: 0.0153 (0.0252)  loss_rpn_box_reg: 0.0052 (0.0070)  time: 0.3739  data: 0.1933  max mem: 2501
Test:  [113/114]  eta: 0:00:00  model_time: 0.1755 (0.1851)  evaluator_time: 0.0015 (0.0010)  loss: 0.2613 (0.2762)  loss_classifier: 0.1283 (0.1373)  loss_box_reg: 0.0883 (0.1058)  loss_objectness: 0.0294 (0.0259)  loss_rpn_box_reg: 0.0077 (0.0072)  time: 0.3842  data: 0.2029  max mem: 2501
Test: Total time: 0:00:44 (0.3868 s / it)
Averaged stats: model_time: 0.1755 (0.1851)  evaluator_time: 0.0015 (0.0010)  loss: 0.2613 (0.2762)  loss_classifier: 0.1283 (0.1373)  loss_box_reg: 0.0883 (0.1058)  loss_objectness: 0.0294 (0.0259)  loss_rpn_box_reg: 0.0077 (0.0072)
========================================For Training [lesion_detection - CL_Fix5Layers]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.005, sgd_momentum=0.9, batch_size=4, weight_decay=0.0005, early_stopping_patience=20, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
FasterRCNNArgs(name='CL_Fix5Layers', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=0, release_fixed_weights_after=None)
===============================================================================================================================

Best model has been saved to: [lesion_detection_CL_Fix5Layers_map_50_0_2072_mar_100_0_3192_epoch30_10-16-2023 12-34-01]
The final model has been saved to: [lesion_detection_CL_Fix5Layers_map_50_0_1830_mar_100_0_3145_epoch50_10-16-2023 14-49-34]

===============================================================================================================================
{'map': 0.06794673204421997, 'map_50': 0.19868482649326324, 'map_75': 0.02073505148291588, 'map_small': 0.01562691293656826, 'map_medium': 0.08279593288898468, 'map_large': -1.0, 'mar_1': 0.1386108100414276, 'mar_10': 0.27955126762390137, 'mar_100': 0.3224300146102905, 'mar_small': 0.22840715944766998, 'mar_medium': 0.3681583106517792, 'mar_large': -1.0, 'map_per_class': -1.0, 'mar_100_per_class': -1.0}
Test:  [  0/114]  eta: 0:00:46  model_time: 0.2401 (0.2401)  evaluator_time: 0.0000 (0.0000)  loss: 0.3059 (0.3059)  loss_classifier: 0.1464 (0.1464)  loss_box_reg: 0.1138 (0.1138)  loss_objectness: 0.0368 (0.0368)  loss_rpn_box_reg: 0.0089 (0.0089)  time: 0.4041  data: 0.1639  max mem: 2501
Test:  [100/114]  eta: 0:00:04  model_time: 0.1718 (0.1804)  evaluator_time: 0.0010 (0.0009)  loss: 0.2788 (0.2662)  loss_classifier: 0.1341 (0.1286)  loss_box_reg: 0.1131 (0.0997)  loss_objectness: 0.0251 (0.0308)  loss_rpn_box_reg: 0.0058 (0.0072)  time: 0.3440  data: 0.1673  max mem: 2501
Test:  [113/114]  eta: 0:00:00  model_time: 0.1734 (0.1797)  evaluator_time: 0.0010 (0.0009)  loss: 0.2608 (0.2683)  loss_classifier: 0.1195 (0.1292)  loss_box_reg: 0.0745 (0.0999)  loss_objectness: 0.0332 (0.0318)  loss_rpn_box_reg: 0.0089 (0.0074)  time: 0.3508  data: 0.1749  max mem: 2501
Test: Total time: 0:00:40 (0.3523 s / it)
Averaged stats: model_time: 0.1734 (0.1797)  evaluator_time: 0.0010 (0.0009)  loss: 0.2608 (0.2683)  loss_classifier: 0.1195 (0.1292)  loss_box_reg: 0.0745 (0.0999)  loss_objectness: 0.0332 (0.0318)  loss_rpn_box_reg: 0.0089 (0.0074)
========================================For Training [lesion_detection - ImageNet_Fix5layers]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.005, sgd_momentum=0.9, batch_size=4, weight_decay=0.0005, early_stopping_patience=20, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
FasterRCNNArgs(name='ImageNet_Fix5layers', weights='imagenet', cl_model_name=None, trainable_backbone_layers=0, release_fixed_weights_after=None)
=====================================================================================================================================

Best model has been saved to: [lesion_detection_ImageNet_Fix5layers_map_50_0_1294_mar_100_0_2542_epoch2_10-16-2023 14-57-44]
The final model has been saved to: [lesion_detection_ImageNet_Fix5layers_map_50_0_1874_mar_100_0_2995_epoch22_10-16-2023 16-52-28]

=====================================================================================================================================
{'map': 0.04249822348356247, 'map_50': 0.138482466340065, 'map_75': 0.010896474123001099, 'map_small': 0.01604882813990116, 'map_medium': 0.051021791994571686, 'map_large': -1.0, 'mar_1': 0.1132112666964531, 'mar_10': 0.21850590407848358, 'mar_100': 0.259224534034729, 'mar_small': 0.13501596450805664, 'mar_medium': 0.3199182450771332, 'mar_large': -1.0, 'map_per_class': -1.0, 'mar_100_per_class': -1.0}
Test:  [  0/114]  eta: 0:00:47  model_time: 0.2465 (0.2465)  evaluator_time: 0.0010 (0.0010)  loss: 0.3565 (0.3565)  loss_classifier: 0.1779 (0.1779)  loss_box_reg: 0.1433 (0.1433)  loss_objectness: 0.0244 (0.0244)  loss_rpn_box_reg: 0.0108 (0.0108)  time: 0.4134  data: 0.1644  max mem: 2501
Test:  [100/114]  eta: 0:00:05  model_time: 0.1792 (0.1962)  evaluator_time: 0.0015 (0.0010)  loss: 0.2807 (0.2778)  loss_classifier: 0.1448 (0.1384)  loss_box_reg: 0.1095 (0.1046)  loss_objectness: 0.0215 (0.0275)  loss_rpn_box_reg: 0.0062 (0.0074)  time: 0.3538  data: 0.1690  max mem: 2501
Test:  [113/114]  eta: 0:00:00  model_time: 0.1853 (0.1959)  evaluator_time: 0.0015 (0.0011)  loss: 0.2663 (0.2812)  loss_classifier: 0.1377 (0.1399)  loss_box_reg: 0.0975 (0.1057)  loss_objectness: 0.0267 (0.0280)  loss_rpn_box_reg: 0.0079 (0.0075)  time: 0.3663  data: 0.1751  max mem: 2501
Test: Total time: 0:00:42 (0.3706 s / it)
Averaged stats: model_time: 0.1853 (0.1959)  evaluator_time: 0.0015 (0.0011)  loss: 0.2663 (0.2812)  loss_classifier: 0.1377 (0.1399)  loss_box_reg: 0.0975 (0.1057)  loss_objectness: 0.0267 (0.0280)  loss_rpn_box_reg: 0.0079 (0.0075)
========================================For Training [lesion_detection - Random]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.005, sgd_momentum=0.9, batch_size=4, weight_decay=0.0005, early_stopping_patience=20, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
FasterRCNNArgs(name='Random', weights=None, cl_model_name=None, trainable_backbone_layers=5, release_fixed_weights_after=None)
========================================================================================================================

Best model has been saved to: [lesion_detection_Random_map_50_0_1230_mar_100_0_2645_epoch3_10-16-2023 17-12-51]
The final model has been saved to: [lesion_detection_Random_map_50_0_2099_mar_100_0_3469_epoch23_10-16-2023 18-58-05]

========================================================================================================================
{'map': 0.045811980962753296, 'map_50': 0.1469825953245163, 'map_75': 0.010914324782788754, 'map_small': 0.020836230367422104, 'map_medium': 0.05647159740328789, 'map_large': -1.0, 'mar_1': 0.11470159143209457, 'mar_10': 0.24236318469047546, 'mar_100': 0.27778497338294983, 'mar_small': 0.14562425017356873, 'mar_medium': 0.34432175755500793, 'mar_large': -1.0, 'map_per_class': -1.0, 'mar_100_per_class': -1.0}






#### DETR

Test:  [  0/114]  eta: 0:00:23  loss: 13.0290 (13.0290)  time: 0.2088  data: 0.1838  max mem: 477
Test:  [ 10/114]  eta: 0:00:22  loss: 14.8152 (13.9087)  time: 0.2149  data: 0.1905  max mem: 477
Test:  [ 20/114]  eta: 0:00:19  loss: 12.5311 (12.3251)  time: 0.2119  data: 0.1865  max mem: 477
Test:  [ 30/114]  eta: 0:00:17  loss: 11.2365 (12.3277)  time: 0.2060  data: 0.1800  max mem: 477
Test:  [ 40/114]  eta: 0:00:15  loss: 11.8390 (12.2564)  time: 0.2054  data: 0.1798  max mem: 477
Test:  [ 50/114]  eta: 0:00:13  loss: 10.8709 (11.8927)  time: 0.2057  data: 0.1803  max mem: 477
Test:  [ 60/114]  eta: 0:00:11  loss: 12.6689 (11.9114)  time: 0.2082  data: 0.1828  max mem: 477
Test:  [ 70/114]  eta: 0:00:09  loss: 12.5734 (11.9755)  time: 0.2072  data: 0.1821  max mem: 477
Test:  [ 80/114]  eta: 0:00:07  loss: 11.5433 (11.8154)  time: 0.2099  data: 0.1845  max mem: 477
Test:  [ 90/114]  eta: 0:00:05  loss: 11.8797 (11.8816)  time: 0.2117  data: 0.1857  max mem: 477
Test:  [100/114]  eta: 0:00:02  loss: 12.3338 (11.8880)  time: 0.2045  data: 0.1791  max mem: 477
Test:  [110/114]  eta: 0:00:00  loss: 12.2471 (11.9685)  time: 0.2054  data: 0.1800  max mem: 477
Test:  [113/114]  eta: 0:00:00  loss: 12.9312 (11.9011)  time: 0.2056  data: 0.1801  max mem: 477
Test: Total time: 0:00:23 (0.2078 s / it)
Averaged stats: loss: 12.9312 (11.9011)
========================================For Training [lesion_detection - CL_NoFix]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.0001, sgd_momentum=0.9, batch_size=4, weight_decay=0.0001, early_stopping_patience=20, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
DETRArgs(name='CL_NoFix', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=5, release_fixed_weights_after=None, hidden_dim=32, dilation=False, position_embedding='sine', dropout=0.1, nheads=4, dim_feedforward=64, enc_layers=3, dec_layers=3, pre_norm=False, num_queries=100, aux_loss=True, set_cost_class=1, set_cost_bbox=5, set_cost_giou=2, giou_loss_coef=2, bbox_loss_coef=5, eos_coef=0.1)
==========================================================================================================================

Best model has been saved to: [lesion_detection_CL_NoFix_map_50_0_0006_mar_100_0_0267_epoch30_10-17-2023 08-19-39]
The final model has been saved to: [lesion_detection_CL_NoFix_map_50_0_0062_mar_100_0_0481_epoch50_10-17-2023 09-07-23]

==========================================================================================================================
{'map': 0.0002875025093089789, 'map_50': 0.0010762253077700734, 'map_75': 0.0001451256248401478, 'map_small': 0.0002875025093089789, 'map_medium': -1.0, 'map_large': -1.0, 'mar_1': 0.010582302697002888, 'mar_10': 0.021824738010764122, 'mar_100': 0.03280381113290787, 'mar_small': 0.03280381113290787, 'mar_medium': -1.0, 'mar_large': -1.0, 'map_per_class': -1.0, 'mar_100_per_class': -1.0}
Test:  [  0/114]  eta: 0:00:20  loss: 16.3449 (16.3449)  time: 0.1791  data: 0.1535  max mem: 477
Test:  [ 10/114]  eta: 0:00:19  loss: 17.4528 (17.2904)  time: 0.1837  data: 0.1615  max mem: 477
Test:  [ 20/114]  eta: 0:00:17  loss: 15.9837 (16.3381)  time: 0.1828  data: 0.1593  max mem: 477
Test:  [ 30/114]  eta: 0:00:15  loss: 15.9180 (16.2206)  time: 0.1810  data: 0.1558  max mem: 477
Test:  [ 40/114]  eta: 0:00:13  loss: 17.1826 (16.7320)  time: 0.1812  data: 0.1559  max mem: 477
Test:  [ 50/114]  eta: 0:00:11  loss: 17.5904 (16.4389)  time: 0.1817  data: 0.1565  max mem: 477
Test:  [ 60/114]  eta: 0:00:09  loss: 16.4581 (16.1274)  time: 0.1860  data: 0.1607  max mem: 477
Test:  [ 70/114]  eta: 0:00:08  loss: 16.3554 (16.0639)  time: 0.1871  data: 0.1614  max mem: 477
Test:  [ 80/114]  eta: 0:00:06  loss: 16.2142 (15.8768)  time: 0.1896  data: 0.1634  max mem: 477
Test:  [ 90/114]  eta: 0:00:04  loss: 16.1076 (15.9183)  time: 0.1875  data: 0.1617  max mem: 477
Test:  [100/114]  eta: 0:00:02  loss: 16.2126 (15.9549)  time: 0.1820  data: 0.1567  max mem: 477
Test:  [110/114]  eta: 0:00:00  loss: 16.0553 (16.0633)  time: 0.1876  data: 0.1617  max mem: 477
Test:  [113/114]  eta: 0:00:00  loss: 16.0553 (16.0466)  time: 0.1889  data: 0.1628  max mem: 477
Test: Total time: 0:00:21 (0.1849 s / it)
Averaged stats: loss: 16.0553 (16.0466)
========================================For Training [lesion_detection - CL_Fix5Layers]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.0001, sgd_momentum=0.9, batch_size=4, weight_decay=0.0001, early_stopping_patience=20, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
DETRArgs(name='CL_Fix5Layers', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=0, release_fixed_weights_after=None, hidden_dim=32, dilation=False, position_embedding='sine', dropout=0.1, nheads=4, dim_feedforward=64, enc_layers=3, dec_layers=3, pre_norm=False, num_queries=100, aux_loss=True, set_cost_class=1, set_cost_bbox=5, set_cost_giou=2, giou_loss_coef=2, bbox_loss_coef=5, eos_coef=0.1)
===============================================================================================================================

Best model has been saved to: [lesion_detection_CL_Fix5Layers_map_50_0_0000_mar_100_0_0012_epoch30_10-17-2023 10-11-43]
The final model has been saved to: [lesion_detection_CL_Fix5Layers_map_50_0_0000_mar_100_0_0064_epoch50_10-17-2023 10-54-39]

===============================================================================================================================
{'map': 3.3370308756275335e-06, 'map_50': 1.786428583727684e-05, 'map_75': 0.0, 'map_small': 3.3370308756275335e-06, 'map_medium': -1.0, 'map_large': -1.0, 'mar_1': 0.0016735185636207461, 'mar_10': 0.001896077417768538, 'mar_100': 0.002349850023165345, 'mar_small': 0.002349850023165345, 'mar_medium': -1.0, 'mar_large': -1.0, 'map_per_class': -1.0, 'mar_100_per_class': -1.0}
Test:  [  0/114]  eta: 0:00:20  loss: 11.2902 (11.2902)  time: 0.1828  data: 0.1528  max mem: 479
Test:  [ 10/114]  eta: 0:00:19  loss: 15.7460 (14.9923)  time: 0.1881  data: 0.1622  max mem: 479
Test:  [ 20/114]  eta: 0:00:17  loss: 14.6711 (13.4482)  time: 0.1859  data: 0.1603  max mem: 479
Test:  [ 30/114]  eta: 0:00:15  loss: 12.7380 (13.5047)  time: 0.1826  data: 0.1566  max mem: 479
Test:  [ 40/114]  eta: 0:00:13  loss: 14.2343 (13.7000)  time: 0.1826  data: 0.1566  max mem: 479
Test:  [ 50/114]  eta: 0:00:11  loss: 13.8720 (13.3164)  time: 0.1827  data: 0.1571  max mem: 479
Test:  [ 60/114]  eta: 0:00:09  loss: 12.7376 (13.2213)  time: 0.1861  data: 0.1606  max mem: 479
Test:  [ 70/114]  eta: 0:00:08  loss: 13.5683 (13.0699)  time: 0.1868  data: 0.1612  max mem: 479
Test:  [ 80/114]  eta: 0:00:06  loss: 13.0450 (12.9387)  time: 0.1891  data: 0.1635  max mem: 479
Test:  [ 90/114]  eta: 0:00:04  loss: 13.5892 (13.1047)  time: 0.1878  data: 0.1617  max mem: 479
Test:  [100/114]  eta: 0:00:02  loss: 14.5197 (13.1093)  time: 0.1839  data: 0.1576  max mem: 479
Test:  [110/114]  eta: 0:00:00  loss: 14.0699 (13.2021)  time: 0.1900  data: 0.1636  max mem: 479
Test:  [113/114]  eta: 0:00:00  loss: 14.3609 (13.1645)  time: 0.1910  data: 0.1647  max mem: 479
Test: Total time: 0:00:21 (0.1862 s / it)
Averaged stats: loss: 14.3609 (13.1645)
========================================For Training [lesion_detection - CL_Fix2Layers]========================================
LesionDetectionArgs(name='lesion_detection', learning_rate=0.0001, sgd_momentum=0.9, batch_size=4, weight_decay=0.0001, early_stopping_patience=20, warmup_epoch=0)
REFLACXLesionDetectionDatasetArgs(image_size=128, label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'])
DETRArgs(name='CL_Fix2Layers', weights='cl', cl_model_name='MCL_resnet50_accuracy_0_1433_epoch48_10-08-2023 15-58-25', trainable_backbone_layers=3, release_fixed_weights_after=None, hidden_dim=32, dilation=False, position_embedding='sine', dropout=0.1, nheads=4, dim_feedforward=64, enc_layers=3, dec_layers=3, pre_norm=False, num_queries=100, aux_loss=True, set_cost_class=1, set_cost_bbox=5, set_cost_giou=2, giou_loss_coef=2, bbox_loss_coef=5, eos_coef=0.1)
===============================================================================================================================

Best model has been saved to: [lesion_detection_CL_Fix2Layers_map_50_0_0011_mar_100_0_0144_epoch30_10-17-2023 12-01-35]
The final model has been saved to: [lesion_detection_CL_Fix2Layers_map_50_0_0004_mar_100_0_0122_epoch50_10-17-2023 12-46-12]

===============================================================================================================================
{'map': 1.8018314221990295e-05, 'map_50': 8.349610288860276e-05, 'map_75': 5.908955699851504e-06, 'map_small': 1.8018314221990295e-05, 'map_medium': -1.0, 'map_large': -1.0, 'mar_1': 0.006549410987645388, 'mar_10': 0.008864641189575195, 'mar_100': 0.012798367999494076, 'mar_small': 0.012798367999494076, 'mar_medium': -1.0, 'mar_large': -1.0, 'map_per_class': -1.0, 'mar_100_per_class': -1.0}