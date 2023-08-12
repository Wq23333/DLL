We provide code for DLL, and code for train and eval.

For DLL:
- dll\dll.py: The method of training including the defination of KDL_loss.
- dll\model.py: The defination of model including the defination of PDL_loss.

For train:  
1. VidVRD-II:
- utils\feature.py: The method of loading features.
- utils\misc.py: The defination of BCE loss.
- utils\model.py: The baseline model of our backbone.
- utils\Traindataset.py: The defination of training data loader.
- utils\common: The definitions of some basic modules.

2. VRD-STGC:
- utils\utils.py: Tools.
- utils\datasets.py: The defination of dataset.
- utils\gen_final_res.py: The method of generating final prediction results.


For eval:
- eval\visual_relation_detection.py: Calculate mR@K, R@K, P@K, mAP metrics.

For data:
- data\category.txt: Subject/object categories.
- data\relation.txt: Final predicate labels.
- data\relation_actional.txt: Actional Pattern labels.
- data\relation_spatial.txt: Spatial Pattern labels.
- data\sum.pth: Count of predicates on VidVRD dataset.
