
# this one was used
# nnUNetv2_plan_and_preprocess -d 003 -pl nnUNetPlannerResEncL -np 32 --verify_dataset_integrity
# nnUNetv2_plan_experiment -d 3 -pl nnUNetPlannerResEncL -gpu_memory_target 48 -overwrite_plans_name nnUNetResEncUNetPlans_48G
# nnUNetv2_plan_and_preprocess -d 003 -pl nnUNetPlannerResEncL -np 32

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0 --npz -p /media/jruffle/DATA/nnUNet/nnUNet_preprocessed/Dataset003_enhance_and_abnormality_batchconfig/nnUNetResEncUNetPlans_48G --c & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 003 3d_fullres 1 --npz -p /media/jruffle/DATA/nnUNet/nnUNet_preprocessed/Dataset003_enhance_and_abnormality_batchconfig/nnUNetResEncUNetPlans_48G --c & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 003 3d_fullres 2 --npz -p /media/jruffle/DATA/nnUNet/nnUNet_preprocessed/Dataset003_enhance_and_abnormality_batchconfig/nnUNetResEncUNetPlans_48G --c & # train on GPU 2
wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 3 --npz -p /media/jruffle/DATA/nnUNet/nnUNet_preprocessed/Dataset003_enhance_and_abnormality_batchconfig/nnUNetResEncUNetPlans_48G --c & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 003 3d_fullres 4 --npz -p /media/jruffle/DATA/nnUNet/nnUNet_preprocessed/Dataset003_enhance_and_abnormality_batchconfig/nnUNetResEncUNetPlans_48G --c & # train on GPU 1
wait


# cd /media/jruffle/DATA/nnUNet/nnUNet_results/Dataset003_enhance_and_abnormality/
# cp -r nnUNetTrainer__nnUNetResEncUNetLPlans__2d/ nnUNetTrainer__nnUNetPlannerResEncL__2d/
# cp -r nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/ nnUNetTrainer__nnUNetPlannerResEncL__3d_fullres
# nnUNetv2_find_best_configuration 003 -p nnUNetPlannerResEncL -np 32

# (nnunet-env) jruffle@ada:~/Documents/nnUNet$ nnUNetv2_find_best_configuration 003 -p nnUNetResEncUNetPlans_80G
# Configuration 2d not found in plans nnUNetResEncUNetPlans_80G.
# Inferred plans file: /media/jruffle/DATA/nnUNet/nnUNet_preprocessed/Dataset003_enhance_and_abnormality_batchconfig/nnUNetResEncUNetPlans_80G.json.
# Configuration 3d_lowres not found in plans nnUNetResEncUNetPlans_80G.
# Inferred plans file: /media/jruffle/DATA/nnUNet/nnUNet_preprocessed/Dataset003_enhance_and_abnormality_batchconfig/nnUNetResEncUNetPlans_80G.json.
# Configuration 3d_cascade_fullres not found in plans nnUNetResEncUNetPlans_80G.
# Inferred plans file: /media/jruffle/DATA/nnUNet/nnUNet_preprocessed/Dataset003_enhance_and_abnormality_batchconfig/nnUNetResEncUNetPlans_80G.json.

# ***All results:***
# nnUNetTrainer__nnUNetResEncUNetPlans_80G__3d_fullres: 0.7859277512207518

# *Best*: nnUNetTrainer__nnUNetResEncUNetPlans_80G__3d_fullres: 0.7859277512207518

# ***Determining postprocessing for best model/ensemble***
# Removing all but the largest foreground region did not improve results!









# Results were improved by removing all but the largest component for 1. Dice before: 0.9917 after: 0.99171
# Removing all but the largest component for 2 did not improve results! Dice before: 0.81763 after: 0.7922

# Removing all but the largest component for 3 did not improve results! Dice before: 0.54845 after: 0.53488

# ***Run inference like this:***

# nnUNetv2_predict -d Dataset003_enhance_and_abnormality_batchconfig -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetPlans_80G

# ***Once inference is completed, run postprocessing like this:***

# nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file /media/jruffle/DATA/nnUNet/nnUNet_results/Dataset003_enhance_and_abnormality_batchconfig/nnUNetTrainer__nnUNetResEncUNetPlans_80G__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /media/jruffle/DATA/nnUNet/nnUNet_results/Dataset003_enhance_and_abnormality_batchconfig/nnUNetTrainer__nnUNetResEncUNetPlans_80G__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json
