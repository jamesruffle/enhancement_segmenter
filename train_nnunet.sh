# nnUNetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncL -gpu_memory_target 48 -overwrite_plans_name nnUNetPlannerResEncXL -np 32
# nnUNetv2_plan_experiment -d 001 -pl nnUNetPlannerResEncL -gpu_memory_target 48 -overwrite_plans_name nnUNetPlannerResEncXL

# this one was used
# nnUNetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncXL -np 32 --verify_dataset_integrity

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 2d 0 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 2d 1 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 001 2d 2 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 2
wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 2d 3 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 2d 4 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 001 3d_fullres 0 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 2
wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 3d_fullres 1 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 3d_fullres 2 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 001 3d_fullres 3 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 2
wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 3d_fullres 4 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 3d_cascade_fullres 1 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 001 3d_cascade_fullres 2 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 2
wait

# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 3d_lowres 0 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 1
# CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 001 3d_lowres 1 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 2
# wait

# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 3d_lowres 2 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 0
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 3d_lowres 3 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 1
# CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 001 3d_lowres 4 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 2
# wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 3d_cascade_fullres 0 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 0
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 3d_cascade_fullres 3 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 3d_cascade_fullres 4 --npz -p nnUNetPlannerResEncXL --c & # train on GPU 1
wait

cd /media/jruffle/DATA/nnUNet/nnUNet_results/Dataset001_enhance/
cp -r nnUNetTrainer__nnUNetResEncUNetXLPlans__2d/ nnUNetTrainer__nnUNetPlannerResEncXL__2d/
cp -r nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/ nnUNetTrainer__nnUNetPlannerResEncXL__3d_fullres
nnUNetv2_find_best_configuration 001 -p nnUNetPlannerResEncXL -np 32

# ***All results:***
# nnUNetTrainer__nnUNetPlannerResEncXL__2d: 0.47406320567015536
# nnUNetTrainer__nnUNetPlannerResEncXL__3d_fullres: 0.521108200649641
# ensemble___nnUNetTrainer__nnUNetPlannerResEncXL__2d___nnUNetTrainer__nnUNetPlannerResEncXL__3d_fullres___0_1_2_3_4: 0.5180029610533374

# *Best*: nnUNetTrainer__nnUNetPlannerResEncXL__3d_fullres: 0.521108200649641

# ***Determining postprocessing for best model/ensemble***
# Removing all but the largest foreground region did not improve results!

# ***Run inference like this:***

# nnUNetv2_predict -d Dataset001_enhance -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlannerResEncXL

# ***Once inference is completed, run postprocessing like this:***

# nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file /media/jruffle/DATA/nnUNet/nnUNet_results/Dataset001_enhance/nnUNetTrainer__nnUNetPlannerResEncXL__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /media/jruffle/DATA/nnUNet/nnUNet_results/Dataset001_enhance/nnUNetTrainer__nnUNetPlannerResEncXL__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json
