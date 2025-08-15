IN_PATH=/home/jruffle/Documents/seq-synth/data/sequences_merged
OUT_PATH=/home/jruffle/Documents/nnUNet/nnUNet_raw/Dataset001_enhance
TRAIN_TXT_PATH=/home/jruffle/Desktop/ENHANCEMENT_ARTICLE/figures/train_filenames.txt
VAL_TXT_PATH=/home/jruffle/Desktop/ENHANCEMENT_ARTICLE/figures/validation_filenames.txt

rm -r $OUT_PATH
mkdir -p $OUT_PATH/imagesTr
mkdir -p $OUT_PATH/imagesTs
mkdir -p $OUT_PATH/labelsTr
mkdir -p $OUT_PATH/labelsTs

cat $TRAIN_TXT_PATH | parallel --progress --jobs 128 fslsplit $IN_PATH/{}.nii.gz $OUT_PATH/imagesTr/{}_ -t
cat $TRAIN_TXT_PATH | parallel --progress --jobs 128 mv $OUT_PATH/imagesTr/{}_0003.nii.gz $OUT_PATH/imagesTr/{}_0002.nii.gz
cat $TRAIN_TXT_PATH | parallel --progress --jobs 128 cp $IN_PATH/../enhancement_masks/{}.nii.gz $OUT_PATH/labelsTr/{}.nii.gz

cat $VAL_TXT_PATH | parallel --progress --jobs 128 fslsplit $IN_PATH/{}.nii.gz $OUT_PATH/imagesTs/{}_ -t
cat $VAL_TXT_PATH | parallel --progress --jobs 128 mv $OUT_PATH/imagesTs/{}_0003.nii.gz $OUT_PATH/imagesTs/{}_0002.nii.gz
cat $VAL_TXT_PATH | parallel --progress --jobs 128 cp $IN_PATH/../enhancement_masks/{}.nii.gz $OUT_PATH/labelsTs/{}.nii.gz