DATA_DIR='/tf/t2t_data'
TRAIN_DIR='/tf/t2t_train/intent_to_code/conala/'
TMP_DIR='/tf/datagen'
PROBLEM='semantic_search'
USR_DIR='/tf/usr_dir'
HPARAMS=transformer_base_single_gpu
MODEL=transformer

BEAM_SIZE=4
ALPHA=0.6

DECODE_FILE=$TMP_DIR/test_intent.txt
DECODE_TO_FILE=$TMP_DIR/test_translated_code.txt 

t2t-decoder \
  --t2t_usr_dir='/tf/usr_dir' \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \ 
  --decode_to_file=$DECODE_TO_FILE
  
