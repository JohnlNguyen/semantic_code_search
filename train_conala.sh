DATA_DIR='/tf/t2t_data/'
TRAIN_DIR='/tf/t2t_train/intent_to_code/conala/'
TMP_DIR='/tf/datagen/'
PROBLEM='semantic_search'
USR_DIR='/tf/usr_dir'
HPARAMS=transformer_base_single_gpu
MODEL=transformer

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
  
  
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=semantic_search \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$USR_DIR \
  --hparams="num_hidden_layers=6"