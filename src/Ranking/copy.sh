#!/bin/bash

cp /home/trec7/lianxiaoying/Track_code/src/Ranking/run_classifier.py /home/trec7/lianxiaoying/bert

export BERT_BASE_DIR=/home/trec7/lianxiaoying/model/uncased_L-12_H-768_A-12
export DATA_DIR=/home/trec7/lianxiaoying/Track_code/src/outputs

python /home/trec7/lianxiaoying/bert/run_classifier.py \
  --task_name=BertCls \
  --do_train=true \
  --do_eval=true \
  --do_test=true \
  --data_dir=$DATA_DIR_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=home/trec7/lianxiaoying/trained_model