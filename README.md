## Setup

```
# Get the E2ENLG dataset (from the link below), and put it under data/E2ENLG/
$ mkdir -p data/E2ENLG/
# take conda for example, create a new environment
$ conda create -n [your_env_name] python=3
$ source activate [your_env_name]
$ conda install pytorch torchvision -c pytorch
$ conda install spacy nltk
# download the spaCy models
$ python -m spacy download en
```

## Usage

We trained and tested two models (the best model of the baseline paper, and a seq2seq baseline) and use them as baseline. The following commands will run the models with coverage type specifiec with argument --type_coverage. Based on the 3 coverage models implemented, the argument can yake following values - 
['att_reg', 'att_see', 'att_reg_multi']


# For running the coverage model with curriculum learning based linguistic hierarchical model Baseline 1 hdecoder - 
## Training
python train.py --data_dir data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --use_embedding 0 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --partition_ratio 0.95 --cell GRU --n_layers 4 --n_en_layers 1 --n_de_layers 1 --en_hidden_size 200 --de_hidden_size 100 --en_embedding 0 --en_use_attr_init_state 1 --share_embedding 0 --embedding_dim 50 --en_embedding_dim 50 --de_embedding_dim 50 --attn_method none --bidirectional 1 --feed_last 1 --repeat_input 1 --batch_norm 0 --epochs 20 --batch_size 32 --en_optimizer Adam --de_optimizer Adam --en_learning_rate 0.001 --de_learning_rate 0.001 --split_teacher_forcing 1 --teacher_forcing_ratio 0.5 --inner_teacher_forcing_ratio 0.5 --inter_teacher_forcing_ratio 0.5 --tf_decay_rate 0.9 --inner_tf_decay_rate 0.9 --inter_tf_decay_rate 0.9 --schedule_sampling 0 --inner_schedule_sampling 1 --inter_schedule_sampling 1 --is_curriculum 1 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --finetune_embedding 0 --verbose_level 1 --verbose_epochs 0 --verbose_batches 500 --valid_epochs 1 --valid_batches 20 --save_epochs 1 --is_load 0 --check_mem_usage_batches 0 --attn_method concat --dir_name hd_cur_repeat_3_attreg2 --type_coverage att_reg


## Testing
python test.py --data_dir data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --use_embedding 0 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --partition_ratio 0.95 --cell GRU --n_layers 4 --n_en_layers 1 --n_de_layers 1 --en_hidden_size 200 --de_hidden_size 100 --en_embedding 0 --en_use_attr_init_state 1 --share_embedding 0 --embedding_dim 50 --en_embedding_dim 50 --de_embedding_dim 50 --attn_method none --bidirectional 1 --feed_last 1 --repeat_input 1 --batch_norm 0 --epochs 20 --batch_size 32 --en_optimizer Adam --de_optimizer Adam --en_learning_rate 0.001 --de_learning_rate 0.001 --split_teacher_forcing 1 --teacher_forcing_ratio 0.5 --inner_teacher_forcing_ratio 0.5 --inter_teacher_forcing_ratio 0.5 --tf_decay_rate 0.9 --inner_tf_decay_rate 0.9 --inter_tf_decay_rate 0.9 --schedule_sampling 0 --inner_schedule_sampling 1 --inter_schedule_sampling 1 --is_curriculum 1 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --finetune_embedding 0 --verbose_level 1 --verbose_epochs 0 --verbose_batches 500 --valid_epochs 1 --valid_batches 20 --save_epochs 1 --is_load 0 --check_mem_usage_batches 0 --attn_method concat  --is_load 1 --dir_name hd_cur_repeat_3_attreg2


# For running the coverage model with seq2seq model (Baseline 2) -
## Training

python train.py --data_dir data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --use_embedding 0 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --partition_ratio 0.95 --cell GRU --n_layers 1 --n_en_layers 1 --n_de_layers 1 --en_hidden_size 200 --de_hidden_size 400 --en_embedding 0 --en_use_attr_init_state 1 --share_embedding 0 --embedding_dim 50 --en_embedding_dim 50 --de_embedding_dim 50 --attn_method none --bidirectional 1 --feed_last 1 --repeat_input 0 --batch_norm 0 --epochs 20 --batch_size 32 --en_optimizer Adam --de_optimizer Adam --en_learning_rate 0.001 --de_learning_rate 0.001 --split_teacher_forcing 1 --teacher_forcing_ratio 0.5 --inner_teacher_forcing_ratio 0.5 --inter_teacher_forcing_ratio 0.5 --tf_decay_rate 0.9 --inner_tf_decay_rate 0.9 --inter_tf_decay_rate 0.9 --schedule_sampling 0 --inner_schedule_sampling 1 --inter_schedule_sampling 1 --is_curriculum 0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --finetune_embedding 0 --verbose_level 1 --verbose_epochs 0 --verbose_batches 500 --valid_epochs 1 --valid_batches 20 --save_epochs 1 --is_load 0 --check_mem_usage_batches 0 --attn_method concat --h_attn 0 --dir_name seq2seq_1 --type_coverage att_see


## Testing
python test.py --data_dir data/ --dataset E2ENLG --fold_attr 1 --vocab_size 500 --use_embedding 0 --regen 0 --replace_model 0 --is_spacy 1 --is_lemma 1 --use_punct 0 --en_max_length -1 --de_max_length -1 --min_length 5 --partition_ratio 0.95 --cell GRU --n_layers 1 --n_en_layers 1 --n_de_layers 1 --en_hidden_size 200 --de_hidden_size 400 --en_embedding 0 --en_use_attr_init_state 1 --share_embedding 0 --embedding_dim 50 --en_embedding_dim 50 --de_embedding_dim 50 --attn_method none --bidirectional 1 --feed_last 1 --repeat_input 0 --batch_norm 0 --epochs 20 --batch_size 32 --en_optimizer Adam --de_optimizer Adam --en_learning_rate 0.001 --de_learning_rate 0.001 --split_teacher_forcing 1 --teacher_forcing_ratio 0.5 --inner_teacher_forcing_ratio 0.5 --inter_teacher_forcing_ratio 0.5 --tf_decay_rate 0.9 --inner_tf_decay_rate 0.9 --inter_tf_decay_rate 0.9 --schedule_sampling 0 --inner_schedule_sampling 1 --inter_schedule_sampling 1 --is_curriculum 0 --padding_loss 0.0 --eos_loss 1.0 --max_norm 0.25 --finetune_embedding 0 --verbose_level 1 --verbose_epochs 0 --verbose_batches 500 --valid_epochs 1 --valid_batches 20 --save_epochs 1 --is_load 0 --check_mem_usage_batches 0 --attn_method concat --h_attn 0 --dir_name seq2seq_1 --is_load 1



