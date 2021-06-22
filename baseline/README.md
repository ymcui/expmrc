# Baselines 

This folder contains the baseline codes written in TensorFlow. 

Hopefully, we will also provide PyTorch baseline codes in the near future.

As there is no training data available for ExpMRC, we use pseudo-training data for training, where the evidences are automatically extracted. Please check the folder `pseudo-training-data` (in GitHub root directory) to get these data.

## Requirements
```
tensorflow 1.5.3
```

## Training Scripts for SQuAD and CMRC 2018

Here is the training script for SQuAD.

```bash
python run_se_mrc.py \
--vocab_file=${PLM_PATH}/vocab.txt \
--bert_config_file=${PLM_PATH}/bert_config.json \
--init_checkpoint=${PLM_PATH}/bert_model.ckpt \
--task_name="squad" \
--do_train=True \
--do_predict=True \
--train_file=${DATA_PATH}/train-pseudo-squad.json \
--predict_file=${DATA_PATH}/expmrc-squad-dev.json \
--train_batch_size=32 \
--predict_batch_size=32 \
--num_train_epochs=2 \
--max_seq_length=512 \
--max_answer_length=40 \
--doc_stride=128 \
--learning_rate=3e-5 \
--loss_lambda=${LAMBDA} \
--rand_seed=12345 \
--save_checkpoints_steps=1000 \
--do_lower_case=False \
--output_dir=${MODEL_PATH} \
--use_tpu=False
```

Here is the training script for CMRC 2018.

```bash
python run_se_mrc.py \
--vocab_file=${PLM_PATH}/vocab.txt \
--bert_config_file=${PLM_PATH}/bert_config.json \
--init_checkpoint=${PLM_PATH}/bert_model.ckpt \
--task_name="cmrc2018" \
--do_train=True \
--do_predict=True \
--train_file=${DATA_PATH}/train-pseudo-cmrc2018.json \
--predict_file=${DATA_PATH}/expmrc-cmrc2018-dev.json \
--train_batch_size=32 \
--predict_batch_size=32 \
--num_train_epochs=2 \
--max_seq_length=512 \
--max_answer_length=40 \
--doc_stride=128 \
--learning_rate=3e-5 \
--rand_seed=12345 \
--save_checkpoints_steps=1000 \
--do_lower_case=True \
--output_dir=${MODEL_PATH} \
--use_tpu=False
```

- `PLM_PATH`: path to the pre-trained language model. 
- `DATA_PATH`: path to the data.
- `MODEL_PATH`: path to the output models.
- `LAMBDA`: loss weight for the evidence, 0.1 as default.
- **`do_lower_case` should match your PLM.**
- If you are using TPUs, please specify `use_tpu` as `True`, and also specify your `tpu_name` and `tpu_zone`.

## Training Scripts for RACE<sup>+</sup> and C<sup>3</sup>

Here is the training script for RACE<sup>+</sup>.

```bash
python run_mc_mrc.py \
--vocab_file=${PLM_PATH}/vocab.txt \
--bert_config_file=${PLM_PATH}/bert_config.json \
--init_checkpoint=${PLM_PATH}/bert_model.ckpt \
--task_name="race" \
--do_train=True \
--do_predict=True \
--train_file=${DATA_PATH}/train-pseudo-race.json \
--predict_file=${DATA_PATH}/expmrc-race-dev.json \
--train_tfrecord=${DATA_PATH}/train.race.tfrecord \
--predict_tfrecord=${DATA_PATH}/dev.race.tfrecord \
--train_batch_size=32 \
--predict_batch_size=32 \
--num_train_epochs=2 \
--max_seq_length=512 \
--doc_stride=128 \
--learning_rate=3e-5 \
--loss_lambda=${LAMBDA} \
--rand_seed=12345 \
--save_checkpoints_steps=1000 \
--do_lower_case=False \
--output_dir=${MODEL_PATH} \
--use_tpu=False
```

Here is the training script for C<sup>3</sup>.

```bash
python run_mc_mrc.py \
--vocab_file=${PLM_PATH}/vocab.txt \
--bert_config_file=${PLM_PATH}/bert_config.json \
--init_checkpoint=${PLM_PATH}/bert_model.ckpt \
--task_name="c3" \
--do_train=True \
--do_predict=True \
--train_file=${DATA_PATH}/train-pseudo-c3.json \
--predict_file=${DATA_PATH}/expmrc-c3-dev.json \
--train_tfrecord=${DATA_PATH}/train.c3.tfrecord \
--predict_tfrecord=${DATA_PATH}/dev.c3.tfrecord \
--train_batch_size=32 \
--predict_batch_size=32 \
--num_train_epochs=2 \
--max_seq_length=512 \
--doc_stride=128 \
--learning_rate=3e-5 \
--loss_lambda=${LAMBDA} \
--rand_seed=12345 \
--save_checkpoints_steps=1000 \
--do_lower_case=True \
--output_dir=${MODEL_PATH} \
--use_tpu=False
```

- `PLM_PATH`: path to the pre-trained language model. 
- `DATA_PATH`: path to the data. `tf_record` will be saved in `train_tfrecord` and `dev_tfrecord`.
- `MODEL_PATH`: path to the output models.
- `LAMBDA`: loss weight for the evidence, 0.1 as default.
- **`do_lower_case` should match with your PLM.**
- If you are using TPUs, please specify `use_tpu` as `True`, and also specify your `tpu_name` and `tpu_zone`.

## Evaluation

After training, there will be a filed called `dev_predictions.json` in your `MODEL_PATH` folder. This file contains the prediction results of `predict_file`.

Now we use official evaluation script `eval_expmrc.py` to get the evaluation results. 

```bash
python eval_expmrc.py expmrc-${task}-dev.json dev_predictions.json
```
`${task}` can be one of the followings: `squad`, `cmrc2018`, `race`, `c3`.