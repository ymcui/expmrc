# Baseline for Span-Extraction MRC

This folder contains the baseline codes for SQuAD and CMRC 2018, where both of them are span-extraction MRC tasks. The original baseline codes are written in TensorFlow. 

Hopefully, we will also provide PyTorch baseline codes in the near future.

As there is no training data available for ExpMRC, we use pseudo-training data for training, where the evidences are automatically extracted. Please check the folder `pseudo-training-data` (in the root GitHub directory) to get these data.

## Requirements
```
tensorflow 1.5.3
```

## Pseudo Training Data

`run.squad.sh` contains the training script for SQuAD.

## Training SQuAD

`run.squad.sh` contains the training script for SQuAD.

```bash
python run_se_mrc.py \
--vocab_file=${PLM_PATH}/vocab.txt \
--bert_config_file=${PLM_PATH}/bert_config.json \
--init_checkpoint=${PLM_PATH}/bert_model.ckpt \
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

- `PLM_PATH`: path to the pre-trained language model. **`do_lower_case` should match your PLM.**
- `DATA_PATH`: path to the data.
- `MODEL_PATH`: path to the output models.
- `LAMBDA`: loss weight for the evidence, 0.1 as default.
- If you are using TPUs, please specify `use_tpu` as `True`, and also specify your `tpu_name` and `tpu_zone`.

## Training CMRC 2018

As there is no training data available for ExpMRC, we use pseudo-training data for training, where the evidences are automatically extracted. Please see our paper for more details.

`run.cmrc2018.sh` contains the training script for CMRC 2018.

```bash
python run_se_mrc.py \
--vocab_file=${PLM_PATH}/vocab.txt \
--bert_config_file=${PLM_PATH}/bert_config.json \
--init_checkpoint=${PLM_PATH}/bert_model.ckpt \
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

- `PLM_PATH`: path to the pre-trained language model. **`do_lower_case` should match your PLM.**
- `DATA_PATH`: path to the data.
- `MODEL_PATH`: path to the output models.
- `LAMBDA`: loss weight for the evidence, 0.1 as default.
- If you are using TPUs, please specify `use_tpu` as `True`, and also specify your `tpu_name` and `tpu_zone`.


## Evaluation

After training, there will be a filed called `dev_predictions.json` in your `MODEL_PATH` folder. This file contains the prediction results of `predict_file`.

Now we use official evaluation script `eval_expmrc.py` to get the evaluation results. 

SQuAD: 

```bash
python eval_expmrc.py expmrc-squad-dev.json dev_predictions.json
```
CMRC 2018: 

```bash
python eval_expmrc.py expmrc-cmrc2018-dev.json dev_predictions.json
```
