# SanText

Code for Findings of ACL-IJCNLP 2021 **"[Differential Privacy for Text Analytics via Natural Text Sanitization](https://arxiv.org/pdf/2106.01221.pdf)"**

```bib
@inproceedings{ACL21/YueDu21,
  author    = {Xiang Yue and Minxin Du and Tianhao Wang and Yaliang Li and Huan Sun and Sherman S. M. Chow},
  title     = {Differential Privacy for Text Analytics via Natural Text Sanitization},
  booktitle = {Findings, {ACL-IJCNLP} 2021},
  year      = {2021},
  }
```

## Setup Environment

### Install required packages

```shell
cd SanText
pip install -r requirements.txt
```

## Run

### SST GloVe SANTEXT

```
python run_SanText.py --task SST-2 --method SanText --epsilon 3.0 --word_embedding_path ./data/glove.840B.300d.txt --word_embedding_size 300 --data_dir ./data/SST-2/ --output_dir ./output_SanText_glove/SST-2/ --threads 8
```

```
python run_glue.py --model_name_or_path ./base_bert_models --task_name sst-2 --do_train --do_eval --data_dir ./output_SanText_glove/SST-2/eps_3.00/ --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./tmp/sst2-sanitize/ --overwrite_output_dir --overwrite_cache --save_steps 2000
```



### QNLI GloVe SANTEXT+

```
python run_SanText.py --task QNLI --method SanText_plus --epsilon 1.0 --word_embedding_path ./data/glove.840B.300d.txt --word_embedding_size 300 --data_dir ./data/QNLI/ --output_dir ./output_SanText_plus_glove/QNLI/ --threads 12 --p 0.3 --sensitive_word_percentage 0.9
```

```
python run_glue.py --model_name_or_path ./base_bert_models --task_name qnli --do_train --do_eval --data_dir ./output_SanText_plus_glove/QNLI/eps_1.00/sword_0.90_p_0.30 --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate 2e-5 --num_train_epochs 2.0 --output_dir ./tmp/qnli-sanitize/ --overwrite_output_dir --overwrite_cache --save_steps 2000
```



### FakeNews GloVe SANTEXT

```
python run_SanText.py --task FakeNews --method SanText --epsilon 3.0 --word_embedding_path ./data/glove.840B.300d.txt --word_embedding_size 300 --data_dir ./data/FakeNews/ --output_dir ./output_SanText_glove/FakeNews/ --threads 8
```

```
python run_glue.py --model_name_or_path ./base_bert_models --task_name FakeNews --do_train --do_eval --data_dir ./output_SanText_glove/FakeNews/eps_3.00/ --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./tmp/fakenews-sanitize/ --overwrite_output_dir --overwrite_cache --save_steps 2000
```



### FakeNews BERT SANTEXT+

```
python run_SanText.py --task FakeNews --method SanText_plus --epsilon 3.0 --embedding_type bert --data_dir ./data/FakeNews/ --output_dir ./output_SanText_plus_bert/FakeNews/ --threads 8 --p 0.3 --sensitive_word_percentage 0.9
```

```
python run_glue.py --model_name_or_path ./base_bert_models --task_name FakeNews --do_train --do_eval --data_dir ./output_SanText_plus_bert/FakeNews/eps_3.00/sword_0.90_p_0.30 --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./tmp/fakenews-bert-sanitize/ --overwrite_output_dir --overwrite_cache --save_steps 2000
```

*数据和运行结果在release中*
