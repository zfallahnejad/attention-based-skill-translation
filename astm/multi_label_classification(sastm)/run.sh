# Java
## SASTM
### BEST
python3 train.py --dataset java --data_path ../data/so_java.txt --post_tags_path ../data/so_java_question_tags.pkl --post_score_path ../data/so_java_post_score.pkl --dataset_percent 1.0 --test_size 0.2 --post_type question --num_words 5000  --max_sentence_len 256 --epochs 50 --batch_size 64 --word_vector_dim 100 --word_index_dir . --lstm_dim 64 --dropout_value 0.15 --result_path java_sastm_best --model_name SASTM --class_weight

# PHP
## SASTM
### BEST
python3 train.py --dataset php --data_path ../data/so_php.txt --post_tags_path ../data/so_php_question_tags.pkl --post_score_path ../data/so_php_post_score.pkl --dataset_percent 1.0 --test_size 0.2 --post_type question --num_words 5000  --max_sentence_len 256 --epochs 50 --batch_size 64 --word_vector_dim 50  --word_index_dir . --lstm_dim 128 --dropout_value 0.15 --result_path php_sastm_best  --model_name SASTM --class_weight
