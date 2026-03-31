python LimeAttack_classification.py ^
    --dataset_path "data/imdb" ^
    --target_model "gpt2"  ^
    --model_path "E:\modelHub\gpt2-finetune-imdb" ^
    --nclasses 2 ^
    --word_embeddings_path "data\embedding\glove.6B.200d.txt" ^
    --counter_fitting_embeddings_path "data\embedding\counter-fitted-vectors.txt" ^
    --query_budget 100 ^
    --k 10 ^
    --syn_num 50
