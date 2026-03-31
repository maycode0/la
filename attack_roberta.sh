echo "roberta"
echo "roberta-imdb"
python LimeAttack_classification.py \
    --dataset_path "data/imdb" \
    --target_model "roberta"  \
    --model_path "/root/autodl-fs/modelHub/roberta-base-imdb" \
    --nclasses 2 \
    --word_embeddings_path "./data/embedding/glove.6B.200d.txt" \
    --counter_fitting_embeddings_path "./data/embedding/counter-fitted-vectors.txt" \
    --query_budget 100 \
    --k 10 \
    --syn_num 50


echo "roberta-yelp"
python LimeAttack_classification.py \
    --dataset_path "data/yelp" \
    --target_model "roberta"  \
    --model_path "/root/autodl-fs/modelHub/roberta-base-yelp" \
    --nclasses 2 \
    --word_embeddings_path "./data/embedding/glove.6B.200d.txt" \
    --counter_fitting_embeddings_path "./data/embedding/counter-fitted-vectors.txt" \
    --query_budget 100 \
    --k 10 \
    --syn_num 50

echo "roberta-sst2"
python LimeAttack_classification.py \
    --dataset_path "data/sst2" \
    --target_model "roberta"  \
    --model_path "/root/autodl-fs/modelHub/roberta-base-sst2" \
    --nclasses 2 \
    --word_embeddings_path "./data/embedding/glove.6B.200d.txt" \
    --counter_fitting_embeddings_path "./data/embedding/counter-fitted-vectors.txt" \
    --query_budget 100 \
    --k 10 \
    --syn_num 50

