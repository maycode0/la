echo "gpt2"

echo "gpt2-imdb"
python LimeAttack_classification.py \
    --dataset_path "data/imdb" \
    --target_model "gpt2"  \
    --model_path "/root/autodl-fs/modelHub/gpt2-finetune-imdb" \
    --nclasses 2 \
    --word_embeddings_path "./data/embedding/glove.6B.200d.txt" \
    --counter_fitting_embeddings_path "./data/embedding/counter-fitted-vectors.txt" \
    --query_budget 100 \
    --k 10 \
    --syn_num 50

echo "gpt2-yelp"
python LimeAttack_classification.py \
    --dataset_path "data/yelp" \
    --target_model "gpt2"  \
    --model_path "/root/autodl-fs/modelHub/gpt2-finetune-yelp" \
    --nclasses 2 \
    --word_embeddings_path "./data/embedding/glove.6B.200d.txt" \
    --counter_fitting_embeddings_path "./data/embedding/counter-fitted-vectors.txt" \
    --query_budget 100 \
    --k 10 \
    --syn_num 50
echo "gpt2-sst2"
python LimeAttack_classification.py \
    --dataset_path "data/sst2" \
    --target_model "gpt2"  \
    --model_path "/root/autodl-fs/modelHub/gpt2-finetune-sst2" \
    --nclasses 2 \
    --word_embeddings_path "./data/embedding/glove.6B.200d.txt" \
    --counter_fitting_embeddings_path "./data/embedding/counter-fitted-vectors.txt" \
    --query_budget 100 \
    --k 10 \
    --syn_num 50