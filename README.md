# Project2_FoundationalAI
second project of foundational AI

How to run the project:
Run the following codes in the terminal. 



# For inference
**************For RNN *******************
python train_language_model_updated.py \
    --inference_only \
    --model_path ./rnn_models/rnn_best.pt \
    --tokenizer_path gutenberg_bpe.model \
    --prompts "She doubted the sincerity" \
    --temperature 0.6 \
    --gen_max_length 50 \
    --log_dir ./inference_logs

# For inference
# for LSTM

python train_language_model_updated.py \
    --inference_only \
    --model_path ./lstm_models/lstm_best.pt \
    --tokenizer_path gutenberg_bpe.model \
    --prompts "She doubted the sincerity" \
    --temperature 0.3 \
    --gen_max_length 50 \
    --log_dir ./inference_logs
************For transformers ******************
python train_language_model_updated.py \
    --inference_only \
    --model_path ./transformer_models/transformer_best.pt \
    --tokenizer_path gutenberg_bpe.model \
    --prompts "She doubted the sincerity" \
    --temperature 0.0 \
    --gen_max_length 50 \
    --log_dir ./inference_logs
