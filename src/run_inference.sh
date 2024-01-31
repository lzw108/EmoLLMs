#export CUDA_VISIBLE_DEVICES='2'
export ABS_PATH="local_path"

model_name_or_path=lzw1008/Emollama-chat-7b # checkpoint

infer_file=$ABS_PATH/data/test.json
predict_file=$ABS_PATH/predicts/predict.json
# inference
python src/inference.py \
    --model_name_or_path $model_name_or_path \
    --infer_file $infer_file \
    --predict_file $predict_file \
    --batch_size 16 \
    --seed 123
    #--llama \

