M=6
hidden_size=256
feed_size=1024
head=8
batch_size=256
ngpu=8
lr=4e-4
matches=L6h256_roberta_hidden_mse
temperature=8
acc=8
length=512
train_steps=100000
save_steps=10000

NAME=M${M}_hs${hidden_size}_fs${feed_size}_h${head}_b${batch_size}_lr${lr}_t${temperature}_RoBERTa

TEACHER_DIR_PATH=./pretrained_model_path/RoBERTa
OUTPUT_DIR=./saves/${NAME}
json_files=./jsons/data.json

student_config_file=./distill_configs/MiniRBT-h256.json
mkdir -p $OUTPUT_DIR

python -m torch.distributed.launch --nproc_per_node=${ngpu} train.py \
    --teacher_name_or_path ${TEACHER_DIR_PATH} \
    --do_lower_case \
    --student_config ${student_config_file} \
    --matches ${matches} \
    --do_train \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --random_seed 1337 \
    --num_train_steps ${train_steps} \
    --ckpt_steps $save_steps \
    --learning_rate $lr \
    --official_schedule linear \
    --output_dir $OUTPUT_DIR \
    --data_files_json $json_files \
    --data_cache_dir  ./dataset \
    --preprocessing_num_workers 20 \
    --gradient_accumulation_steps $acc \
    --temperature ${temperature} \
    --output_encoded_layers true \
    --fp16 
