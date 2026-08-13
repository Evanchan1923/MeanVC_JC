export PYTHONPATH=$PYTHONPATH:$PWD


ckpt_dir=/srv/scratch/speechdata/SAPC_Team
model_path=${ckpt_dir}/model_200ms.safetensors
vocoder_path=${ckpt_dir}/vocos.pt
source_path=path/to/source_dir
reference_path=path/to/reference.wav
output_dir=src/outputs

mkdir -p $output_dir


python3 src/infer/infer_ref.py \
    --model-config src/config/config_200ms.json \
    --ckpt-path ${model_path} \
    --vocoder-ckpt-path ${vocoder_path} \
    --source-path ${source_path} \
    --reference-path ${reference_path} \
    --output-dir ${output_dir} \
    --chunk-size 20 \
    --steps 2
