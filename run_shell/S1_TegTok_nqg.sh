PARAMS=config/squad_nqg/TegTok.json
OUTPUT_PRE=TegTok_nqg

if [[ $1 == "7677" ]]; then
  GPUID=3
elif [[ $1 == "43" ]]; then
  GPUID=2
elif [[ $1 == "13" ]]; then
  GPUID=0
elif [[ $1 == "91" ]]; then
  GPUID=1
fi
OUTPUT=${OUTPUT_PRE}_$1

if [[ $2 != "test" ]]; then
  CUDA_VISIBLE_DEVICES=$GPUID python -u trainer.py \
    --params_file ${PARAMS} --output_path $OUTPUT --n_epochs 15 \
    --train_batch_size 8 --valid_batch_size 8 --gradient_accumulation_steps 8 --seed $1
fi

if [[ $2 != "train" ]]; then
  CUDA_VISIBLE_DEVICES=$GPUID python generator.py --params_file ${PARAMS} --generate_config config/generation/generate.json \
    --model_checkpoint runs/$OUTPUT \
    --result_file `date +%Y%m%d`_${OUTPUT}_single_example.json --batch_size 16
fi
