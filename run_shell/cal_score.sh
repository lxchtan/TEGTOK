export PYTHONPATH=`pwd`

#embeddings bin file: recommend to download from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
EMB=data/GoogleNews-vectors-negative300.bin

calculation(){
  CUDA_VISIBLE_DEVICES=$GPUID python analyse/score.py --embeddings=${EMB} --result_file=${RES} --scorefile=${SCORE}
}

SCORE=results/Paper/scores.json
GPUID=1
RES=results/Paper/TegTok_reddit.json
calculation 
