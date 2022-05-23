export PYTHONPATH=`pwd`

LOGDIR=logs/`date +%Y%m%d`
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi

# Reddit
bash run_shell/R1_TegTok_reddit.sh 43     > ${LOGDIR}/`date +%Y%m%d%H`_S7_S43.log 2>&1 &  
bash run_shell/R1_TegTok_reddit.sh 13     > ${LOGDIR}/`date +%Y%m%d%H`_S7_S13.log 2>&1 &  
bash run_shell/R1_TegTok_reddit.sh 91     > ${LOGDIR}/`date +%Y%m%d%H`_S7_S91.log 2>&1 &  
bash run_shell/R1_TegTok_reddit.sh 7677   > ${LOGDIR}/`date +%Y%m%d%H`_S7_S7677.log 2>&1 &  

# NQG
bash run_shell/S1_TegTok_nqg.sh 43     > ${LOGDIR}/`date +%Y%m%d%H`_S1_43.log 2>&1 &  
bash run_shell/S1_TegTok_nqg.sh 13     > ${LOGDIR}/`date +%Y%m%d%H`_S1_13.log 2>&1 &  
bash run_shell/S1_TegTok_nqg.sh 91     > ${LOGDIR}/`date +%Y%m%d%H`_S1_91.log 2>&1 &  
bash run_shell/S1_TegTok_nqg.sh 7677   > ${LOGDIR}/`date +%Y%m%d%H`_S1_7677.log 2>&1 &  