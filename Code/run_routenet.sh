#!/bin/bash
if [[ "$1" = "tfrecords" ]]; then

    python3 routenet_with_link_cap.py data -d /home/datasets/SIGCOM/$2/

fi

if [[ "$1" = "train" ]]; then

    python3 routenet_with_link_cap.py train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=16,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8"  --train  /home/datasets/SIGCOM/$2/tfrecords/train/*.tfrecords --train_steps $3 --eval_ /home/datasets/SIGCOM/$2/tfrecords/evaluate/*.tfrecords --model_dir ./CheckPoints/$2

fi

if [[ "$1" = "train_multiple" ]]; then

    python3 routenet_with_link_cap.py train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=16,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8"  --train  /home/datasets/SIGCOM/$2/tfrecords/train/*.tfrecords /home/datasets/SIGCOM/$3/tfrecords/train/*.tfrecords --train_steps $5 --eval_ /home/datasets/SIGCOM/geant2bw/tfrecords/evaluate/*.tfrecords /home/datasets/SIGCOM/geant2bw/tfrecords/train/*.tfrecords --shuffle_buf 30000 --model_dir ./CheckPoints/$4
fi

if [[ "$1" = "predict" ]]; then

    python3 routenet_with_link_cap.py predict --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=16,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8" --model_dir=./CheckPoints/$2 --eval_steps=$3 --predict /home/datasets/SIGCOM/$4/tfrecords/evaluate/*.tfrecords
fi

if [[ "$1" = "normalize" ]]; then

    python3 normalize.py --dir /home/datasets/SIGCOM/nsfnetbw/tfrecords/train/ /home/datasets/SIGCOM/nsfnetbw/tfrecords/evaluate/ /home/datasets/SIGCOM/synth50bw2/tfrecords/evaluate/ /home/datasets/SIGCOM/synth50bw2/tfrecords/train/ --ini configNSFNET50.ini
fi
