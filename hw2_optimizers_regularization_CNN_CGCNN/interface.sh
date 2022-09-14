#
PARTITION=account:gpu
# you can also use account:scholar if you want to use CPU
DEVICE=cuda
EPOCHS=100


#
minibatch=0
optimizer=0
l2=0
cnn=0
cgcnn=0
shuffle=0
amp=0
for arg in $*; do
    #
    if [[ ${arg} == "minibatch" ]]; then
        #
        minibatch=1
    elif [[ ${arg} == "optimizer" ]]; then
        #
        optimizer=1
    elif [[ ${arg} == "l2" ]]; then
        #
        l2=1
    elif [[ ${arg} == "cnn" ]]; then
        #
        cnn=1
    elif [[ ${arg} == "cgcnn" ]]; then
        #
        cgcnn=1
    elif [[ ${arg} == "shuffle" ]]; then
        #
        shuffle=1
    elif [[ ${arg} == "amp" ]]; then
        #
        amp=1
    else
        #
        echo "unknown interface ${arg}."
        exit 1
    fi
done


#
if [[ ${minibatch} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    for bsz in 100 500 3000 5000; do
        #
        for lr in 1e-3 1e-4 1e-5; do
            #
            python main.py --sbatch ${PARTITION} --batch-size ${bsz} \
                --lr ${lr} --device ${DEVICE} --num-epochs ${EPOCHS}
        done
    done
fi


#
if [[ ${optimizer} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    for optimalg in sgd momentum nesterov adam; do
        #
        python main.py --sbatch ${PARTITION} --batch-size 100 --lr 1e-3 \
            --optim-alg ${optimalg} --device ${DEVICE} --num-epochs ${EPOCHS}
    done
fi


#
if [[ ${l2} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    for lambda in 1 0.1 0.01; do
        #
        python main.py --sbatch ${PARTITION} --batch-size 100 --lr 1e-3 \
            --l2-lambda ${lambda} --device ${DEVICE} --num-epochs ${EPOCHS}
    done
fi


#
if [[ ${cnn} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    python main.py --sbatch ${PARTITION} --batch-size 100 --cnn --lr 1e-3 \
        --kernel 5 --stride 1 --device ${DEVICE} --num-epochs ${EPOCHS}
    python main.py --sbatch ${PARTITION} --batch-size 100 --cnn --lr 1e-3 \
        --kernel 3 --stride 3 --device ${DEVICE} --num-epochs ${EPOCHS}
    python main.py --sbatch ${PARTITION} --batch-size 100 --cnn --lr 1e-3 \
        --kernel 14 --stride 1 --device ${DEVICE} --num-epochs ${EPOCHS}
fi


#
if [[ ${cgcnn} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    python main.py --sbatch ${PARTITION} --batch-size 100 --cgcnn --lr 1e-3 \
        --kernel 5 --stride 1 --device ${DEVICE} --num-epochs ${EPOCHS}
fi


#
if [[ ${shuffle} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    python main.py --sbatch ${PARTITION} --batch-size 100 --cgcnn --lr 1e-3 \
        --shuffle-label --device ${DEVICE} --num-epochs ${EPOCHS}
fi


#
if [[ ${amp} -gt 0 ]]; then
    # YOU SHOULD FILL IN THIS FUNCTION
    python main.py --sbatch ${PARTITION} --batch-size 100 --cnn --lr 1e-3 \
        --amprec --device ${DEVICE} --num-epochs ${EPOCHS}
fi
