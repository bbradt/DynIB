result_args="--dataset emnist --model mlp --logdir results/initial_emnist_mlp --model_args [dataset[0][0].shape,num_classes] --model_kwargs {'hidden_layers':[512,256,128,64,32,16]} --estimator npeet --estimator_args [] --estimator_kwargs {} --seed 0"
for ((k=0; k<5; k++))
do
for ((epoch=1; epoch<=100; epoch++))
do 
myargs=$result_args" --epoch "$epoch" --k "$k 
sbatch -J "EM-IB-k"$k"-e"$epoch experiments/inforunner.sh $myargs
done
done