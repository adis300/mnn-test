#MODEL_NAME="model-mobilenet_v1_075"
MODEL_NAME=$1
./mnn-mac/MNNConvert -f TF --modelFile model/${MODEL_NAME}.pb --MNNModel model/${MODEL_NAME}.mnn --bizCode biz