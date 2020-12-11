MODEL_NAME=$1
./mnn-mac/MNNConvert -f ONNX --modelFile model/${MODEL_NAME}.onnx --MNNModel model/${MODEL_NAME}.mnn --bizCode biz