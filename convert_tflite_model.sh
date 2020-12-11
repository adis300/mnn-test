MODEL_NAME=$1
#./mnn-mac/MNNConvert -f TFLITE -modelFile model/${MODEL_NAME}.tflite --MNNModel model/${MODEL_NAME}.mnn --bizCode biz
tflite2onnx model/${MODEL_NAME}.tflite model/${MODEL_NAME}.onnx

./convert_onnx_model.sh ${MODEL_NAME}