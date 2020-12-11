MODEL_NAME=$1
python3 -m tf2onnx.convert --saved-model model/${MODEL_NAME} --output model/${MODEL_NAME}.onnx

./convert_onnx_model.sh ${MODEL_NAME}