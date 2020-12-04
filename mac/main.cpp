
#include <MNN/expr/Executor.hpp>

#define NUM_OF_THREADS 4
int main(int argc, const char* argv[]) {

    if (argc < 2) {
        MNN_ERROR("./main.a [model_path]\n");
        return 0;
    }

    auto modelFileName = argv[1];
    FUNC_PRINT_ALL(modelFileName, s);

    auto exe = MNN::Express::Executor::getGlobalExecutor();
    MNN::BackendConfig config;
    config.precision = MNN::BackendConfig::Precision_Normal; //Precision_Low uses 16 bit operations
    MNNForwardType forwardType = MNN_FORWARD_METAL; // MNN_FORWARD_CPU; Use metal for iOS and Mac

    exe->setGlobalExecutorConfig(forwardType, config, NUM_OF_THREADS);

    // Loading a model
    auto model = MNN::Express::Variable::loadMap(modelFileName);
    auto inputOutput = MNN::Express::Variable::getInputAndOutput(model);

    // Populating a model

    // Get model results

}