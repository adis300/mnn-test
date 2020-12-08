
#include <MNN/expr/Executor.hpp>

#include <iostream>

#define NUM_OF_THREADS 4
int main(int argc, const char* argv[]) {

    if (argc < 2) {
        MNN_ERROR("Please specify a model path\n");
        MNN_ERROR("USAGE:./main.a [model_path]\n");
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
    // static std::map<std::string, VARP> loadMap(const uint8_t* buffer, size_t length);
    auto inputOutput = MNN::Express::Variable::getInputAndOutput(model);

    std::map<std::string, MNN::Express::VARP> inputs = inputOutput.first;
    std::map<std::string, MNN::Express::VARP> outputs = inputOutput.second;
    for(std::map<std::string, MNN::Express::VARP>::iterator iter = inputs.begin(); iter != inputs.end(); ++iter)
    {
        std::string k =  iter->first;
        std::cout<<"Key:"<<k<<std::endl;
    }

    std::cout<<inputs.begin()->first<<std::endl;
    // Populating a model

    // Get model results

}