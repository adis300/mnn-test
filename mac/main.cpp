
#include <MNN/expr/Executor.hpp>
#include <MNN/AutoTime.hpp> // Timer

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
    MNNForwardType forwardType = MNN_FORWARD_CPU;// MNN_FORWARD_METAL; // MNN_FORWARD_CPU; Use metal for iOS and Mac

    exe->setGlobalExecutorConfig(forwardType, config, NUM_OF_THREADS);

    // Loading a model
    std::map<std::string, MNN::Express::VARP> model = MNN::Express::Variable::loadMap(modelFileName);
    for(std::map<std::string, MNN::Express::VARP>::iterator iter = model.begin(); iter != model.end(); ++iter) std::cout<<"Model key:"<<iter->first<<std::endl;
    // static std::map<std::string, VARP> loadMap(const uint8_t* buffer, size_t length);
    auto inputOutput = MNN::Express::Variable::getInputAndOutput(model);

    std::map<std::string, MNN::Express::VARP> inputs = inputOutput.first;
    std::map<std::string, MNN::Express::VARP> outputs = inputOutput.second;

    std::cout<<"Input key:"<<inputs.begin()->first<<std::endl;
    for(std::map<std::string, MNN::Express::VARP>::iterator iter = outputs.begin(); iter != outputs.end(); ++iter) std::cout<<"Output key:"<<iter->first<<std::endl;

    MNN::Express::VARP input = outputs.begin()->second;
    MNN::Express::VARP output = outputs.begin()->second;

    const MNN::Express::Variable::Info* input_info = input->getInfo();
    const MNN::Express::Variable::Info* output_info = output->getInfo();

    if (nullptr == input_info || nullptr == output_info) {
        MNN_ERROR("Unable to get input or output info\n");
        return 0;
    }
    // std::vector<int> shape = input_info->dim;
    for(auto dim : input_info->dim) std::cout << "Input shape:" << dim << std::endl;
    for(auto dim : output_info->dim) std::cout << "Output shape:" << dim << std::endl;

    AUTOTIME;

    
    // Populating a model

    // Get model results

}