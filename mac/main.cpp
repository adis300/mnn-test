#include <MNN/AutoTime.hpp> // Timer
#include <iostream>

#include <MNN/Interpreter.hpp>

/* express_demo
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp> // _Input
*/

#define NUM_OF_THREADS 4


#define OFFSET_NODE_NAME "offset"
#define DISPLACE_FWD_NODE_NAME "displacement_fwd"
#define DISPLACE_BWD_NODE_NAME "displacement_bwd"
#define HEATMAPS "heatmap"

/*
int express_demo(model_path){

    auto exe = MNN::Express::Executor::getGlobalExecutor();
    MNN::BackendConfig config;
    config.precision = MNN::BackendConfig::Precision_Normal; //Precision_Low uses 16 bit operations
    MNNForwardType forwardType = MNN_FORWARD_CPU;// MNN_FORWARD_METAL; // MNN_FORWARD_CPU; Use metal for iOS and Mac

    exe->setGlobalExecutorConfig(forwardType, config, NUM_OF_THREADS);

    // Loading a model
    std::map<std::string, MNN::Express::VARP> model = MNN::Express::Variable::loadMap(model_path);
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
}
*/
int main(int argc, const char* argv[]) {

    if (argc < 2) {
        MNN_ERROR("Please specify a model path\n");
        MNN_ERROR("USAGE:./main.a [model_path]\n");
        return 0;
    }

    auto model_path = argv[1];
    FUNC_PRINT_ALL(model_path, s);

    auto interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path));

    MNN::ScheduleConfig config;
    config.type      = MNN_FORWARD_AUTO; // Use GPU and fallback to CPU
    config.numThread = NUM_OF_THREADS;

    MNN::Session* session  = interpreter -> createSession(config);

    MNN::Tensor* input = interpreter->getSessionInput(session, nullptr);
    // auto inputs = interpreter->getSessionInputAll(session);
    // auto outputs = interpreter->getSessionOutputAll(session);
    // for(std::map<std::string, MNN::Tensor>::iterator iter = inputs.begin(); iter != inputs.end(); ++iter) std::cout<<"Input key:"<<iter->first<<std::endl;
    // for(std::map<std::string, MNN::Tensor>::iterator iter = outputs.begin(); iter != outputs.end(); ++iter) std::cout<<"Output key:"<<iter->first<<std::endl;

    
    if (input->elementSize() <= 4) {
        interpreter->resizeTensor(input, {1, 1, 100, 12}); // Batch, Channel, Width, Height
        interpreter->resizeSession(session);
    }
    
    std::cout<<"Input data size:" << input->size() << std::endl;
    std::cout<<"Input element size:" << input->elementSize() << std::endl;
    std::cout<<"Input width:" << input->width() << std::endl;
    std::cout<<"Input height:" << input->height() << std::endl;
    std::cout<<"Input channel:" << input->channel() << std::endl;
    std::cout<<"Input batch:" << input->batch() << std::endl;

    for(auto dim : input->shape()) std::cout << "Input shape:" << dim << std::endl;

    std::cout<<"Creating an input copy:" <<std::endl; 
    MNN::Tensor inputCache(input);
    const int input_size = inputCache.elementSize();
    std::cout<<"DEBUG:Input dimensions:" << input_size << std::endl;
    auto input_data= inputCache.host<float>();
    for(int i = 0; i < input_size; ++i){
        input_data[i] = static_cast<float>(1);
    }

    input->copyFromHostTensor(&inputCache);

    std::cout<<"Creating an output copy:" <<std::endl; 
    MNN::Tensor* output = interpreter->getSessionOutput(session, nullptr);
    MNN::Tensor outputCopy(output); // Use tensorflow here for this model only MNN::Tensor::TENSORFLOW

    std::cout<<"Running a session:" <<std::endl; 
    auto output_test_data= output->host<float>();
    for(int i = 0; i < 10/*outputCopy.elementSize()*/; ++i){ std::cout<<"Output before:" << output_test_data[i] << std::endl;}
    
    AUTOTIME;
    for(int i = 0; i < 1; i ++){
        std::cout << "=============== Round:" << i << std::endl;
        interpreter->runSession(session);
        output_test_data= output->host<float>();
        for(int i = 0; i < 10/*outputCopy.elementSize()*/; ++i){ std::cout<<"Output after:" << output_test_data[i] << std::endl;}
        
        output->copyToHostTensor(&outputCopy);
        std::cout<<"Printing copied output:" << outputCopy.elementSize() <<std::endl; 
        output_test_data = outputCopy.host<float>();
        for(int i = 0; i < outputCopy.elementSize(); ++i){ std::cout<<"Output res:" << output_test_data[i] << std::endl;}
    }
    
    for(auto dim : output->shape()) std::cout << "Output shape:" << dim << std::endl;

    AUTOTIME;

    
    // Populating a model
    //MNN::Express::VARP input_data = MNN::Express::_Input({1,32}, MNN::Express::NC4HW4);
    //input_data->setName("data");

    // model.forward();
    // Get model results

}