#include <MNN/AutoTime.hpp> // Timer
#include <iostream>
#include "model.h"

#include <MNN/Interpreter.hpp>

#define NUM_OF_THREADS 4

int main(int argc, const char* argv[]) {

    if (argc < 2) {
        MNN_ERROR("Please specify a model path\n");
        MNN_ERROR("USAGE:./main.a [model_path]\n");
        return 0;
    }

    auto model_path = argv[1];
    FUNC_PRINT_ALL(model_path, s);

    auto interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path));
    // auto interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(ATTENTION_MODEL, sizeof(ATTENTION_MODEL)));
    
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
        std::cout<<"Resizing tensor:" << std::endl;
        interpreter->resizeTensor(input, {1, 1, 1, 1250}); // Batch, Channel, Width, Height
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
    float* input_test_buff = (float*)malloc(1250* sizeof(float));

    const int input_size = inputCache.elementSize();
    std::cout<<"DEBUG:Input dimensions:" << input_size << std::endl;
    auto input_data= input->host<float>();
    for(int i = 0; i < input_size; ++i){
        input_test_buff[i] = 1;
        input_data[i] = static_cast<float>(1);
    }
    MNN::Tensor* inputTestTensor = MNN::Tensor::create<float>({1, 1, 1, 1250}, input_test_buff);
    input->copyFromHostTensor(inputTestTensor);

    std::cout<<"Creating an output copy:" <<std::endl; 
    MNN::Tensor* output = interpreter->getSessionOutput(session, nullptr);
    MNN::Tensor outputCopy(output); // Use tensorflow here for this model only MNN::Tensor::TENSORFLOW

    std::cout << "Running a session:" <<std::endl; 
    float* output_test_data= output->host<float>();
    for(int i = 0; i < output->elementSize(); ++i){ std::cout<<"Output before:" << output_test_data[i] << std::endl;}
    
    AUTOTIME;
    for(int i = 0; i < 1; i ++){
        std::cout << "=============== Round:" << i << std::endl;
        interpreter->runSession(session);
        output_test_data= output->host<float>();
        for(int i = 0; i < outputCopy.elementSize(); ++i){ std::cout<<"Output after:" << output_test_data[i] << std::endl;}
        
        output->copyToHostTensor(&outputCopy);
        std::cout<<"Printing copied output:" << outputCopy.elementSize() <<std::endl; 
        output_test_data = outputCopy.host<float>();
        for(int i = 0; i < outputCopy.elementSize(); ++i){ std::cout<<"Output res:" << output_test_data[i] << std::endl;}
    }
    
    std::cout << "Output size:" << output -> elementSize() << std::endl;
    for(auto dim : output -> shape()) std::cout << "Output shape:" << dim << std::endl;

    AUTOTIME;



}