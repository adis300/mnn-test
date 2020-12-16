rm main.a
g++ -Bstatic -std=c++11 -I ../include -L ../mnn-mac/static -lMNN -o main.a main.cpp #-rpath @executable_path #-lMNN_Express
#g++ -std=c++11 -I ../include -L ../mnn-mac -lMNN -o main.a main.cpp -rpath @executable_path #-lMNN_Express

# install_name_tool -change @rpath/libMNN_Express.dylib @executable_path/`basename libMNN_Express.dylib` ./main.a
#./main.a ../model/model-mobilenet_v1_075.mnn
./main.a ../model/emgplus.mnn