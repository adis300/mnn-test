rm *.a
g++ -std=c++11 -I ../include -L ../mnn-mac -lMNN -o main_cmsn.a main_cmsn.cpp -rpath @executable_path #-lMNN_Express

./main_cmsn.a ../model/cmsn.mnn