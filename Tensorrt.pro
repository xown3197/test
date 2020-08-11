#QT += core
QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp
####################
SYSTEM_TYPE = 64
CUDA_ARCH = sm_61

CUDA_SDK = "/usr/local/cuda"   # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda"   # Path to cuda toolkit install

NVCC_OPTIONS  =  --use_fast_math
INCLUDEPATH   += $$CUDA_DIR/include
QMAKE_LIBDIR  += $$CUDA_DIR/lib64/

CUDA_OBJECTS_DIR = ./
CUDA_LIBS = cudart cufft
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
NVCC_LIBS = $$join(CUDA_LIBS,' -l','-l', '')

INCLUDEPATH += /usr/local/cuda-10.0/include
LIBS += -L/usr/local/cuda/lib64 \
-lcublas  -lcufft    -lcurand    -lnppc  -lnvToolsExt   \
-lcudart  -lcufftw   -lcusolver  \
-lcudnn   -lcuinj64  -lcusparse  -lnpps

LIBS += -lnvcaffe_parser\
    -lnvinfer_plugin\
    -lnvparsers\
    -lnvonnxparser\
    -lnvinfer\
    -lnvinfer_plugin\
    -lnvonnxparser_runtime\

CONFIG(debug, debug|release) {
     # Debug mode
     cuda_d.input = CUDA_SOURCES
     cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
     cuda_d.commands = $$CUDA_DIR/bin/nvcc -g -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
     cuda_d.dependency_type = TYPE_C
     QMAKE_EXTRA_COMPILERS += cuda_d
 }
 else {
     # Release mode
     cuda.input = CUDA_SOURCES
     cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
     cuda.commands = $$CUDA_DIR/bin/nvcc -g -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
     cuda.dependency_type = TYPE_C
     QMAKE_EXTRA_COMPILERS += cuda
}
####################################################
## Torch && Opencv
win32:CONFIG(release, debug|release): LIBS += -L/usr/lib/release/ -lopencv_aruco
else:win32:CONFIG(debug, debug|release): LIBS += -L/usr/lib/debug/ -lopencv_aruco
else:unix: LIBS += -L/usr/lib/aarch64-linux-gnu -lopencv_calib3d -lopencv_core  \
                   -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_gapi -lopencv_highgui \
                   -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_videoio \


INCLUDEPATH += /usr/include/opencv4/
DEPENDPATH += /usr/include/opencv4/

LIBS += \
        -L/usr/local/lib/python3.6/dist-packages/torch/lib

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../usr/local/lib/python3.6/dist-packages/torch/lib/release/ -lc10_cuda
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../usr/local/lib/python3.6/dist-packages/torch/lib/debug/ -lc10_cuda
else:unix: LIBS += -L/usr/local/lib/python3.6/dist-packages/torch/lib/ -lc10 -lc10_cuda -lcaffe2 -lcaffe2_detectron_ops_gpu \
        -lcaffe2_gpu -lcaffe2_module_test_dynamic -lcaffe2_observers -lfoxi -lfoxi_dummy -lonnxifi -lonnxifi_dummy \
        -lshm -lthnvrtc -ltorch -ltorch_python

INCLUDEPATH += /usr/local/lib/python3.6/dist-packages/torch/include
DEPENDPATH += /usr/local/lib/python3.6/dist-packages/torch/include


INCLUDEPATH +=/usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include
DEPENDPATH += /usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include



INCLUDEPATH += /usr/include/python3.6
DEPENDPATH += /usr/include/python3.6

HEADERS += \
    entropycalibrator.h \
    batchstream.h

