#include <QCoreApplication>

#include "NvInfer.h"
#
#include "NvCaffeParser.h"
#include "NvOnnxConfig.h"
#include "NvInferPlugin.h"
#include "NvUtils.h"
#include "NvOnnxParser.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

#include <batchstream.h>
#include <entropycalibrator.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <string.h>
#include <cmath>
#include <time.h>
#include <map>
#include <memory.h>
#include <ctime>
#include <vector>
#include <cstdarg>
#include <cuda_runtime.h>
#include <vector>
#include <opencv2/dnn.hpp>


#define CHECK(status)									\
{														\
    if (status != 0)									\
    {													\
        std::cout << "Cuda failure: " << status;		\
        abort();										\
    }													\
}


class Logger : public nvinfer1::ILogger
{
       void log( Severity severity, const char* msg ) override
        {
            if( severity != Severity::kINFO /*|| mEnableDebug*/ )
               printf( "%s\n", msg);
        }
} gLogger;

// Check input size
static const int INPUT_H = 512;
static const int INPUT_W = 640;
static const int CHANNEL_NUM_R = 3;
static const int CHANNEL_NUM_T = 1;

// Check ONNX input && output name 
const char* INPUT_BLOB_NAME = "RGB";
const char* OUTPUT_BLOB_NAME = "loc";
const char* INPUT_BLOB_NAME2 = "Thermal";
const char* OUTPUT_BLOB_NAME2 = "cls";

// For NMS
// Make Prio Box
torch::Tensor create_prior_boxes(){

        std::map<std::string,float> fmap_dims[2];

        fmap_dims[0]["conv4_3"]=80;

        fmap_dims[0]["conv6"]=40;
        fmap_dims[0]["conv7"]=20;
        fmap_dims[0]["conv8"]=10;
        fmap_dims[0]["conv9"]=10;
        fmap_dims[0]["conv10"]=10;

        fmap_dims[1]["conv4_3"]=64;

        fmap_dims[1]["conv6"]=32;
        fmap_dims[1]["conv7"]=16;
        fmap_dims[1]["conv8"]=8;
        fmap_dims[1]["conv9"]=8;
        fmap_dims[1]["conv10"]=8;

        std::map<std::string,float> scale_ratios[3];
        scale_ratios[0]["conv4_3"]=1.;
        scale_ratios[0]["conv6"]=1.;
        scale_ratios[0]["conv7"]=1.;
        scale_ratios[0]["conv8"]=1.;
        scale_ratios[0]["conv9"]=1.;
        scale_ratios[0]["conv10"]=1.;

        scale_ratios[1]["conv4_3"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv6"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv7"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv8"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv9"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv10"]=(float)pow(2,1/3.);


        scale_ratios[2]["conv4_3"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv6"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv7"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv8"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv9"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv10"]=(float)pow(2,2/3.);

        std::map<std::string, float> aspect_ratios[2];
        aspect_ratios[1]["conv4_3"]=1.;
        aspect_ratios[1]["conv6"]=1.;
        aspect_ratios[1]["conv7"]=1.;
        aspect_ratios[1]["conv8"]=1.;
        aspect_ratios[1]["conv9"]=1.;
        aspect_ratios[1]["conv10"]=1.;

        aspect_ratios[0]["conv4_3"]=(float)1/2;
        aspect_ratios[0]["conv6"]=(float)1/2;
        aspect_ratios[0]["conv7"]=(float)1/2;
        aspect_ratios[0]["conv8"]=(float)1/2;
        aspect_ratios[0]["conv9"]=(float)1/2;
        aspect_ratios[0]["conv10"]=(float)1/2;

        std::map<std::string, double> anchor_areas;

        anchor_areas["conv4_3"]=40*40.;
        anchor_areas["conv6"]=80*80.;
        anchor_areas["conv7"]=160*160.;
        anchor_areas["conv8"]=200*200.;
        anchor_areas["conv9"]=280*280.;
        anchor_areas["conv10"]=360*360.;
        std::string fmaps[6]={"conv4_3", "conv6", "conv7", "conv8", "conv9", "conv10"};
        double cx,cy;
        double h,w,anchor_h,anchor_w;
        std::string fmap_i;
        double prior_box[4];

        torch::Tensor prior_boxs = torch::rand({41760,4});

        int numbers=0;

        for(int fmap =0 ; fmap<6;fmap++){
            fmap_i=fmaps[fmap];

            for(int i=0;i<fmap_dims[1][fmap_i];i++){
                for(int j=0;j<fmap_dims[0][fmap_i];j++){
                   cx=(j+0.5)/fmap_dims[0][fmap_i];
                   cy=(i+0.5)/fmap_dims[1][fmap_i];
                   for(int s=0;s<1;s++){
                       for(int ar =0;ar<2;ar++){

                           h=sqrt(anchor_areas[fmap_i]/aspect_ratios[ar][fmap_i]);
                           w=aspect_ratios[ar][fmap_i]*h;
                           for( int sr =0;sr<3;sr++){
                               anchor_h=h*scale_ratios[sr][fmap_i]/512.;
                               anchor_w=w*scale_ratios[sr][fmap_i]/640.;
                               prior_boxs[numbers][0]=cx;
                               prior_boxs[numbers][1]=cy;
                               prior_boxs[numbers][2]=anchor_w;
                               prior_boxs[numbers][3]=anchor_h;
                               numbers++;

                           }
                       }
                   }
                }
            }
        }


        return prior_boxs;
}
torch::Tensor cxcy_to_xy(torch::Tensor cxcy){

    return torch::cat({cxcy.slice(1,0,2) - (cxcy.slice(1,2) / 2),cxcy.slice(1,0,2) + (cxcy.slice(1,2) / 2)}, 1);
}
torch::Tensor gcxgcy_to_cxcy(torch::Tensor gcxgcy,torch::Tensor priors_cxcy){
    torch::Tensor a,b,c;

    a=torch::mul(gcxgcy.slice(1,0,2),priors_cxcy.slice(1,2))/10+priors_cxcy.slice(1,0,2);
    b=torch::exp(gcxgcy.slice(1,2)/5)*priors_cxcy.slice(1,2);

    return torch::cat({a,b},1);
}
torch::Tensor find_intersection(torch::Tensor set_1,torch::Tensor set_2){
    torch::Tensor lower_bounds,upper_bounds,uml,intersection_dims;

    lower_bounds=torch::max(set_1.slice(1,0,2).unsqueeze(1),set_2.slice(1,0,2).unsqueeze(0));
    upper_bounds=torch::min(set_1.slice(1,2).unsqueeze(1),set_2.slice(1,2).unsqueeze(0));
    uml= (upper_bounds-lower_bounds);
    intersection_dims=uml.clamp(0);

     return intersection_dims.slice(2,0,1)*intersection_dims.slice(2,1,2);
}
// Calculate IOU BBOX
torch::Tensor find_overlap(torch::Tensor set_1,torch::Tensor set_2){
    torch::Tensor intersection,areas_set_1,areas_set_2,union_;

    intersection=find_intersection(set_1,set_2);
    areas_set_1 = (set_1.slice(1,2,3) - set_1.slice(1,0,1)) * (set_1.slice(1,3,4)- set_1.slice(1,1,2)) ;

    areas_set_2 = (set_2.slice(1,2,3) - set_2.slice(1,0,1)) * (set_2.slice(1,3,4)- set_2.slice(1,1,2)) ;
    union_=areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection;

    return   intersection / union_;

}
// NMS Main
torch::Tensor detect_objects(torch::Tensor predicted_locs,torch::Tensor predicted_scores,torch::Tensor priors_xy,double min_score,double max_overlap,int top_k){
    torch::Tensor decode_loc,class_scores ,score_above_min_score;

    predicted_scores=predicted_scores.softmax(2);

    decode_loc=cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[0],priors_xy));
  //std::cout<<decode_loc<<std::endl;
    int n_above_min_score;
    int classname=1;
    class_scores=predicted_scores[0].slice(1,classname,classname+1);

    score_above_min_score=class_scores>min_score;
    n_above_min_score=score_above_min_score.sum().item<int>();
    //std::cout<<n_above_min_score<<std::endl;
    if(n_above_min_score==0){
        torch::Tensor out=torch::zeros({1,5}).to(at::kCUDA);
        out[0][2]=1.;
        out[0][3]=1.;
        return out;
    }

    torch::Tensor suppress;
    torch::Tensor out_boxes,out_scores,out_label;
    int up=0;

    //auto indexing_minscore=torch::nonzero(score_above_min_score);

    auto order_t=std::get<1>(class_scores.sort(0,true));
    auto class_sorted=class_scores.index_select(0,order_t.squeeze(-1));
    torch::Tensor decode_loc_sorted=decode_loc.index_select(0,order_t.squeeze(-1));
    int dets_num = decode_loc.size(0);

    class_sorted=class_sorted.slice(0,0,n_above_min_score);
    decode_loc_sorted=decode_loc_sorted.slice(0,0,n_above_min_score);

//std::cout<<"find_oveerlap"<<std::endl;

    auto overlap=find_overlap(decode_loc_sorted,decode_loc_sorted);
    //NMS
    suppress=torch::zeros((n_above_min_score)).to(at::kCUDA);

  //std::cout<<"start NMS"<<std::endl;
    for(int box=0;box<decode_loc_sorted.size(0);box++){



        if(suppress[box].item<int>()==1){
            continue;
        }

        torch::Tensor what=overlap[box]>max_overlap;
        what=what.to(at::kFloat);

        auto maxes=torch::max(suppress,what.squeeze(1));

        suppress=maxes;


        suppress[box]=0;

    }

    suppress=1-suppress;

    auto indexing=torch::nonzero(suppress);

    out_boxes=decode_loc_sorted.index_select(0,indexing.squeeze(-1));

    out_scores=class_sorted.index_select(0,indexing.squeeze(-1));

    return torch::cat({out_boxes,out_scores},1);
}
std::vector<std::string> split(std::string str, char delimiter) {
    std::vector<std::string> internal;
    std::stringstream ss(str);
    std::string temp;

    while (std::getline(ss, temp, delimiter)) {
        internal.push_back(temp);
    }

    return internal;
}


std::string format(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    #ifndef _MSC_VER
        size_t size = std::snprintf( nullptr, 0, format, args) + 1; // Extra space for '\0'
        std::unique_ptr<char[]> buf( new char[ size ] );
        std::vsnprintf( buf.get(), size, format, args);
        return std::string(buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
    #else
        int size = _vscprintf(format, args);
        std::string result(++size, 0);
        vsnprintf_s((char*)result.data(), size, _TRUNCATE, format, args);
        return result;
    #endif
    va_end(args);
}

// Image Normalize on Torch
at::Tensor Normalize(at::Tensor img,int what){//
    at::Tensor normimg,tensor_img_R_R,tensor_img_R_G,tensor_img_R_B;

    if(what==1){
//        normimg=img.div_(255.).sub_(0.4126);
        normimg=img.div_(255.).mul_(2).sub_(1);
    }
    else{
        tensor_img_R_R=img.slice(1,0,1);
        tensor_img_R_G=img.slice(1,1,2);
        tensor_img_R_B=img.slice(1,2,3);
//        tensor_img_R_R=tensor_img_R_R.div_(255.).sub_(0.5873);
//        tensor_img_R_G=tensor_img_R_G.div_(255.).sub_(0.5328);
//        tensor_img_R_B=tensor_img_R_B.div_(255.).sub_(0.4877);
        tensor_img_R_R=tensor_img_R_R.div_(255.).mul_(2).sub_(1);
        tensor_img_R_G=tensor_img_R_G.div_(255.).mul_(2).sub_(1);
        tensor_img_R_B=tensor_img_R_B.div_(255.).mul_(2).sub_(1);

        normimg=torch::cat({tensor_img_R_R,tensor_img_R_G,tensor_img_R_B},1);
    }
    return normimg;
}
at::Tensor Mat2Tensor(cv::Mat img){
    int channel=img.channels();
    at::Tensor tensor_img;
    cv::resize(img,img,cv::Size(640,512));
    if(channel==1){
        std::vector<int64_t> dims{1,static_cast<int64_t>(img.channels()),
                                           static_cast<int64_t>(img.rows),
                                           static_cast<int64_t>(img.cols)};
        tensor_img=torch::from_blob(img.data,dims,at::kByte);
       tensor_img = tensor_img.to(at::kFloat);
    }
    else{
        std::vector<int64_t> dims{1,
                                           static_cast<int64_t>(img.rows),
                                           static_cast<int64_t>(img.cols),static_cast<int64_t>(img.channels())};
        tensor_img=torch::from_blob(img.data,dims,at::kByte);
        tensor_img=tensor_img.permute({0,3,1,2});
    }
    tensor_img = tensor_img.to(at::kFloat);

    tensor_img=Normalize(tensor_img,channel);
    return tensor_img;
}
void setAllTensorScales(nvinfer1::INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f)
{
    // Ensure that all layer inputs have a scale.

    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                input->setDynamicRange(-inScales, inScales);
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                  //std::cout<<layer->getType() <<std::endl;
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    output->setDynamicRange(-inScales, inScales);
                }
                else
                {
                    output->setDynamicRange(-outScales, outScales);
                }
            }
        }
    }
}

void enableDLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            //assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {

            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }

        if (!builder->getInt8Mode() && !config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            builder->setFp16Mode(true);
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }
}

nvinfer1::ICudaEngine* loadEngine(const std::string& engine, int DLACore, std::ostream& err)
{
    std::ifstream engineFile(engine, std::ios::binary);
    if (!engineFile)
    {
        err << "Error opening engine file: " << engine << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        err << "Error loading engine file: " << engine << std::endl;
        return nullptr;
    }
    nvinfer1::IRuntime* runtime=   nvinfer1::createInferRuntime(gLogger);

    if (DLACore != -1)
        {
            runtime->setDLACore(DLACore);
        }


    return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}
bool saveEngine(const nvinfer1::ICudaEngine& engine, const std::string& fileName, std::ostream& err)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        err << "Cannot open engine file: " << fileName << std::endl;
        return false;
    }
    nvinfer1::IHostMemory *serializedEngine=engine.serialize();


    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

// Make Engine from ONNX
nvinfer1::ICudaEngine *constructNetwork(nvinfer1::IBuilder* builder,nvinfer1::INetworkDefinition* network,
    nvonnxparser::IParser* parser)
{
    //nvinfer1::IInt8Calibrator *calibrator;

    int DLACore=-1;
	// Check ONNX path
    std::string onnx_filename ="/home/rcvsejong2/raid/hhnam_workspace/Xavier2/tensorrt/dc_workspace/halfway_v6.onnx";
    auto parsed = parser->parseFromFile(onnx_filename.c_str(), 0);

    auto config=builder->createBuilderConfig();
    builder->setMaxBatchSize(1);
    std::size_t size=1ULL<<64,size2;
    config->setMaxWorkspaceSize(size);
    size2=config->getMaxWorkspaceSize();
    std::cout<<size2<<std::endl;
//    std::ifstream filepath("/media/rcvsejong2/XavierSSD256/raid/datasets/kaist-rgbt/imageSets/test-all-20.txt");
//    std::vector<std::string> R_paths,T_paths;
//    std::string str;
//    while(std::getline(filepath, str))
//    {
//        std::string imgname_T,imgname_R;
//        std::vector<std::string> line_vector = split(str, '/');
//        imgname_T="/media/rcvsejong2/XavierSSD256/raid/datasets/kaist-rgbt/images/"+line_vector[0]+"/"+line_vector[1]+"/lwir/"+line_vector[2].substr(0,6)+".jpg";
//        imgname_R="/media/rcvsejong2/XavierSSD256/raid/datasets/kaist-rgbt/images/"+line_vector[0]+"/"+line_vector[1]+"/visible/"+line_vector[2].substr(0,6)+".jpg";
//        R_paths.push_back(imgname_R);
//        T_paths.push_back(imgname_T);

//    }

//    std::unique_ptr<nvinfer1::IInt8EntropyCalibrator> calibrator;


//    BatchStream calibrationStream(4,2250/4,R_paths,T_paths);

//    calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(calibrationStream,0,"SF_BATCH","RGB"));

    if (false)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (true)
    {
        std::cout<<"Int 8 Setting"<<std::endl;
//        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        //setAllTensorScales(network, 127.0f, 127.0f);
//        config->setInt8Calibrator(calibrator.get());
        //config->setInt8Calibrator(NULL);

    }
    std::cout<<"Done config"<<std::endl;
    enableDLA(builder, config, DLACore);
    bool error;
    nvinfer1::ICudaEngine *mEngine;
    mEngine=builder->buildEngineWithConfig(*network,*config);
    if(!mEngine){
        std::cout<<"ERRO::MAKE ENGINE"<<std::endl;
    }
    builder->destroy();
    network->destroy();
    parser->destroy();
    config->destroy();
    std::ofstream err;
    std::cout<<"Done Make Engine"<<std::endl;
    // Check Engine Path
    saveEngine(*mEngine,"/home/rcvsejong2/raid/hhnam_workspace/Xavier2/tensorrt/dc_workspace/Halfway_data_callibration_full_batch.engine",err);
    std::cout<<"Done Save Engine"<<std::endl;
    return mEngine;
}

// image Mat to float pointer 
void imageCalculation(cv::Mat img_input, const int INPUT_W_, const int INPUT_H_, const int CHANNEL_NUM, float* data)
{
        cv::Mat Img;
        Img = img_input;
        cv::Mat channel[CHANNEL_NUM];
        std::cout<<"Start split"<<std::endl;
        if(Img.channels()>1){
            cv::split(Img,channel);

        }
        else{
            channel[0]=Img;

        }
        int num_time=0;
        //std::vector<
        for(int k=0;k<CHANNEL_NUM;k++)
        {
            for(int i=0;i<INPUT_H;i++)
            {
                for(int j=0;j<INPUT_W;j++)
                {
                    //std::cout<<(int)channel[k].at<uchar>(i, sj)<<std::endl;
                    data[num_time]=channel[k].at<float>(i,j);
                    num_time++;
                }
            }
        }

}
// Forward
void doInference2(nvinfer1::IExecutionContext& context, float* input, float* input2, float32_t* output,float32_t* output2, int batchSize, int output_size_, int output_size_2)
{
    const nvinfer1::ICudaEngine& engine = context.getEngine();

    //assert(engine.getNbBindings() == 2);
    void* buffers[4];
    size_t input_size= batchSize * INPUT_H * INPUT_W * CHANNEL_NUM_R * sizeof(float32_t);
    size_t input_size2= batchSize * INPUT_H * INPUT_W * CHANNEL_NUM_T * sizeof(float32_t);

   int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
   int inputIndex2 = engine.getBindingIndex(INPUT_BLOB_NAME2),outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);

    CHECK(cudaMalloc(&buffers[inputIndex],input_size));
    CHECK(cudaMalloc(&buffers[inputIndex2],input_size2));

    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * output_size_ * sizeof(float32_t)));
    CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * output_size_2 * sizeof(float32_t)));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, input_size, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex2], input2, input_size2, cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * output_size_*sizeof(float32_t), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output2, buffers[outputIndex2], batchSize * output_size_2*sizeof(float32_t), cudaMemcpyDeviceToHost, stream));

    CHECK(cudaStreamSynchronize(stream));

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));

    CHECK(cudaFree(buffers[inputIndex2]));
   CHECK( cudaFree(buffers[outputIndex]));
   CHECK( cudaFree(buffers[outputIndex2]));

}
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

void torch2float(at::Tensor img,float* data){

    data=img.data<float>();

}












int main(int argc, char *argv[])
{
    int DLACores=-1;
    bool mEnableDebug    = true;
   bool mOverride16     = false;
   // if you have engine? can_load =TRUE : can_load =FALSE 
    bool can_load=true;

    QCoreApplication a(argc, argv);
    //Builder
     nvinfer1::ICudaEngine* mEngine;
     std::cout<<can_load<<std::endl;
    if(!can_load)
    {   // Make Engine from ONNX
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
        //Network
        nvinfer1::INetworkDefinition* network = builder->createNetwork();
        builder->setDebugSync(mEnableDebug);
        builder->setMinFindIterations(5);	// allow time for TX1 GPU to spin up
        builder->setAverageFindIterations(5);
        //Config

        
		//Parser

        auto parser = nvonnxparser::createParser(*network,gLogger);
        std::cout<<"Load Start"<<std::endl;
        nvinfer1::ICudaEngine *mEngine;
        mEngine=constructNetwork(builder,network,parser);

        if(!mEngine){
            std::cout<<"ERROR:: Construct Network\n"<<std::endl;
        }


    }
    else{
		// Check Engine Path
        std::ofstream err;;
        mEngine=loadEngine("/home/rcvsejong2/raid/hhnam_workspace/Xavier2/tensorrt/dc_workspace/Halfway_data_callibration_full_batch.engine",DLACores,err);
    }
    //return -1;
    std::cout<<"Load Done"<<std::endl;
    // ////////////////////////////////////////////////////////////////////////////////////////////
    // Infer
    // ////////////////////////////////////////////////////////////////////////////////////////////

    int batchSize=1;
    int size_of_single_input=640*512*4;
    int size_of_single_output=41760*4;
    int size_of_single_output2=41760*2;
    nvinfer1::IExecutionContext *context=mEngine->createExecutionContext();
    std::vector<void*>  buffers;
    std::string str;
    // img_R,img_T;
//    std::ofstream myfile;
//    myfile.open ("/home/rcvsejong2/tensorrt/dc_workspace/output_before_cls.txt");
//    //Load Image
    std::ifstream file2("/media/rcvsejong2/XavierSSD256/raid/datasets/kaist-rgbt/imageSets/test-all-20.txt");
    torch::Tensor prior_xy=create_prior_boxes(),result;
    std::string imgname_T,imgname_R;
    std::string eval_path="/home/rcvsejong2/eval_calibration_1.txt";
    torch::Tensor detect_loc,detect_score;
    int i=0;
    std::ofstream writefile(eval_path.data());
    int64 e1,e2;
    while (std::getline(file2, str))
    {
         e1 = cv::getTickCount();
        std::vector<std::string> line_vector = split(str, '/');
         imgname_T="/media/rcvsejong2/XavierSSD256/raid/datasets/kaist-rgbt/images/"+line_vector[0]+"/"+line_vector[1]+"/lwir/"+line_vector[2].substr(0,6)+".jpg";
         imgname_R="/media/rcvsejong2/XavierSSD256/raid/datasets/kaist-rgbt/images/"+line_vector[0]+"/"+line_vector[1]+"/visible/"+line_vector[2].substr(0,6)+".jpg";

        cv::Mat img_R,img_T,img_R_ori,original_img;
        img_R_ori=cv::imread(imgname_R,cv::IMREAD_COLOR);
        img_T=cv::imread(imgname_T,cv::IMREAD_GRAYSCALE);
        original_img=img_R_ori.clone();
        int width,height;
        width=original_img.cols;
        height=original_img.rows;
        auto original_dims = torch::tensor({width,height ,width,height}).unsqueeze(0);
        original_dims=original_dims.to(at::kFloat);

        original_dims=original_dims.to(at::kCUDA);

        std::vector<cv::Mat> bgr(3),rgb;
        //std::cout<<"img_T"<<std::endl;
        cv::resize(img_R_ori,img_R_ori,{640,512});
        cv::resize(img_T,img_T,{640,512});

        cv::cvtColor(img_R_ori,img_R_ori,cv::COLOR_RGB2BGR);
        img_R.convertTo(img_R,CV_32FC3);
        at::Tensor img_R_tensor,img_T_tensor;
        //img_R_tensor=Mat2Tensor(img_R);
        //img_T_tensor=Mat2Tensor(img_T);
        //Noramlize;
        //        normimg=img.div_(255.).sub_(0.4126);

        //        tensor_img_R_R=tensor_img_R_R.div_(255.).sub_(0.5873);
        //        tensor_img_R_G=tensor_img_R_G.div_(255.).sub_(0.5328);
        //        tensor_img_R_B=tensor_img_R_B.div_(255.).sub_(0.4877);
        std::vector<cv::Mat> channel;
        cv::split(img_R_ori,channel);
        channel[0].convertTo(channel[0],CV_32FC1);
        channel[1].convertTo(channel[1],CV_32FC1);
        channel[2].convertTo(channel[2],CV_32FC1);

        img_T.convertTo(img_T,CV_32FC1);
        //        img_T=(img_T/255.-(0.4126));
        //        channel[0]=(channel[0]/255.-(0.5873));
        //        channel[1]=(channel[1]/255.-(0.5328));
        //        channel[2]=(channel[2]/255.-(0.4877));

        cv::merge(channel,img_R);

        //return -1;
        //std::cout<<img_R<<std::endl;

        img_T=((2/255.)*img_T-(1));
        img_R=((2/255.)*img_R-(1));

        // ////////////////////////////////////////////////////////
        //
        // ////////////////////////////////////////////////////////
        std::cout<<"START"<<std::endl;


        float32_t *output=(float32_t*)malloc(size_of_single_output*sizeof(float32_t));
        float32_t *output2=(float32_t*)malloc(size_of_single_output2*sizeof(float32_t));

        float data_R[INPUT_H*INPUT_W*CHANNEL_NUM_R];
        imageCalculation(img_R, INPUT_W,INPUT_H, CHANNEL_NUM_R, data_R);
        //torch2float(img_R_tensor,data_R);
        float data_T[INPUT_H*INPUT_W*CHANNEL_NUM_T];
        //torch2float(img_T_tensor,data_T);
        imageCalculation(img_T, INPUT_W,INPUT_H, CHANNEL_NUM_T,  data_T);

        doInference2(*context, data_R,data_T, output,output2, 1, size_of_single_output, size_of_single_output2);

        std::cout<<"Done Inference"<<std::endl;

        torch::Tensor loc=torch::zeros({1,41760,4}).to(at::kFloat);
        torch::Tensor cls=torch::zeros({1,41760,2}).to(at::kCUDA).to(at::kFloat);
        loc=torch::from_blob((void *)output,{1,41760,4},at::kFloat).to(at::kCUDA);
        cls=torch::from_blob((void *)output2,{1,41760,2},at::kFloat).to(at::kCUDA);



        prior_xy=prior_xy.to(at::kCUDA);
        std::cout<<"Start"
                   " NMS"<<std::endl;
        //std::cout<<cls<<std::endl;

        result=detect_objects(loc,cls,prior_xy,0.1,0.45,200);

        detect_score=result.slice(1,4,5);

        detect_loc=result.slice(1,0,4);
        std::cout<<detect_loc.sizes()<<std::endl;
        detect_loc=detect_loc*original_dims;
        std::cout<<detect_loc<<std::endl;
        i++;
        for(int box_num=0;box_num<detect_loc.size(0);box_num++){
            //std::cout<<detect_loc[box_num]<<std::endl;
            cv::Rect point;
            int x1=detect_loc[box_num][0].item<int>();
            int y1=detect_loc[box_num][1].item<int>();
            int x2=detect_loc[box_num][2].item<int>();
            int y2=detect_loc[box_num][3].item<int>();
            //          cout<<x1<<endl;
            point.x=x1;
            point.y=y1;
            point.height=y2-y1;
            point.width=x2-x1;

            // Eval
//            if(writefile.is_open()){
//                writefile<<"score "<<detect_score[box_num].item<float>()<<" image_id "<<i<<" bbox "<<x1<<","<<y1<<","<<x2<<","<<y2<<"\n";

//            }
            cv::rectangle(original_img,point,(255,0,255));

        }
        char ii[100];
        sprintf(ii,"test_New_0_%06d.png",i);
        std::string savename="/home/rcvsejong2/raid/raid/test_4/";
        savename+=ii;
        cv::imwrite(savename,original_img);
        e2 = cv::getTickCount();
        std::cout<<(e2-e1)/cv::getTickFrequency()<<std::endl;
        std::cout<<i<<"/2252"<<std::endl;
        //return -1;
    }
    //writefile.close();
    return a.exec();
}
