#ifndef ENTROPYCALIBRATOR_H
#define ENTROPYCALIBRATOR_H

#include "batchstream.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>

#define CHECK(status)									\
{														\
    if (status != 0)									\
    {													\
        std::cout << "Cuda failure: " << status;		\
        abort();										\
    }													\
}
template <typename TBatchStream>
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl(
        TBatchStream stream, int firstBatch, std::string networkName, const char* inputBlobName, bool readCache = true)
        : mStream{stream}
        , mCalibrationTableName("CalibrationTable" + networkName)
        , mInputBlobName(inputBlobName)
        , mReadCache(readCache)
    {
        std::cout<<"Initialize Impl Calibrator!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
        mInputCountR = 3*512*640*mStream.getBatchSize();
        mInputCountT = 1*512*640*mStream.getBatchSize();

        CHECK(cudaMalloc(&mDeviceInputR, mInputCountR * sizeof(float)));
        CHECK(cudaMalloc(&mDeviceInputT, mInputCountT * sizeof(float)));

    }

    virtual ~EntropyCalibratorImpl()
    {
        std::cout<<"Done Impl Calibrator!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;

        CHECK(cudaFree(mDeviceInputR));
        CHECK(cudaFree(mDeviceInputT));

    }

    int getBatchSize() const
    {
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings)
    {
        std::cout<<"Load Batch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;

        if (!mStream.next())
        {
            return false;
        }

        CHECK(cudaMemcpy(mDeviceInputR, mStream.getBatchR(), mInputCountR * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(mDeviceInputT, mStream.getBatchT(), mInputCountT * sizeof(float), cudaMemcpyHostToDevice));
        //std::cout<<mDeviceInputR<<std::endl;
        bindings[0] = mDeviceInputR;
        bindings[1] = mDeviceInputT;

        return true;
    }

    const void* readCalibrationCache(size_t& length)
    {
        std::cout<<"Read Cache!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;

        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;

        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length)
    {
        std::cout<<"write Batch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;

        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:

    TBatchStream mStream;
    size_t mInputCountR;
    size_t mInputCountT;

    std::string mCalibrationTableName;
    const char* mInputBlobName;
    bool mReadCache{true};
    void* mDeviceInputR{nullptr};
    void* mDeviceInputT{nullptr};
    std::vector<char> mCalibrationCache;
};

// \class Int8EntropyCalibrator2
//
// \brief Implements Entropy calibrator 2.
//  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//
template <typename TBatchStream>
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator
{
public:
    Int8EntropyCalibrator2(
        TBatchStream stream, int firstBatch, const char* networkName, const char* inputBlobName, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, inputBlobName, readCache)
    {
        std::cout<<"Initialize Calibrator!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
    }

    int getBatchSize() const override
    {
        std::cout<<"Get Batchsize Calibrator!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
        std::cout<<mImpl.getBatchSize()<<std::endl;
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        std::cout<<"Load Batch !!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;

        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) override
    {
        std::cout<<"Read Cache 1  !!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;

        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::cout<<"write Cache 1  !!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;

        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl<TBatchStream> mImpl;
};

#endif

