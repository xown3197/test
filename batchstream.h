#ifndef BATCHSTREAM_H
#define BATCHSTREAM_H

#include "NvInfer.h"
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS
class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getBatchR() = 0;
    virtual float* getBatchT() = 0;

    virtual float* getLabels() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};
void imagetodata(cv::Mat img_input, const int INPUT_W_, const int INPUT_H_, const int CHANNEL_NUM, std::vector<float> data,int batch)
{
        cv::Mat Img;
        Img = img_input;
        cv::Mat channel[CHANNEL_NUM];
        if(Img.channels()>1){
            cv::split(Img,channel);

        }
        else{
            channel[0]=Img;

        }
        int num_time=0;
        int total=INPUT_H_*INPUT_W_*CHANNEL_NUM;
        //std::vector<
        for(int k=0;k<CHANNEL_NUM;k++)
        {
            for(int i=0;i<INPUT_H_;i++)
            {
                for(int j=0;j<INPUT_W_;j++)
                {
                    data[batch*total+num_time]=channel[k].at<float>(i,j);
                    num_time++;
                }
            }
        }

}
class BatchStream : public IBatchStream
{
public:
    BatchStream(
        int batchSize, int maxBatches, std::vector<std::string> directories_R, std::vector<std::string> directories_T)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mDataDirR(directories_R)
        , mDataDirT(directories_T)
    {
        mDimsR.nbDims = 4;  // The number of dimensions.
        mDimsR.d[0] =mBatchSize; //d[0]; // Batch Size
        mDimsR.d[1] =3; //d[1]; // Channels
        mDimsR.d[2] =512; //d[2]; // Height
        mDimsR.d[3] =640; //d[3]; // Width

        mDimsT.nbDims = 4;  // The number of dimensions.
        mDimsT.d[0] =mBatchSize; //d[0]; // Batch Size
        mDimsT.d[1] =1; //d[1]; // Channels
        mDimsT.d[2] =512; //d[2]; // Height
        mDimsT.d[3] =640; //d[3]; // Width

        mImageSizeR = mDimsR.d[1] * mDimsR.d[2] * mDimsR.d[3];
        mImageSizeT = mDimsT.d[1] * mDimsT.d[2] * mDimsT.d[3];

        mBatchR.resize(mBatchSize * mImageSizeR, 0);
        mBatchT.resize(mBatchSize * mImageSizeT, 0);

        mFileBatchR.resize(mDimsR.d[0] * mImageSizeR, 0);
        mFileBatchT.resize(mDimsT.d[0] * mImageSizeT, 0);

        reset(0);
    }

    // Resets data members
    void reset(int firstBatch) override
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDimsR.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next() override
    {
        if (mBatchCount == mMaxBatches)
        {
            return false;
        }

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            if (mFileBatchPos == mDimsR.d[0] && !update())
            {
                return false;
            }
            csize = std::min(mBatchSize - batchPos, mDimsR.d[0] - mFileBatchPos);
            std::copy_n(getFileBatchR() + mFileBatchPos * mImageSizeR, csize * mImageSizeR, getBatchR() + batchPos * mImageSizeR );
            std::copy_n(getFileBatchT()+ mFileBatchPos * mImageSizeT , csize * mImageSizeT, getBatchT()  + batchPos * mImageSizeT);

        }
        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount) override
    {
        if (mBatchSize >= mDimsR.d[0] && mBatchSize % mDimsR.d[0] == 0 && mFileBatchPos == mDimsR.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDimsR.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
        {
            next();
        }
        mBatchCount = x;

    }

    float* getBatchR() override
    {
        return mBatchR.data();
    }
    float* getLabels() override
    {
        return mBatchR.data();
    }
    float* getBatchT() override
    {
        return mBatchT.data();
    }
    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return mDimsR;
    }

private:
    float* getFileBatchR()
    {
        return mFileBatchR.data();
    }
    float* getFileBatchT()
    {
        return mFileBatchT.data();
    }
    float* getFileLabels()
    {
        return mFileLabels.data();
    }

    bool update()
    {
           std::vector<cv::Mat> R_imgs,T_imgs;
           for(int batchs=0;batchs<mDimsR.d[0];batchs++)
           {
                std::string R_path,T_path;
                R_path=mDataDirR[mFileCount];
                T_path=mDataDirT[mFileCount];
                R_img=cv::imread(R_path);
                T_img=cv::imread(T_path,0);
                cv::cvtColor(R_img,R_img,cv::COLOR_RGB2BGR);
                mFileCount++;
                R_img=R_img*2/255.-1.0;
                T_img=T_img*2/255.-1.0;
                R_imgs.push_back(R_img);
                T_imgs.push_back(T_img);
            }

            std::vector<float> data_R(std::accumulate(mDimsR.d, mDimsR.d+ mDimsR.nbDims, 1, std::multiplies<int64_t>()));
            std::vector<float> data_T(std::accumulate(mDimsT.d, mDimsT.d + mDimsT.nbDims, 1, std::multiplies<int64_t>()));
            for(int batches=0;batches<mDimsR.d[0];batches++){
                imagetodata(R_imgs[batches],mDimsR.d[3],mDimsR.d[2],mDimsR.d[1],data_R,batches);
                imagetodata(T_imgs[batches],mDimsT.d[3],mDimsT.d[2],mDimsT.d[1],data_T,batches);
            }
            std::copy_n(data_R.data(), mDimsR.d[0] * mImageSizeR, getFileBatchR());
            std::copy_n(data_T.data(), mDimsT.d[0] * mImageSizeT, getFileBatchT());

        mFileBatchPos = 0;
        return true;
    }

    cv::Mat channelR[3],channelT[1];
    cv::Mat ImgR,ImgT;
    cv::Mat R_img,T_img;
    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};

    int mImageSizeR{0};
    int mImageSizeT{0};

    std::vector<float> mBatchR;         //!< Data for the batch
    std::vector<float> mBatchT;         //!< Data for the batch

    std::vector<float> mLabels;        //!< Labels for the batch
    std::vector<float> mFileBatchR;     //!< List of image files
    std::vector<float> mFileBatchT;     //!< List of image files

    std::vector<float> mFileLabels;    //!< List of label files
    std::string mPrefix;               //!< Batch file name prefix
    std::string mSuffix;               //!< Batch file name suffix
    nvinfer1::Dims mDimsR;              //!< Input dimensions
    nvinfer1::Dims mDimsT;              //!< Input dimensions

    std::string mListFile;             //!< File name of the list of image names
    std::vector<std::string> mDataDirR; //!< Directories where the files can be found
    std::vector<std::string> mDataDirT; //!< Directories where the files can be found

};

#endif // BATCHSTREAM_H
