#include <thread>
#include <math.h>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
//#include <opencv2/imgcodecs.hpp> // Needed for imwrite/imread


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;


cv::Mat imfill(cv::Mat inmask);

//py::array_t<uint8_t> 
py::object process(py::array_t<uint8_t>& imgdata, float rel_th)
{
    cv::Mat outbin, stats, centroids, imbin, imin_bw, mroi;
    int nlabels,i,j,wsize,wstep,center_offset,yres,xi,yi,res_offset;
    double *outcentroid, *outbb;
    unsigned char *LUTdata;
    unsigned char r,g,b;
    cv::Scalar fv;
    
    double max_val,th;
    
    cv::MatIterator_<cv::Vec3b> it_24, end_24;
    cv::MatIterator_<ushort> it_16, end_16;
    cv::MatIterator_<uchar> it_8, it_roi_i,it_roi_f;
    cv::MatIterator_<double> it_focus;

    cv::setNumThreads(std::thread::hardware_concurrency());

    auto rows = imgdata.shape(0);
    auto cols = imgdata.shape(1);
    auto channels = imgdata.shape(2);
    auto type = CV_8UC3;

    // Transform input image to cv::Mat
    cv::Mat imin(rows, cols, type, (unsigned char*)imgdata.data());

    cv::cvtColor(imin, imin_bw, cv::COLOR_RGB2GRAY);
    
    wsize = 50;
    wstep = 6; //wstep cannot be bigger than wsize
    
    res_offset = wsize/wstep;
    center_offset = wsize/2;
       
    cv::Size in_size = imin.size();
    cv::Size aux_size;
    aux_size.height = (in_size.height/wstep)-res_offset;
    aux_size.width = (in_size.width/wstep)-res_offset;
    
    //Output focal metric image
    cv::Mat fmat = cv::Mat::Mat(aux_size,CV_64FC1);
    int iwidth = 0;

    
    cv::parallel_for_(cv::Range(0, aux_size.height*aux_size.width), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            cv::Scalar t_mean,t_std;
            double maxfm;
            int i = r / aux_size.width;
            int j = r % aux_size.width;
                        
            //Mat mroi = imin_bw(Rect(j*wstep,i*wstep,wsize,wsize));
            cv::Mat mroi = imin(cv::Rect(j*wstep, i*wstep, wsize, wsize));
            cv::meanStdDev(mroi, t_mean, t_std);
            maxfm = t_std[0];
            for(int i=1;i<3;i++)
            {
                if(maxfm<t_std[i]) maxfm=t_std[i];
            }
            fmat.ptr<double>(i)[j] = maxfm;
            //fmat.ptr<double>(i)[j] = (t_std[0]+t_std[1]+t_std[2])/3.0; //Try maximum of 3
        }
    });
    
    cv::minMaxLoc(fmat,NULL,&max_val,NULL,NULL);
    
    cv::threshold(fmat, imbin, max_val*rel_th, 0xFF,cv::THRESH_BINARY_INV);
    
    imbin.convertTo(imbin, CV_8U);
   
    fv = cv::mean(fmat, imbin);
	th = fv.val[0];

    // NOTE: Should return fmat if needed.

    cv::threshold(fmat, imbin, th, 0xFF,cv::THRESH_BINARY_INV);
    imbin.convertTo(imbin, CV_8U); //May cause weird looking imbin output
    //cv::imwrite( "imbin.jpg", imbin);
    
    //dilate(imbin,outbin, Mat(),Point(-1,-1),1);
    //imwrite( "imdilate1.jpg", outbin);
    //outbin = imfill(outbin); //This would delete small samples that does not touch image edges
    //imwrite( "imfill1.jpg", outbin);
    
    cv::bitwise_not(imbin,outbin);
    outbin = imfill(outbin);

    //imwrite( "imfill2.jpg", outbin);
    
    //dilate(outbin,outbin, Mat(),Point(-1,-1),3);
    
    cv::erode(outbin,outbin, cv::Mat(),cv::Point(-1,-1),10);
    //cv::imwrite( "imerode2.jpg", outbin);
    
    //Add margins for erode
    for (i = 0; i < outbin.cols; i++)
    {
        outbin.at<char>(0,i) = 0x00;
        outbin.at<char>(outbin.rows-1,i) = 0x00;
    }
    for (i = 0; i < outbin.rows; i++)
    {
        outbin.at<char>(i,0) = 0x00;
        outbin.at<char>(i,outbin.cols-1) = 0x00;
    }
    
    erode(outbin,outbin, cv::Mat(),cv::Point(-1,-1),4);
    outbin = imfill(outbin);
    
    ////////////// CONNECTED COMPONENT LABELING
    cv::Mat imlabel;
    nlabels = cv::connectedComponentsWithStats(outbin,imlabel,stats,centroids,8,CV_16U);
    
    if(nlabels < 2) 
    { //No leaf pixels found
        //py::print("No leaf pixels found. No mask.");
        return py::cast<py::none>(Py_None);
    }
    
    ushort big_l = (ushort) 1;
    for (i=1;i<nlabels;i++)
    {
        if(stats.at<int>(cv::Point(4, i)) > stats.at<int>(cv::Point(4, big_l)))
        {
            big_l = (ushort) i;
        }      
    }

    if(stats.at<int>(cv::Point(4, big_l)) < (outbin.rows * outbin.cols)*0.15) 
    {// Leaf area so small, skip
        //py::print("Leaf area so small! No mask.");
        return py::cast<py::none>(Py_None);
    }
    
    //Delete other objects, keep the biggest
    cv::parallel_for_(cv::Range(0, aux_size.height*aux_size.width), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            int i = r / aux_size.width;
            int j = r % aux_size.width;
            if(imlabel.ptr<uint16_t>(i)[j]== big_l) outbin.ptr<uchar>(i)[j] = 0xFF;
            else outbin.ptr<uchar>(i)[j] = 0;
        }
    });
    
    erode(outbin,outbin, cv::Mat(), cv::Point(-1,-1),10);
    resize(outbin,outbin,imin.size(), 0, 0, cv::INTER_NEAREST);
    
    // Create output image for Python OpenCV for Grayscale images
    py::array_t<uint8_t> output(py::buffer_info(outbin.data,
                                                sizeof(uint8_t), //itemsize
                                                py::format_descriptor<uint8_t>::format(),
                                                2, // ndim, 2-Grayscale, 3-RGB
                                                std::vector<py::ssize_t> {rows, cols}, // shape
                                                std::vector<size_t> {sizeof(uint8_t) * cols * 1, sizeof(uint8_t)} // strides
                                                )
                                );

    /*
    // Create output image for Python OpenCV for RGB images
    py::array_t<uint8_t> output0(py::buffer_info(out.data,
                                                sizeof(uint8_t), //itemsize
                                                py::format_descriptor<uint8_t>::format(),
                                                3, // ndim
                                                std::vector<py::ssize_t> {rows, cols , 3}, // shape
                                                std::vector<size_t> {sizeof(uint8_t) * cols * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)} // strides
                                                )
                                );
    */
    return output;
}

PYBIND11_MODULE(leaf_masking, m)
{
    m.doc() = "Leaf Masking algorithm";
    m.def("process", &process, "Process an input image to get bacground masked image.\nArguments: input image, relative threshold");

    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}





cv::Mat imfill(cv::Mat inmask)
{
    cv::Mat t_mask;
    inmask.copyTo(t_mask);

    for (int i = 0; i < t_mask.cols; i++)
    {
        if (t_mask.at<char>(0, i) == 0)
        {
            cv::floodFill(t_mask, cv::Point(i, 0), 255, 0, 10, 10);
        }   
        if (t_mask.at<char>(t_mask.rows-1, i) == 0)
        {
            cv::floodFill(t_mask, cv::Point(i, t_mask.rows-1), 255, 0, 10, 10);
        }
    }
    for (int i = 0; i < t_mask.rows; i++) {
        if (t_mask.at<char>(i, 0) == 0)
        {
            cv::floodFill(t_mask, cv::Point(0, i), 255, 0, 10, 10);
        }
        if (t_mask.at<char>(i, t_mask.cols-1) == 0)
        {
            cv::floodFill(t_mask, cv::Point(t_mask.cols-1, i), 255, 0, 10, 10);
        }
    }

    cv::Mat outmask;
    inmask.copyTo(outmask);
 
    for (int row = 0; row < t_mask.rows; ++row) {
        for (int col = 0; col < t_mask.cols; ++col) {
            if (t_mask.at<char>(row, col) == 0) {
                outmask.at<char>(row, col) = 0xFF;
            }           
        }
    }    
    
    return outmask;
}
