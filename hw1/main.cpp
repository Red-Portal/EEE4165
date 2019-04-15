
#include <chrono>
#include <exception>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "interpolation.hpp"
#include "affine_transform.hpp"

inline double
psnr(cv::Mat const& source,
     cv::Mat const& processed)
{
    auto s1 = cv::Mat();
    cv::absdiff(source, processed, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    auto s = cv::sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];
    double snr = 0;
    
    if( sse <= 1e-10)
        return snr;

    double mse = sse / static_cast<double>(source.channels() * source.total());
    snr = 10.0 * log10( (255 * 255) / mse);
    return snr;
}


double
rmse(cv::Mat const& delta)
{
    auto delta_32f = delta.clone();
    delta_32f.convertTo(delta_32f, CV_32FC3);
    auto squared = delta_32f.mul(delta_32f);
    double sum = 0.0;
    for(size_t i = 0; i < static_cast<size_t>(delta.rows); ++i) {
        for(size_t j = 0; j < static_cast<size_t>(delta.cols); ++j) {
            for(size_t c = 0; c < static_cast<size_t>(delta.channels()); ++c) {
                sum += delta_32f.at<cv::Vec3f>(i, j)[c];
            }
        }
    }
    return sum / (delta.rows * delta.cols);
}

inline void
implementation1(std::string const& window_name,
                interpolation::method mode,
                std::string const& save_name)
{
    using namespace std::literals::string_literals;
    std::cout << "   < " << save_name << " >"
              << std::endl;

    auto filename = std::string("specification/Chronometer.tif");
    auto image = cv::imread(filename);
    if(image.empty())
    {
        throw std::runtime_error("could not open or find the image!");
    }
    else
    {
        std::cout << "successfully loaded image " << filename << '\n'
                  << "dimension: [" << image.rows << ',' << image.cols << ']'
                  << std::endl;
    }
    auto affine_shrink = dip::static_matrix<float, 3, 3>({{2.0, 0.0, 0.0},
                                                          {0.0, 2.0, 0.0},
                                                          {0.0, 0.0, 1.0}});
    auto affine_grow   = dip::static_matrix<float, 3, 3>({{1.0/2, 0.0,   0.0},
                                                          {0.0,   1.0/2, 0.0},
                                                          {0.0,   0.0,   1.0}});

    auto transformed = dip::affine_transform(
        affine_shrink, image,
        std::make_pair<size_t, size_t>(image.rows / 2,
                                       image.cols / 2), mode);

    {
        
        auto start = std::chrono::steady_clock::now();
        transformed = dip::affine_transform(
            affine_grow, transformed,
            std::make_pair<size_t, size_t>(image.rows,
                                           image.cols), mode);
        auto stop = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1,1>>>(stop - start).count();
        std::cout << "Throughput: " <<  1.0 / duration << " images/sec\n"
                  << "Throughput: " <<  (image.rows * image.cols) / (duration) << " pixels/sec"
                  << std::endl;
    }

    auto diff = cv::Mat();
    cv::absdiff(transformed, image, diff);
    double error = psnr(image, transformed);
    std::cout << "Signal to Noise Ratio: " << error
              << std::endl;

    cv::imshow(window_name, transformed);
    cv::waitKey(0);
    cv::imshow(window_name, diff);
    cv::waitKey(0);
    cv::imwrite(save_name + "_impl1.png"s, transformed);
    cv::imwrite("diff_"s + save_name + "_impl1.png"s, diff);
    std::cout << std::endl;
}

inline void
implementation2(std::string const& window_name,
                interpolation::method mode,
                std::string const& save_name)
{
    using namespace std::literals::string_literals;
    std::cout << "   < " << save_name << " >"
              << std::endl;

    auto filename = std::string("specification/Right_arrow.tif");
    auto image = cv::imread(filename);
    if(image.empty())
    {
        throw std::runtime_error("Could not open or find the image!");
    }
    else
    {
        std::cout << "Successfully loaded image " << filename << '\n'
                  << "dimension: [" << image.rows << ',' << image.cols << ']'
                  << std::endl;
    }
    auto rotate_translate = dip::rotation_transform(15, image.rows, image.cols, true);
    rotate_translate(0, 2) = -10;
    rotate_translate(1, 2) = -20;
    auto transformed = dip::affine_transform(
        rotate_translate, image,
        std::make_pair<size_t, size_t>(image.rows, image.cols), mode);

    cv::imshow(window_name, transformed);
    cv::waitKey(0);
    cv::imwrite(save_name + "_impl2.png", transformed);
    std::cout << std::endl;
}

int main()
{
    auto window_name = std::string("OpenCV");
    cv::namedWindow(window_name);
    implementation1(window_name, interpolation::method::nearest,  "nearest");
    implementation1(window_name, interpolation::method::bilinear, "bilinear");
    implementation1(window_name, interpolation::method::bicubic,  "bicubic");

    implementation2(window_name, interpolation::method::nearest,  "nearest");
    implementation2(window_name, interpolation::method::bilinear, "bilinear");
    implementation2(window_name, interpolation::method::bicubic,  "bicubic");
    cv::destroyWindow(window_name);
}
