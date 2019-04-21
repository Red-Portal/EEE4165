
#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

float const pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286;

struct kernel_t
{
    std::vector<float> coeff;
    size_t m;
    size_t n;
};

template<typename Type>
inline Type
clamp(Type const& val, Type const& lo, Type const& hi) noexcept
{
    return std::min(std::max(val, lo), hi);
}

enum class pad_type { zero, ext };

inline cv::Vec3f
dot(std::vector<cv::Vec3f> const& x,
    std::vector<float> const& y)
{
    auto value = cv::Vec3f(0.0, 0.0, 0.0);
    size_t n = x.size();
    for(size_t c = 0; c < 3; ++c)
    {
        for(size_t i = 0; i < n; ++i)
        {
            value[c] += x[i][c] * y[i];
        }
    }
    return value;
}

inline cv::Vec3f
load_pixel(cv::Mat const& source,
           size_t i, size_t j)
{
    auto raw = source.at<cv::Vec3b>(i, j);
    cv::Vec3f result;
    for(size_t c = 0; c < static_cast<size_t>(source.channels()); ++c)
    {
        result[c] = static_cast<float>(raw[c]) / 255.0;
    }
    return result;
}

inline void
store_pixel(cv::Mat& source,
            size_t i, size_t j,
            cv::Vec3f value)
{
    cv::Vec3b result;
    for(size_t c = 0; c < static_cast<size_t>(source.channels()); ++c)
    {
        result[c] = cv::saturate_cast<unsigned char>(value[c] * 255) ;
    }
    source.at<cv::Vec3b>(i, j) = result;
}

template<pad_type Pad>
inline cv::Vec3f fetch_pad(cv::Mat const& src,
                           size_t i, size_t j,
                           size_t m, size_t n,
                           size_t off_m, size_t off_n);

template<>
inline cv::Vec3f
fetch_pad<pad_type::zero>(cv::Mat const& src,
                          size_t i, size_t j,
                          size_t m, size_t n,
                          size_t off_m, size_t off_n)
{
    if(i < off_m || i >= m + off_m
       || j < off_n || j >= n + off_n)
    {
        return cv::Vec3f(0.0, 0.0, 0.0);
    }
    else
        return load_pixel(src, i - off_m, j - off_n);
}

template<>
inline cv::Vec3f
fetch_pad<pad_type::ext>(cv::Mat const& src,
                         size_t i, size_t j,
                         size_t m, size_t n,
                         size_t off_m, size_t off_n)
{
    i = clamp(i, off_m, m - 1);
    j = clamp(j, off_n, n - 1);
    return load_pixel(src, i - off_m, j - off_n);
}

template<pad_type Pad>
inline cv::Mat
conv2d_impl(cv::Mat const& src,
            kernel_t const& kernel)
{
    size_t off_m  = kernel.m / 2;
    size_t off_n  = kernel.n / 2;
    size_t src_m  = src.rows;
    size_t src_n  = src.cols;
    auto dst = cv::Mat(src_m, src_n, CV_32FC3);
#pragma omp parallel for schedule(static) collapse(2) 
    for(size_t i = 0; i < src_m; ++i) {
        for(size_t j = 0; j < src_n; ++j)
        {
            auto buffer = std::vector<cv::Vec3f>(kernel.m * kernel.n);
            for(size_t u = 0; u < kernel.m; ++u) {
                for(size_t v = 0; v < kernel.n; ++v) {
                    buffer[u + v * kernel.m] = fetch_pad<Pad>(
                        src, i + u, j + v,
                        src_m, src_n,
                        off_m, off_n);
                }
            }
            dst.at<cv::Vec3f>(i, j) = dot(buffer, kernel.coeff);
        }
    }
    return dst;
}

inline cv::Mat
conv2d(cv::Mat const& src,
       kernel_t const& kernel,
       pad_type pad)
{
    switch(pad)
    {
    case pad_type::zero:
        return conv2d_impl<pad_type::zero>(src, kernel);
    case pad_type::ext:
        return conv2d_impl<pad_type::ext>(src, kernel);
    default:
        return cv::Mat();
    }
}

cv::Mat
quantize(cv::Mat const& src, bool bias)
{
    size_t src_m  = src.rows;
    size_t src_n  = src.cols;
    auto dst = cv::Mat(src_m, src_n, CV_8UC3);
    float peak;
    if(bias)
    {
        double lo_peak;
        double hi_peak;
        cv::minMaxLoc(src, &lo_peak, &hi_peak);
        peak = std::max(abs(lo_peak), abs(hi_peak));
    }

    for(size_t i = 0; i < src_m; ++i) {
        for(size_t j = 0; j < src_n; ++j){
            if(bias)
            {
                auto value = src.at<cv::Vec3f>(i, j);
                value /= peak;
                value += cv::Vec3f(1.0, 1.0, 1.0);
                store_pixel(dst, i, j, value / 2);
            }
            else
            {
                store_pixel(dst, i, j, src.at<cv::Vec3f>(i, j));
            }
        }
    }
    return dst;
}

namespace kernel 
{
    inline kernel_t
    gaussian_filter(float sigma, size_t n)
    {
        kernel_t kernel;
        float sigma2 = sigma * sigma;
        float origin = floor(n / 2);
        auto pdf =
            [sigma2, origin](float x, float y)
            {
                x = x - origin;
                y = y - origin;
                return exp((x * x + y * y) / (-2 * sigma2));
            } ;

        kernel.coeff = std::vector<float>(n * n);
        kernel.m = n;
        kernel.n = n;

        for(size_t x = 0; x < n; ++x) {
            for(size_t y = 0; y < n; ++y) {
                kernel.coeff[x + y * n] = pdf(x, y);
            }
        }

        auto normalize = std::accumulate(kernel.coeff.begin(),
                                         kernel.coeff.end(),
                                         0.0);
        std::transform(kernel.coeff.begin(),
                       kernel.coeff.end(),
                       kernel.coeff.begin(),
                       [normalize](float elem){
                           return elem / normalize;
                       });
        return kernel;
    }

    inline kernel_t
    laplacian_filter(bool is_4_direction)
    {
        kernel_t kernel;
        kernel.m = 3;
        kernel.n = 3;
        if(is_4_direction)
        {
            kernel.coeff = std::vector<float>({
                    0, -1, 0, 
                    -1, 4, -1, 
                    0, -1, 0, 
                });
        }
        else
        {
            kernel.coeff = std::vector<float>({
                    -1, -1, -1, 
                    -1,  8, -1, 
                    -1, -1, -1, 
                });
        }
        return kernel;
    }

    inline kernel_t
    sobel_filter(bool is_x_direction)
    {
        kernel_t kernel;
        kernel.m = 3;
        kernel.n = 3;
        if(is_x_direction)
        {
            kernel.coeff = std::vector<float>({
                    -1, -2, -1,
                    0, 0, 0, 
                    1, 2, 1
                });
        }
        else
        {
            kernel.coeff = std::vector<float>({
                    -1, 0, 1,
                    -2, 0, 2, 
                    -1, 0, 1
                });
        }
        return kernel;
    }

    inline kernel_t
    box_filter(size_t n)
    {
        kernel_t kernel;
        kernel.coeff = std::vector<float>(n * n, 1.0 / (n * n));
        kernel.m = n;
        kernel.n = n;
        return kernel;
    }
}

inline void
apply_filter(kernel_t const& kernel,
             pad_type pad,
             std::string const& filename,
             std::string const& window_name,
             std::string const& save_name,
             bool bias)
{
    using namespace std::literals::string_literals;
    std::cout << "   < " << save_name << " >"
              << std::endl;

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

    auto smooth = cv::Mat();
    auto start = std::chrono::steady_clock::now();
    {
        smooth = quantize(conv2d(image, kernel, pad), bias);
    }
    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::duration<double, std::ratio<1,1>>>(stop - start).count();
    std::cout << "Throughput: " <<  1.0 / duration << " images/sec\n"
              << "Throughput: " <<  (image.rows * image.cols) / (duration) << " pixels/sec"
              << std::endl;

    cv::imshow(window_name, image);
    cv::waitKey(0);
    cv::imshow(window_name, smooth);
    cv::waitKey(0);
    cv::imwrite(save_name + ".png"s, smooth);
    std::cout << std::endl;
}

inline void
edge_enhance(kernel_t const& kernel,
             pad_type pad,
             std::string const& filename,
             std::string const& window_name,
             std::string const& save_name)
{
    using namespace std::literals::string_literals;
    std::cout << "   < " << save_name << " >"
              << std::endl;

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

    auto enhanced = cv::Mat();
    auto start = std::chrono::steady_clock::now();
    {
        auto edge_response = conv2d(image, kernel, pad);
        for(int i = 0; i < image.rows; ++i)
        {
            for(int j = 0; j < image.cols; ++j)
            {
                edge_response.at<cv::Vec3f>(i,j)
                    = load_pixel(image, i, j) + edge_response.at<cv::Vec3f>(i,j);
            }
        }
        enhanced = quantize(edge_response, false);
    }
    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::duration<double, std::ratio<1,1>>>(stop - start).count();
    std::cout << "Throughput: " <<  1.0 / duration << " images/sec\n"
              << "Throughput: " <<  (image.rows * image.cols) / (duration) << " pixels/sec"
              << std::endl;

    cv::imshow(window_name, image);
    cv::waitKey(0);
    cv::imshow(window_name, enhanced);
    cv::waitKey(0);
    cv::imwrite(save_name + ".png"s, enhanced);
    std::cout << std::endl;
}

int main()
{
    auto window_name = std::string("OpenCV");
    cv::namedWindow(window_name);

    auto gaussian = kernel::gaussian_filter(1, 7);
    apply_filter(gaussian, pad_type::ext,  "specification/Test_pattern.tif", window_name, "gaussian1_extpad", false);
    apply_filter(gaussian, pad_type::zero, "specification/Test_pattern.tif", window_name, "gaussian1_zeropad", false);

    gaussian = kernel::gaussian_filter(3.5, 21);
    apply_filter(gaussian, pad_type::ext,  "specification/Test_pattern.tif", window_name, "gaussian2_extpad", false);
    apply_filter(gaussian, pad_type::zero, "specification/Test_pattern.tif", window_name, "gaussian2_zeropad", false);

    gaussian = kernel::gaussian_filter(7, 43);
    apply_filter(gaussian, pad_type::ext,  "specification/Test_pattern.tif", window_name, "gaussian3_extpad", false);
    apply_filter(gaussian, pad_type::zero, "specification/Test_pattern.tif", window_name, "gaussian3_zeropad", false);

    auto box = kernel::box_filter(3);
    apply_filter(box, pad_type::ext,  "specification/Test_pattern.tif", window_name, "box1_extpad", false);
    apply_filter(box, pad_type::zero, "specification/Test_pattern.tif", window_name, "box1_zeropad", false);

    box = kernel::box_filter(13);
    apply_filter(box, pad_type::ext,  "specification/Test_pattern.tif", window_name, "box2_extpad", false);
    apply_filter(box, pad_type::zero, "specification/Test_pattern.tif", window_name, "box2_zeropad", false);

    box = kernel::box_filter(25);
    apply_filter(box, pad_type::ext,  "specification/Test_pattern.tif", window_name, "box3_extpad", false);
    apply_filter(box, pad_type::zero, "specification/Test_pattern.tif", window_name, "box3_zeropad", false);

    auto sobel_x  = kernel::sobel_filter(true);
    apply_filter(sobel_x,  pad_type::zero, "specification/Boy.tif", window_name, "sobelx_zeropad", true);
    apply_filter(sobel_x,  pad_type::ext,  "specification/Boy.tif", window_name, "sobelx_extpad", true);

    auto sobel_y  = kernel::sobel_filter(false);
    apply_filter(sobel_y,  pad_type::zero, "specification/Boy.tif", window_name, "sobely_zeropad", true);
    apply_filter(sobel_y,  pad_type::ext,  "specification/Boy.tif", window_name, "sobely_extpad", true);

    auto lapla_4  = kernel::laplacian_filter(true);
    apply_filter(lapla_4,  pad_type::zero, "specification/Boy.tif", window_name, "laplace4_zeropad", true);
    apply_filter(lapla_4,  pad_type::ext,  "specification/Boy.tif", window_name, "laplace4_extpad", true);

    auto lapla_8  = kernel::laplacian_filter(false);
    apply_filter(lapla_8,  pad_type::zero, "specification/Boy.tif", window_name, "laplace8_zeropad", true);
    apply_filter(lapla_8,  pad_type::ext,  "specification/Boy.tif", window_name, "laplace8_extpad", true);

    edge_enhance(lapla_4,  pad_type::zero, "specification/Boy.tif", window_name, "laplace4_en_zeropad");
    edge_enhance(lapla_4,  pad_type::ext,  "specification/Boy.tif", window_name, "laplace4_en_extpad");

    edge_enhance(lapla_8,  pad_type::zero, "specification/Boy.tif", window_name, "laplace8_en_zeropad");
    edge_enhance(lapla_8,  pad_type::ext,  "specification/Boy.tif", window_name, "laplace8_en_extpad");

    cv::destroyWindow(window_name);
}
