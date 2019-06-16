
#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <string>

template<typename Type>
inline Type
clamp(Type const& val, Type const& lo, Type const& hi) noexcept
{
    return std::min(std::max(val, lo), hi);
}

/* Routine for calculating and rendering Histogram image */
inline cv::Mat
histogram(cv::Mat const& image)
{
    float range[] = { 0, 255 } ;
    float const* histrange = { range };
    int histsize = 256;
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1,
                 &histsize, &histrange, uniform, accumulate);
    size_t hist_w = 512;
    size_t hist_h = 400;
    size_t bin_w = cvRound( static_cast<double>(hist_w)/histsize );
    auto hist_image = cv::Mat(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
    cv::normalize(hist, hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
    for(int i = 1; i < histsize; ++i)
    {
        cv::line(hist_image, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
                 cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                 cv::Scalar(255), 2, 8, 0  );
    }
    return hist_image;
}


/* Routing loading pixel value from image applying border extension padding */
inline uint8_t
fetch_pad(cv::Mat const& src,
          size_t i, size_t j,
          size_t m, size_t n,
          size_t off_m, size_t off_n)
{
    i = clamp(i, off_m, m);
    j = clamp(j, off_n, n);
    return src.at<uint8_t>(i - off_m, j - off_n);
}

/* Image frequency domain representation calculation (2D Fourier transform) */
inline std::tuple<cv::Mat, size_t, size_t>
image_fft(cv::Mat const& src)
{
    cv::Mat padded;
    cv::Mat src_real;
    cv::Mat cplx_image;
    cv::Mat dst;
    size_t m     = cv::getOptimalDFTSize( src.rows );
    size_t n     = cv::getOptimalDFTSize( src.cols );
    size_t pad_m = m - src.rows;
    size_t pad_n = n - src.cols;
    src.convertTo(src_real, CV_64F);
    for(size_t i = 0; i < m - pad_m; ++i) {
        for(size_t j = 0; j < n - pad_n; ++j) {
            double coeff = (i + j) % 2 == 0 ? 1 : -1;
            src_real.at<double>(i, j) = src_real.at<double>(i, j) * coeff;
        }
    }
    cv::copyMakeBorder(src_real, padded, 0, pad_m, 0, pad_n, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[2] = { src_real, cv::Mat::zeros(src_real.size(), CV_64F) };
    cv::merge(planes, 2, cplx_image);
    cv::dft(cplx_image, cplx_image, cv::DFT_SCALE);
    return {cplx_image, pad_m, pad_n};
}

/* Image spatial domain calculation (2D inverse Fourier transform) */
inline cv::Mat
image_ifft(cv::Mat const& spectrum)
{
    cv::Mat inverse;
    cv::Mat planes[2] = {cv::Mat(inverse.size(), CV_64F),
                         cv::Mat(inverse.size(), CV_64F)};
    cv::idft(spectrum, inverse, cv::DFT_REAL_OUTPUT);
    cv::split(inverse, planes);
    inverse = planes[0];
    inverse = cv::abs(inverse);
    inverse.convertTo(inverse, CV_8U);
    return inverse;
}

/* Image frequency component power spectral density calculation */
inline cv::Mat
image_spectrum(cv::Mat const& cplx)
{
    cv::Mat planes[2] = {cv::Mat(cplx.size(), CV_64F),
                         cv::Mat(cplx.size(), CV_64F)};
    cv::Mat mag;
    cv::split(cplx * cplx.total(), planes);
    cv::magnitude(planes[0], planes[1], mag);
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX, CV_8U);
    return mag;
}

/* Find median of array */
template<typename Type, typename Iter>
inline Type
median(Iter begin, Iter end, size_t n)
{
    auto nth = begin + floor(n / 2.0);
    std::nth_element(begin, nth, end);
    return *nth;
}

/* 2D Median filtering */
inline cv::Mat
median_filter(cv::Mat const& src, size_t window_size)
{
    size_t m            = src.rows;
    size_t n            = src.cols;
    size_t offset_m     = floor(window_size / 2.0);
    size_t offset_n     = offset_m;
    size_t window_elems = window_size * window_size;
    auto dest = cv::Mat(m, n, CV_8UC1);
#pragma omp parallel for schedule(static) collapse(2)
    for(size_t i = 0; i < m; ++i) {
        for(size_t j = 0; j < n; ++j) {
            auto buffer = std::vector<uint8_t>(window_elems);
            for(size_t u = 0; u < window_size; ++u) {
                for(size_t v = 0; v < window_size; ++v) {
                    buffer[u + v * window_size] =
                        fetch_pad(src, i + u, j + v, m, n, offset_m, offset_n);
                }
            }
            auto pix = median<uint8_t>(buffer.begin(), buffer.end(), window_elems);
            dest.at<uint8_t>(i, j) = pix;
        }
    }
    return dest;
}

/* Contraharmonic mean filtering */
inline cv::Mat
contraharmonic_mean_filter(cv::Mat const& src, double Q)
{
    size_t window_size  = 3;
    size_t m            = src.rows;
    size_t n            = src.cols;
    size_t offset_m     = floor(window_size / 2.0);
    size_t offset_n     = offset_m;
    size_t window_elems = window_size * window_size;
    auto dest = cv::Mat(m, n, CV_8UC1);
#pragma omp parallel for schedule(static) collapse(2)
    for(size_t i = 0; i < m; ++i) {
        for(size_t j = 0; j < n; ++j) {
            auto buffer = std::vector<uint8_t>(window_elems);
            for(size_t u = 0; u < window_size; ++u) {
                for(size_t v = 0; v < window_size; ++v) {
                    buffer[u + v * window_size] =
                        fetch_pad(src, i + u, j + v, m, n, offset_m, offset_n);
                }
            }
            double num = 0;
            double den = 0;
            for(size_t i = 0; i < window_elems; ++i)
            {
                //auto prod = pow(static_cast<double>(buffer[i]), Q);
                den += pow(static_cast<double>(buffer[i]), Q);//prod;
                num += pow(static_cast<double>(buffer[i]), Q+1);//prod * buffer[i];
            }
            dest.at<uint8_t>(i, j) = cv::saturate_cast<uint8_t>(num / den);
        }
    }
    return dest;
}

/* Adaptive median filtering */
inline cv::Mat
adaptive_median_filter(cv::Mat const& src, size_t initial_window_size)
{
    size_t window_max   = 16;
    size_t m            = src.rows;
    size_t n            = src.cols;
    auto dest = cv::Mat(m, n, CV_8UC1);
#pragma omp parallel for schedule(static) collapse(2)
    for(size_t i = 0; i < m; ++i) {
        for(size_t j = 0; j < n; ++j) {
            auto window_size = initial_window_size;
            uint8_t pix      = 0;
            auto zxy         = src.at<uint8_t>(i, j);
            while(true)
            {
                /* Load patch */
                size_t offset       = floor(window_size / 2.0);
                size_t window_elems = window_size * window_size;
                auto buffer = std::vector<uint8_t>(window_elems);
                for(size_t u = 0; u < window_size; ++u) {
                    for(size_t v = 0; v < window_size; ++v) {
                        buffer[u + v * window_size] =
                            fetch_pad(src, i + u, j + v, m, n, offset, offset);
                    }
                }

                /* Compute patch statistics */
                auto median_iter = buffer.begin() + floor(window_elems / 2.0);
                //std::sort(buffer.begin(), buffer.end());
                std::nth_element(buffer.begin(), median_iter, buffer.end());
                auto zmed = *median_iter;
                std::nth_element(buffer.begin(), buffer.begin(), median_iter);
                auto zmin = *buffer.begin();
                std::nth_element(median_iter, buffer.begin() + window_elems - 1, buffer.end());
                auto zmax = *(buffer.begin() + window_elems - 1);

                /* Termination condition */
                if(zmin < zmed && zmed < zmax)
                {
                    pix = zmin < zxy && zxy < zmax ? zxy : zmed;
                    break;
                }
                window_size += 2;
                if(window_size >= window_max)
                {
                    pix = zmed;
                    break;
                }
            }
            dest.at<uint8_t>(i, j) = pix;
        }
    }
    return dest;
}

template<typename NoiseDist, typename Rng>
inline cv::Mat
apply_noise(cv::Mat const& mat,
            Rng& rng,
            NoiseDist noise)
{
    size_t m = mat.rows;
    size_t n = mat.cols;
    auto dest = cv::Mat(m, n, CV_8UC1);
    for(size_t i = 0; i < m; ++i) {
        for(size_t j = 0; j < n; ++j) {
            int pix_noise = noise(rng);
            auto pix = mat.at<uint8_t>(i ,j);
            pix = cv::saturate_cast<uint8_t>(static_cast<int>(pix) + pix_noise);
            dest.at<uint8_t>(i, j) = pix;
        }
    }
    return dest;
}

/* Compute PSNR of two images */
inline std::pair<double, double>
psnr(cv::Mat const& x,
     cv::Mat const& y)
{
    cv::Mat delta;
    cv::absdiff(x, y, delta);

    delta.convertTo(delta, CV_32F);
    delta = delta.mul(delta);

    cv::Scalar s = cv::sum(delta);
    double sse = s.val[0] + s.val[1] + s.val[2];

    if( sse <= 1e-10)
        return {0, 0};
    else
    {
        double mse = sse /(double)(x.channels() * x.total());
        double psnr = 10.0*log10((255*255)/mse);
        return {psnr, mse};
    }
}


/* Turbulence image degradiation filter as gaussian bluring */
inline cv::Mat
degradation(size_t m, size_t n, double k)
{
    auto filter = cv::Mat(m, n, CV_64F);
    auto H = [k](double u, double v){
                 return exp(-k * pow(u * u + v * v, 5.0 / 6));
             };

#pragma omp parallel for schedule(static) collapse(2)
    for(size_t i = 0; i < m; ++i) {
        for(size_t j = 0; j < n; ++j) {
            filter.at<double>(i, j) =
                H(static_cast<double>(i) - static_cast<double>(m - 1) / 2,
                  static_cast<double>(j) - static_cast<double>(n - 1) / 2);
        }
    }
    cv::Mat temp[] = {filter, cv::Mat::zeros(filter.size(), CV_64FC1)};
    cv::Mat result;
    cv::merge(temp, 2, result);
    return result;
}

/* Apply turbulence degradation using convolution */
inline cv::Mat
apply_degradation(double k,
                  std::string const& filename,
                  std::string const& window_name,
                  std::string const& save_name)
{
    using namespace std::literals::string_literals;
    std::cout << "   < " << save_name << " >"
              << std::endl;

    auto image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

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

    auto [cplx, pad_m, pad_n] = image_fft(image);
    auto specdb     = image_spectrum(cplx);
    auto filter     = degradation(cplx.rows, cplx.cols, k);
    cv::mulSpectrums(cplx, filter, cplx, 0);

    auto filtereddb = image_spectrum(cplx);
    auto filterdb   = image_spectrum(filter);
    auto degraded   = image_ifft(cplx);

    auto [snr, mse] = psnr(image, degraded);

    std::cout << "MSE : " << mse << std::endl;
    std::cout << "PSNR: " << snr << " (dB)\n" << std::endl;
    
    cv::imshow(window_name, image);
    cv::waitKey(0);
    cv::imshow(window_name, specdb);
    cv::waitKey(0);
    cv::imshow(window_name, filterdb);
    cv::waitKey(0);
    cv::imshow(window_name, filtereddb);
    cv::waitKey(0);
    cv::imshow(window_name, degraded);
    cv::waitKey(0);
    cv::imwrite(save_name + "_spec.png", specdb);
    cv::imwrite(save_name + "_spec_degrad.png", filterdb);
    cv::imwrite(save_name + "_spec_degraded.png", filtereddb);
    cv::imwrite(save_name + "_degraded.png", degraded);
    return degraded;
}

/* Inverse filtering using degradation model */
inline void
deconv_inverse(double k,
               cv::Mat const& image,
               cv::Mat const& degraded,
               std::string const& window_name,
               std::string const& save_name)
{
    using namespace std::literals::string_literals;
    std::cout << "   < " << save_name << " >"
              << std::endl;

    auto filtered_image = cv::Mat();
    auto filtereddb     = cv::Mat();
    auto filterdb       = cv::Mat();
    auto start = std::chrono::steady_clock::now();
    {
        auto [cplx, pad_m, pad_n] = image_fft(degraded);
        
        auto filter     = degradation(cplx.rows, cplx.cols, k);

        /* Compute inverse filter */
        size_t m = filter.rows;
        size_t n = filter.cols;
        for(size_t i = 0; i < m; ++i) {
            for(size_t j = 0; j < n; ++j) {
                double u = static_cast<double>(i) - static_cast<double>(m-1) / 2;
                double v = static_cast<double>(j) - static_cast<double>(n-1) / 2;
                if(sqrt(u * u + v * v) < 64)
                {
                    filter.at<double>(i, j, 0) = 1.0 / filter.at<double>(i, j, 0);
                }
                else
                {
                    filter.at<double>(i, j, 0) = 1.0;
                }
            }
        }

        /* Compute inverse filter */
        cv::mulSpectrums(cplx, filter, cplx, 0);
        filtereddb     = image_spectrum(cplx);
        filterdb       = image_spectrum(filter);
        filtered_image = image_ifft(cplx);
    }
    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::duration<double, std::ratio<1,1>>>(stop - start).count();
    std::cout << "Throughput: " <<  1.0 / duration << " images/sec\n"
              << "Throughput: " <<  (filtered_image.rows * filtered_image.cols)
        / (duration) << " pixels/sec"
              << std::endl;

    auto [snr, mse] = psnr(image, filtered_image);

    std::cout << "MSE : " << mse << std::endl;
    std::cout << "PSNR: " << snr << " (dB)\n" << std::endl;
    
    cv::imshow(window_name, filterdb);
    cv::waitKey(0);
    cv::imshow(window_name, filtereddb);
    cv::waitKey(0);
    cv::imshow(window_name, filtered_image);
    cv::waitKey(0);
    cv::imwrite(save_name + "_sepc_undegrad.png", filterdb);
    cv::imwrite(save_name + "_spec_undegraded.png", filtereddb);
    cv::imwrite(save_name + "_undegraded.png", filtered_image);
}

/* Weiner filtering using turbulence model */
inline void
deconv_weiner(double k,
              cv::Mat const& image,
              cv::Mat const& degraded,
              std::string const& window_name,
              std::string const& save_name)
{
    using namespace std::literals::string_literals;
    std::cout << "   < " << save_name << " >"
              << std::endl;

    auto filtered_image = cv::Mat();
    auto filtereddb     = cv::Mat();
    auto filterdb       = cv::Mat();
    auto start = std::chrono::steady_clock::now();
    {
        auto [cplx, pad_m, pad_n] = image_fft(degraded);
        
        auto filter = degradation(cplx.rows, cplx.cols, k);

        /* Compute weiner filter from model */
        size_t m = filter.rows;
        size_t n = filter.cols;
        for(size_t i = 0; i < m; ++i) {
            for(size_t j = 0; j < n; ++j) {
                auto h = filter.at<double>(i, j, 0);
                filter.at<double>(i, j, 0) = (h*h)/((h*h)+ 0.001 )/h;
            }
        }

        /* Apply weiner filter */
        cv::mulSpectrums(cplx, filter, cplx, 0);
        filtereddb     = image_spectrum(cplx);
        filterdb       = image_spectrum(filter);
        filtered_image = image_ifft(cplx);
    }
    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::duration<double, std::ratio<1,1>>>(stop - start).count();
    std::cout << "Throughput: " <<  1.0 / duration << " images/sec\n"
              << "Throughput: " <<  (filtered_image.rows * filtered_image.cols)
        / (duration) << " pixels/sec"
              << std::endl;

    auto [snr, mse] = psnr(image, filtered_image);

    std::cout << "MSE : " << mse << std::endl;
    std::cout << "PSNR: " << snr << " (dB)\n" << std::endl;
    
    cv::imshow(window_name, filterdb);
    cv::waitKey(0);
    cv::imshow(window_name, filtereddb);
    cv::waitKey(0);
    cv::imshow(window_name, filtered_image);
    cv::waitKey(0);
    cv::imwrite(save_name + "_sepc_undegrad.png", filterdb);
    cv::imwrite(save_name + "_spec_undegraded.png", filtereddb);
    cv::imwrite(save_name + "_undegraded.png", filtered_image);
}

/* Apply noise to image according to noise distribution */
template<typename NoiseDist,
         typename Rng>
inline void
apply_noise(Rng& rng,
            NoiseDist noise,
            std::string const& filename,
            std::string const& window_name,
            std::string const& save_name)
{
    using namespace std::literals::string_literals;
    std::cout << "   < " << save_name << " >"
              << std::endl;

    auto image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

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

    cv::imshow(window_name, image);
    cv::waitKey(0);

    auto noisy_image = apply_noise(image, rng, noise);
    auto hist = histogram(noisy_image);

    auto [snr, mse] = psnr(image, noisy_image);
    std::cout << "MSE : " << mse << std::endl;
    std::cout << "PSNR: " << snr << " (dB)\n" << std::endl;

    cv::imshow(window_name, noisy_image);
    cv::waitKey(0);
    cv::imshow(window_name, hist);
    cv::waitKey(0);
    cv::imwrite(save_name + ".png", noisy_image);
    cv::imwrite(save_name + "_hist.png", hist);
}

template<typename Filter>
inline cv::Mat
apply_filter(Filter filter,
             cv::Mat const& image,
             cv::Mat const& noisy_image,
             std::string const& window_name,
             std::string const& save_name)
{
    using namespace std::literals::string_literals;
    std::cout << "   < " << save_name << " >"
              << std::endl;

    cv::imshow(window_name, noisy_image);
    cv::waitKey(0);

    auto filtered = cv::Mat();
    auto start = std::chrono::steady_clock::now();
    {
        filtered = filter(noisy_image);
    }
    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::duration<double, std::ratio<1,1>>>(stop - start).count();
    std::cout << "Throughput: " <<  1.0 / duration << " images/sec\n"
              << "Throughput: " <<  (image.rows * image.cols) / (duration) << " pixels/sec"
              << std::endl;

    auto [snr, mse] = psnr(image, filtered);
    std::cout << "MSE : " << mse << std::endl;
    std::cout << "PSNR: " << snr << " (dB)\n" << std::endl;

    cv::imshow(window_name, filtered);
    cv::waitKey(0);
    cv::imwrite(save_name + ".png", filtered);
    return filtered;
}

int main()
{
    auto window_name = std::string("OpenCV");
    cv::namedWindow(window_name);

    std::mt19937 rng;

    auto unifo_noise = std::uniform_int_distribution<int>(5, 25);
    apply_noise(rng, unifo_noise, "specification/Pattern.tif", window_name, "uniform");

    auto dist = std::normal_distribution<double>(20, 20);
    auto gauss_noise =
        [&dist](std::mt19937& rng){
            return static_cast<int>(round(dist(rng)));
        };
    apply_noise(rng, gauss_noise, "specification/Pattern.tif", window_name, "gaussian");

    auto salt_pepper_dist = std::bernoulli_distribution(0.2);
    auto salt_dist        = std::bernoulli_distribution(0.5);
    auto salt_pepper =
        [&salt_pepper_dist, &salt_dist](std::mt19937& rng){
            if(salt_pepper_dist(rng))
                return salt_dist(rng) ? 255 : -255;
            else
                return 0;
        };
    apply_noise(
        rng, salt_pepper, "specification/Pattern.tif", window_name, "saltpepper");


    auto image       = cv::imread("specification/Ckt_board/Ckt_board.tif", cv::IMREAD_GRAYSCALE);
    auto noisy_image = cv::imread("specification/Ckt_board/Ckt_board_salt&pepper_0.3.tif", cv::IMREAD_GRAYSCALE);

    apply_filter([](cv::Mat const& image){ return median_filter(image, 3); } ,
                 image, noisy_image, window_name, "median_3");

    auto filtered = apply_filter([](cv::Mat const& image){ return median_filter(image, 5); } ,
                                 image, noisy_image, window_name, "median_5");

    apply_filter([](cv::Mat const& image){ return contraharmonic_mean_filter(image, 1.5); } ,
                 image, noisy_image, window_name, "contraharmonic_1p5");

    apply_filter([](cv::Mat const& image){ return contraharmonic_mean_filter(image, 0); } ,
                 image, noisy_image, window_name, "contraharmonic_0");

    apply_filter([](cv::Mat const& image){ return contraharmonic_mean_filter(image, -1.5); } ,
                 image, noisy_image, window_name, "contraharmonic_m1p5");

    apply_filter([](cv::Mat const& image){ return adaptive_median_filter(image, 7); } ,
                 image, noisy_image, window_name, "apaptive_7");

    apply_filter([](cv::Mat const& image){ return adaptive_median_filter(image, 11); } ,
                 image, noisy_image, window_name, "adaptive_11");

    filtered = apply_filter([](cv::Mat const& image){ return median_filter(image, 5); } ,
                            image, filtered, window_name, "median_twice");

    filtered = apply_filter([](cv::Mat const& image){ return median_filter(image, 5); } ,
                            image, filtered, window_name, "median_third");

    filtered = apply_filter([](cv::Mat const& image){ return median_filter(image, 5); } ,
                            image, filtered, window_name, "median_fourth");

    auto p001   = apply_degradation(0.001, "specification/Sogang.tif", window_name, "sogang_0p001");
    auto p0025  = apply_degradation(0.0025, "specification/Sogang.tif", window_name, "sogang_0p0025");
    auto p00025 = apply_degradation(0.00025, "specification/Sogang.tif", window_name, "sogang_0p00025");

    image = cv::imread("specification/Sogang.tif", cv::IMREAD_GRAYSCALE);
    deconv_inverse(0.00025, image, p00025, window_name, "sogang_0p00025");
    deconv_inverse(0.0025,  image, p0025,  window_name, "sogang_0p0025");
    deconv_inverse(0.001,   image, p001,   window_name, "sogang_0p001");

    deconv_weiner(0.00025, image, p00025, window_name, "sogang_wienerp00025");
    deconv_weiner(0.0025,  image, p0025, window_name, "sogang_wienerp0025");
    deconv_weiner(0.001,   image, p001, window_name, "sogang_wienerp001");

    cv::destroyWindow(window_name);
}
