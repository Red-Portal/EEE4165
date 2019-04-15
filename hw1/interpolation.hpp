
#ifndef _INTERPOLATE_HPP_
#define _INTERPOLATE_HPP_

#include <opencv2/core/core.hpp>
#include <algorithm>

namespace interpolation
{
    template<typename Type>
    inline Type
    clamp(Type const& val, Type const& lo, Type const& hi) noexcept
    {
        return std::min(std::max(val, lo), hi);
    }

    enum class method
    {
        nearest,
        bilinear,
        bicubic
    };

    inline float
    l1dist(std::pair<float, float> const& x,
           std::pair<float, float> const& y) noexcept
    {
        auto x_delta = x.first - y.first;
        auto y_delta = x.second - y.second;
        return abs(x_delta) + abs(y_delta);
    }

    inline cv::Vec3f
    fetch_pixel(cv::Mat const& source,
                size_t i, size_t j)
    {
        auto raw = source.at<cv::Vec3b>(
            clamp(static_cast<float>(i), 0.0f,
                  static_cast<float>(source.rows - 1)),
            clamp(static_cast<float>(j), 0.0f,
                  static_cast<float>(source.cols - 1)));
        cv::Vec3f result;
        for(size_t c = 0; c < static_cast<size_t>(source.channels()); ++c)
        {
            result[c] = static_cast<float>(raw[c]) / 255.0;
        }
        return result;
    }

    inline cv::Vec3b
    quantize_pixel(cv::Mat const& source,
                   cv::Vec3f const& value)
    {
        auto result = cv::Vec3b();
        for(size_t c = 0; c < static_cast<size_t>(source.channels()); ++c)
        {
            result[c] = cv::saturate_cast<unsigned char>(value[c] * 255.0);
        }
        return result;
    }

    inline cv::Vec3b 
    nearest_neighbor(cv::Mat const& source,
                     std::pair<float, float> const& coord)
    {
        float i = coord.first;
        float j = coord.second;
        float i_c = ceil(i);
        float i_f = floor(i);
        float j_c = ceil(j);
        float j_f = floor(j);
        auto upper_left  = std::make_pair(i_c, j_f);
        auto upper_right = std::make_pair(i_c, j_c);
        auto lower_left  = std::make_pair(i_f, j_f);
        auto lower_right = std::make_pair(i_f, j_c);

        cv::Vec3f pixel;
        float closest_dist = std::numeric_limits<float>::max();

        auto compare =
            [&](std::pair<float, float> const& src_pix){
                float dist = l1dist(src_pix, coord);
                if(dist < closest_dist)
                {
                    closest_dist = dist;
                    pixel = fetch_pixel(source, src_pix.first, src_pix.second);
                }
            };
        compare(upper_right);
        compare(upper_left);
        compare(lower_right);
        compare(lower_left);
        return quantize_pixel(source, pixel);
    }


    inline cv::Vec3b 
    bilinear(cv::Mat const& source,
             std::pair<float, float> const& coord)
    {
        float i = coord.first;
        float j = coord.second;
        float i_c = ceil(i);
        float i_f = floor(i);
        float j_c = ceil(j);
        float j_f = floor(j);
        float i_norm = i_c - i;
        float j_norm = j_c - j;

        auto upper_right = fetch_pixel(source, i_c, j_c);
        auto upper_left  = fetch_pixel(source, i_c, j_f);
        auto lower_right = fetch_pixel(source, i_f, j_c);
        auto lower_left  = fetch_pixel(source, i_f, j_f);

        auto upper  = ((upper_left - upper_right) * j_norm) + upper_right;
        auto lower  = ((lower_left - lower_right) * j_norm) + lower_right;
        auto result = ((lower - upper) * i_norm) + upper;
        return quantize_pixel(source, result);
    }

    inline cv::Vec3b 
    bicubic(cv::Mat const& source,
            std::pair<float, float> const& coord)
    {
        float i = coord.first;
        float j = coord.second;
        float i_c = ceil(i);
        float i_f = floor(i);
        float j_c = ceil(j);
        float j_f = floor(j);

        auto p00 = fetch_pixel(source, i_c+1, j_c+1);
        auto p01 = fetch_pixel(source, i_c+1, j_c);
        auto p02 = fetch_pixel(source, i_c+1, j_f);
        auto p03 = fetch_pixel(source, i_c+1, j_f-1);

        auto p10 = fetch_pixel(source, i_c,   j_c+1);
        auto p11 = fetch_pixel(source, i_c,   j_c);
        auto p12 = fetch_pixel(source, i_c,   j_f);
        auto p13 = fetch_pixel(source, i_c,   j_f-1);

        auto p20 = fetch_pixel(source, i_f,   j_c+1);
        auto p21 = fetch_pixel(source, i_f,   j_c);
        auto p22 = fetch_pixel(source, i_f,   j_f);
        auto p23 = fetch_pixel(source, i_f,   j_f-1);

        auto p30 = fetch_pixel(source, i_f-1, j_c+1);
        auto p31 = fetch_pixel(source, i_f-1, j_c);
        auto p32 = fetch_pixel(source, i_f-1, j_f);
        auto p33 = fetch_pixel(source, i_f-1, j_f-1);

        auto curve_fit =
            [&](double x,
                cv::Vec3f const& p0,
                cv::Vec3f const& p1,
                cv::Vec3f const& p2,
                cv::Vec3f const& p3)
            {
                cv::Vec3f result;
                for(size_t c = 0; c < static_cast<size_t>(source.channels()); ++c)
                {
                    float alpha = (p0[c]/-2.0f) + ((3.0f/2)*p1[c]) - ((3.0f/2)*p2[c]) + (p3[c]/2.0f);
                    float beta  = p0[c] - ((5.0f/2)*p1[c]) + (2.0f*p2[c]) - (p3[c]/2.0f);
                    float ceta  = (p0[c]/-2.0f) + (p2[c]/2.0f);
                    result[c] = (((alpha*x) + beta)*x + ceta)*x + p1[c];
                }
                return result;
            };

        float i_norm = i_c - i;
        float j_norm = j_c - j;

        auto row0 = curve_fit(j_norm, p00, p01, p02, p03);
        auto row1 = curve_fit(j_norm, p10, p11, p12, p13);
        auto row2 = curve_fit(j_norm, p20, p21, p22, p23);
        auto row3 = curve_fit(j_norm, p30, p31, p32, p33);
        auto column = curve_fit(i_norm, row0, row1, row2, row3);
        return quantize_pixel(source, column);
    }
}

#endif
