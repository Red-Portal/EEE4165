
#ifndef _AFFINE_TRANSFORM_HPP_
#define _AFFINE_TRANSFORM_HPP_

#include "blaze/Blaze.h"
#include "interpolation.hpp"
#include <limits>

namespace dip
{
    template<typename Type, size_t M, size_t N>
    using static_matrix = blaze::StaticMatrix<Type, M, N, blaze::columnMajor>;

    template<typename Type, size_t N>
    using static_vector = blaze::StaticVector<Type, N>;

    inline static_matrix<float, 3, 3>
    rotation_transform(double angle,
                       size_t M, size_t N,
                       bool crop) noexcept
    {
        double const pi = 3.141592;
        float rad = angle / 180.0 * pi;
        // inverse mapping matrix
        float cosi = cos((-1) * rad);
        float sine = sin((-1) * rad);
        auto T = static_matrix<float, 3, 3>(
            {{cosi,      sine, 0.0f},
             {-1 * sine, cosi, 0.0f},
             {0.0f,      0.0f, 1.0f}});

        if(!crop)
        { // computing the required positive translation to prevent cropping
            auto coord_matrix = static_matrix<float, 3, 3>(
                {{static_cast<float>(M), 0.0f,                  static_cast<float>(M)},
                 {0.0f,                  static_cast<float>(N), static_cast<float>(N)},
                 {1.0f,                  1.0f,                  1.0f                 }});
            auto trans_coord  = blaze::evaluate(T * coord_matrix);
            for(size_t row = 0; row < 2; ++row)
            {
                for(auto const coord : blaze::row(trans_coord, row))
                {
                    if(coord < 0)
                        T(row, 2) = std::max(abs(coord), T(row, 2));
                }
            }
        }
        return T;
    }

    inline cv::Vec3b
    interpolate(cv::Mat const& source,
                std::pair<float, float> const& coord,
                interpolation::method mode)
    {
        if(mode == interpolation::method::nearest)
        {
            return interpolation::nearest_neighbor(source, coord); 
        }
        else if(mode == interpolation::method::bicubic)
        {
            return interpolation::bicubic(source, coord); 
        }
        else
        {
            return interpolation::bilinear(source, coord); 
        }
    }

    inline cv::Mat
    affine_transform(static_matrix<float, 3, 3> const& transform,
                     cv::Mat const& image,
                     std::pair<size_t, size_t> canvas_size,
                     interpolation::method mode)
    {
        auto dst = cv::Mat(
            cv::Mat::zeros(canvas_size.first,
                           canvas_size.second,
                           image.type()));

        size_t dst_M = dst.rows;
        size_t dst_N = dst.cols;
        auto dst_coord = static_vector<float, 3>(1.0);

        for(size_t i = 0; i < dst_M; ++i)
        {
            for(size_t j = 0; j < dst_N; ++j)
            {
                dst_coord[0] = i;
                dst_coord[1] = j;
                auto src_coord = transform * dst_coord;
                auto coord = std::make_pair<float, float>(
                    src_coord[0], src_coord[1]);
                dst.at<cv::Vec3b>(i, j) = interpolate(image, coord, mode);
            }
        }
        return dst;
    }

    inline cv::Mat
    affine_transform(static_matrix<float, 3, 3> const& transform,
                     cv::Mat const& image,
                     interpolation::method mode)
    {
        auto canvas = std::make_pair<size_t, size_t>(
            image.rows, image.cols);
        return affine_transform(transform, image, canvas, mode);
    }
}

#endif
