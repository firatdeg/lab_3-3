#include "grayscale_image.h"
#include <iostream>
#include <omp.h>
#include "intermediate_image.h"


void GrayscaleImage::convert_bitmap(BitmapImage& bitmap) {
    height = bitmap.height;
    width = bitmap.width;
    
    std::uint32_t total_pixels = height * width;
    pixels.resize(total_pixels);

    #pragma omp parallel for default(none) shared(bitmap, pixels, total_pixels)
    for (std::uint32_t i = 0; i < total_pixels; ++i) {
        const auto& pixel = bitmap.pixels[i]; 
        
        double L = 0.2126 * pixel.get_red_channel() + 
                   0.7152 * pixel.get_green_channel() + 
                   0.0722 * pixel.get_blue_channel() + 0.5;
                   
        pixels[i] = static_cast<std::uint8_t>(L);
    }
}

void GrayscaleImage::convert_intermediate_image(IntermediateImage& image) {
    height = image.height;
    width = image.width;
    
    std::uint32_t total_pixels = height * width;
    pixels.resize(total_pixels);

    if (total_pixels == 0) return;

    image.update_min_pixel_value();
    image.update_max_pixel_value();

    double min_val = image.min_pixel_value;
    double max_val = image.max_pixel_value;

    if (min_val >= 0.0 && max_val <= 255.0) {
        min_val = 0.0;
        max_val = 255.0;
    }

    double range = max_val - min_val;
    if (range == 0.0) {
        range = 1.0; 
    }

    #pragma omp parallel for default(none) shared(image, pixels, total_pixels, min_val, range)
    for (std::uint32_t i = 0; i < total_pixels; ++i) {
        double v = image.pixels[i];
        
        double g = ((v - min_val) / range) * 255.0;
        
        if (g < 0.0) g = 0.0;
        if (g > 255.0) g = 255.0;
        
        pixels[i] = static_cast<std::uint8_t>(g);
    }
}