#include <filesystem>
#include <stdexcept>

#include "test.h"
#include "io/image_parser.h"
#include "grayscale_image.h"
#include "image/bitmap_image.h"
#include "image/pixel.h"
#include <cmath>



TEST_F(DevTest, cpu_grayscale) {
    // define test image
    BitmapImage bitmap(2,2);

    Pixel<std::uint8_t> px1(134,161,12);
    Pixel<std::uint8_t> px2(12,61,227);
    Pixel<std::uint8_t> px3(112,200,173);
    Pixel<std::uint8_t> px4(0,255,113);

    bitmap.set_pixel(0,0, px1);
    bitmap.set_pixel(0,1, px2);
    bitmap.set_pixel(1,0, px3);
    bitmap.set_pixel(1,1, px4);

    // define expected results (no rounding)
    std::uint8_t ex1 = 0.2126*134 + 0.7152*161 + 0.0722*12 + 0.5;
    std::uint8_t ex2 = 0.2126*12 + 0.7152*61 + 0.0722*227 + 0.5;
    std::uint8_t ex3 = 0.2126*112 + 0.7152*200 + 0.0722*173 + 0.5;
    std::uint8_t ex4 = 0.2126*0 + 0.7152*255 + 0.0722*113 + 0.5;
    
    // define expected results (with rounding)
    std::uint8_t ex1_round = round(0.2126*134 + 0.7152*161 + 0.0722*12 + 0.5);
    std::uint8_t ex2_round = round(0.2126*12 + 0.7152*61 + 0.0722*227 + 0.5);
    std::uint8_t ex3_round = round(0.2126*112 + 0.7152*200 + 0.0722*173 + 0.5);
    std::uint8_t ex4_round = round(0.2126*0 + 0.7152*255 + 0.0722*113 + 0.5);

    GrayscaleImage image;
    image.convert_bitmap(bitmap);

    ASSERT_EQ(image.width, bitmap.width);
    ASSERT_EQ(image.height, bitmap.height);

    // check results (no rounding)
    bool results_correct_no_rounding = true;
    results_correct_no_rounding =  results_correct_no_rounding && (image.pixels[0] == ex1);
    results_correct_no_rounding =  results_correct_no_rounding && (image.pixels[1] == ex2);
    results_correct_no_rounding =  results_correct_no_rounding && (image.pixels[2] == ex3);
    results_correct_no_rounding =  results_correct_no_rounding && (image.pixels[3] == ex4);

    // check results (with rounding)
    bool results_correct_with_rounding = true;
    results_correct_with_rounding =  results_correct_with_rounding && (image.pixels[0] == ex1_round);
    results_correct_with_rounding =  results_correct_with_rounding && (image.pixels[1] == ex2_round);
    results_correct_with_rounding =  results_correct_with_rounding && (image.pixels[2] == ex3_round);
    results_correct_with_rounding =  results_correct_with_rounding && (image.pixels[3] == ex4_round);

    ASSERT_TRUE(results_correct_no_rounding || results_correct_with_rounding);
}