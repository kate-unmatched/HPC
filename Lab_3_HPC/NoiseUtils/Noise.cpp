#include <iostream>
#include <random>
#include <string>
#include "EasyBMP.h"

void addSaltPepperNoise(BMP& image, double noisePercent) {
    int width = image.TellWidth();
    int height = image.TellHeight();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r = dis(gen);

            if (r < noisePercent / 2.0) {
                // соль (белый)
                RGBApixel pixel;
                pixel.Red = 255;
                pixel.Green = 255;
                pixel.Blue = 255;
                pixel.Alpha = 0;
                image.SetPixel(x, y, pixel);
            }
            else if (r < noisePercent) {
                // перец (чёрный)
                RGBApixel pixel;
                pixel.Red = 0;
                pixel.Green = 0;
                pixel.Blue = 0;
                pixel.Alpha = 0;
                image.SetPixel(x, y, pixel);
            }
        }
    }
}

//int main(int argc, char* argv[]) {
//    if (argc < 4) {
//        std::cout << "Usage: noise <input.bmp> <output.bmp> <noise_percent>\n";
//        return 1;
//    }
//
//    std::string inputFile = argv[1];
//    std::string outputFile = argv[2];
//    double noisePercent = std::stod(argv[3]); // например 0.1 = 10%
//
//    BMP image;
//
//    if (!image.ReadFromFile(inputFile.c_str())) {
//        std::cerr << "Cannot open input file\n";
//        return 1;
//    }
//
//    addSaltPepperNoise(image, noisePercent);
//
//    if (!image.WriteToFile(outputFile.c_str())) {
//        std::cerr << "Cannot save output file\n";
//        return 1;
//    }
//
//    std::cout << "Noise added: " << noisePercent * 100 << "%\n";
//    std::cout << "Saved to: " << outputFile << "\n";
//
//    return 0;
//}