#include <iostream>
#include <vector>
#include <cmath>
#include "EasyBMP.h"

std::vector<unsigned char> loadGrayscale(const std::string& filename, int& w, int& h) {
    BMP img;
    if (!img.ReadFromFile(filename.c_str())) {
        throw std::runtime_error("Cannot open file");
    }

    w = img.TellWidth();
    h = img.TellHeight();

    std::vector<unsigned char> data(w * h);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            auto p = img.GetPixel(x, y);
            data[y * w + x] = (unsigned char)(0.299 * p.Red + 0.587 * p.Green + 0.114 * p.Blue);
        }
    }

    return data;
}

double computeMSE(const std::vector<unsigned char>& a,
    const std::vector<unsigned char>& b) {
    double mse = 0.0;
    int n = a.size();

    for (int i = 0; i < n; i++) {
        double diff = (double)a[i] - b[i];
        mse += diff * diff;
    }

    return mse / n;
}

double computePSNR(double mse) {
    if (mse == 0) return 100; // идеал
    return 10.0 * log10((255.0 * 255.0) / mse);
}

double computeRecovery(double mse) {
    return (1.0 - mse / (255.0 * 255.0)) * 100.0;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: metrics <original.bmp> <processed.bmp>\n";
        return 1;
    }

    int w1, h1, w2, h2;

    auto img1 = loadGrayscale(argv[1], w1, h1);
    auto img2 = loadGrayscale(argv[2], w2, h2);

    if (w1 != w2 || h1 != h2) {
        std::cerr << "Images must be same size\n";
        return 1;
    }

    double mse = computeMSE(img1, img2);
    double psnr = computePSNR(mse);
    double recovery = computeRecovery(mse);

    std::cout << "MSE: " << mse << std::endl;
    std::cout << "PSNR: " << psnr << " dB" << std::endl;
    std::cout << "Recovery: " << recovery << " %" << std::endl;

    return 0;
}