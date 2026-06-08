#ifndef POOLINGLAYER_HPP
#define POOLINGLAYER_HPP
#include <vector>

class PoolingLayer {
    public:
    PoolingLayer(int inputHeight, int inputWidth, int poolSize, int stride, int padding);

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) const;

    private:
    int inputHeight;
    int inputWidth;
    int poolSize;
    int stride;
    int padding;
};
#endif // POOLINGLAYER_HPP