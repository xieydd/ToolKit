/*
 * @Author: xieydd
 * @since: 2020-04-07 23:23:24
 * @lastTime: 2020-04-08 16:06:02
 * @LastAuthor: Do not edit
 * @message: Inference Model Definition
 */
// SYSTEM
#include <stdio.h>
#include <vector>
#include <string>

// NCNN
#include "net.h"
#include "datareader.h"
#include "cpu.h"
#include <float.h>

// MNN
#include "core/Backend.hpp"
#include <Interpreter.hpp>
#include <MNNDefine.h>
#include <Tensor.hpp>
#include "revertMNNModel.hpp"

// Tengine
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "tengine_cpp_api.h"
#include "common_util.hpp"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char *format, void *p) const { return 0; }
    virtual size_t read(void *buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

class Model
{
public:
    Model(int argc, char **argv);
    ~Model();
    int benchmark_ncnn();
    int benchmark_mnn();
    int benchmark_tengine();

public:
    int height;
    int width;
    int channel;
    const char *model_dir;
    std::vector<std::string> files_name;

    // NCNN
    int loop_count = 6;
    int num_threads = ncnn::get_cpu_count();
    int powersave = 0;
    int cooling_down = 1;
    int warmup_loop_count = 8;

    // MNN
    int precision = 2;
    int forward = MNN_FORWARD_CPU;

private:
    // MNN
    MNN::ScheduleConfig config;

    // NCNN
    ncnn::Mat input_mat;
    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;
    ncnn::Option opt;

    // Tengine
    tengine::Tensor input_tensor;
    std::string device = "0";
    char *cpu_list_str = nullptr;
};
