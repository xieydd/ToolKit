#include "model.h"
#include <dirent.h>
#include <unistd.h>
#include <sys/time.h>

static inline void print_usage(char **argv)
{
    fprintf(stderr, "Usage %s [model dir] [h] [w] [c] [loop_count] [num_threads] [powersave] [cooling_down] [device]\n", argv[0]);
    exit(0);
}

static inline void getFiles(const char *basePath, std::vector<std::string> &files)
{
    char path[1000];
    struct dirent *dp;
    DIR *dir = opendir(basePath);

    // Unable to open directory stream
    if (!dir)
        return;

    while ((dp = readdir(dir)) != NULL)
    {
        if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0)
        {
            printf("%s\n", dp->d_name);

            // Construct new path from our base path
            strcpy(path, basePath);
            strcat(path, "/");
            strcat(path, dp->d_name);
            files.push_back(std::string(path));

            getFiles(path, files);
        }
    }

    closedir(dir);
}

static inline double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static inline std::string forwardType(MNNForwardType type)
{
    switch (type)
    {
    case MNN_FORWARD_CPU:
        return "CPU";
    case MNN_FORWARD_VULKAN:
        return "Vulkan";
    case MNN_FORWARD_OPENCL:
        return "OpenCL";
    case MNN_FORWARD_METAL:
        return "Metal";
    default:
        break;
    }
    return "N/A";
}

Model::Model(int argc, char **argv)
{
    if (argc >= 5)
    {
        model_dir = argv[1];
        getFiles(model_dir, files_name);
        height = atoi(argv[2]);
        width = atoi(argv[3]);
        channel = atoi(argv[4]);
    }
    else
    {
        print_usage(argv);
    }
    if (argc >= 6)
    {
        loop_count = atoi(argv[5]);
    }
    if (argc >= 7)
    {
        num_threads = atoi(argv[6]);
    }
    if (argc >= 8)
    {
        powersave = atoi(argv[7]);
    }
    if (argc >= 9)
    {
        cooling_down = atoi(argv[8]);
    }
    if (argc >= 10)
    {
        forward = static_cast<MNNForwardType>(atoi(argv[9]));
        device = std::string(argv[9]);
    }

    // NCNN
    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = false;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;
    ncnn::set_cpu_powersave(powersave);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stdout, "################ NCNN Option ##############\n");
    fprintf(stdout, "loop_count = %d\n", loop_count);
    fprintf(stdout, "num_threads = %d\n", num_threads);
    fprintf(stdout, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stdout, "cooling_down = %d\n", (int)(cooling_down != 0));
    fprintf(stdout, "############################################\n");
    fprintf(stdout, "\n");

    //MNN
    config.numThread = num_threads;
    config.type = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;
    fprintf(stdout, "################# MNN Config ###############\n");
    fprintf(stdout, "MNN Bankend = %s\n", forwardType(config.type).c_str());
    fprintf(stdout, "MNN precision = %d\n", precision);
    fprintf(stdout, "MNN num_threads = %d\n", num_threads);
    fprintf(stdout, "############################################\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "################# Tengine Config ###############\n");
    fprintf(stdout, "Tengine num_threads = %d\n", num_threads);
    fprintf(stdout, "Tengine device %s\n", device.c_str());
    fprintf(stdout, "############################################\n");
    fprintf(stdout, "\n");
}

int Model::benchmark_ncnn()
{
    fprintf(stdout, "NCNN Speed\n");
    input_mat = ncnn::Mat(height, width, channel);
    input_mat.fill(0.01f);
    std::string suffix = ".param";
    for (int i = 0; i < files_name.size(); i++)
    {
        if ((files_name[i].find(suffix)) == std::string::npos)
        {
            continue;
        }
        else
        {
            ncnn::Net net;
            net.opt = opt;
            net.load_param(files_name[i].c_str());
            DataReaderFromEmpty dr;
            net.load_model(dr);
            g_blob_pool_allocator.clear();
            g_workspace_pool_allocator.clear();
            if (cooling_down)
            {
                sleep(10);
            }
            ncnn::Mat out;
            // warm up
            for (int i = 0; i < warmup_loop_count; i++)
            {
                ncnn::Extractor ex = net.create_extractor();
                ex.input("data", input_mat);
                ex.extract("output", out);
            }

            double time_min = DBL_MAX;
            double time_max = -DBL_MAX;
            double time_avg = 0;

            for (int i = 0; i < loop_count; i++)
            {
                double start = get_current_time();

                {
                    ncnn::Extractor ex = net.create_extractor();
                    ex.input("data", input_mat);
                    ex.extract("output", out);
                }

                double end = get_current_time();

                double time = end - start;

                time_min = std::min(time_min, time);
                time_max = std::max(time_max, time);
                time_avg += time;
            }
            time_avg /= loop_count;

            size_t start = files_name[i].find_last_of('/') + 1;
            size_t end = files_name[i].find_last_of('.');
            std::string model_name = files_name[i].substr(start, end - start);
            fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", model_name.c_str(), time_min, time_max, time_avg);
        }
    }

    return 0;
}

int Model::benchmark_mnn()
{
    fprintf(stdout, "MNN Speed\n");
    std::string suffix = ".mnn";
    std::vector<int> dims{1, channel, height, width};

    for (int i = 0; i < files_name.size(); i++)
    {
        if ((files_name[i].find(suffix)) == std::string::npos)
        {
            continue;
        }
        else
        {
            auto revertor = std::unique_ptr<Revert>(new Revert(files_name[i].c_str()));
            revertor->initialize();
            auto modelBuffer = revertor->getBuffer();
            const auto bufferSize = revertor->getBufferSize();
            auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));

            revertor.reset();

            MNN::Session *session = net->createSession(config);
            net->releaseModel();
            MNN::Tensor *input = net->getSessionInput(session, NULL);
            //net->resizeTensor(input, dims);
            //net->resizeSession(session);
            const MNN::Backend *inBackend = net->getBackend(session, input);

            std::shared_ptr<MNN::Tensor> givenTensor(MNN::Tensor::createHostTensorFromDevice(input, false));

            auto outputTensor = net->getSessionOutput(session, NULL);
            std::shared_ptr<MNN::Tensor> expectTensor(MNN::Tensor::createHostTensorFromDevice(outputTensor, false));
            // Warming up...
            for (int i = 0; i < warmup_loop_count; ++i)
            {
                input->copyFromHostTensor(givenTensor.get());
                net->runSession(session);
                outputTensor->copyToHostTensor(expectTensor.get());
            }

            double time_min = DBL_MAX;
            double time_max = -DBL_MAX;
            double time_avg = 0;
            for (int round = 0; round < loop_count; round++)
            {
                double start = get_current_time();

                input->copyFromHostTensor(givenTensor.get());
                net->runSession(session);
                outputTensor->copyToHostTensor(expectTensor.get());

                double end = get_current_time();

                double time = end - start;

                time_min = std::min(time_min, time);
                time_max = std::max(time_max, time);
                time_avg += time;
            }
            time_avg /= loop_count;

            size_t start = files_name[i].find_last_of('/') + 1;
            size_t end = files_name[i].find_last_of('.');
            std::string model_name = files_name[i].substr(start, end - start);
            fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", model_name.c_str(), time_min, time_max, time_avg);
        }
    }
    return 0;
}

int Model::benchmark_tengine()
{
    fprintf(stdout, "Tengine Speed\n");
    input_tensor.create(width, height, channel);
    input_tensor.fill(0.01f);
    std::string suffix = ".tmfile";
    for (int i = 0; i < files_name.size(); i++)
    {
        if ((files_name[i].find(suffix)) == std::string::npos)
        {
            continue;
        }
        else
        {
            tengine::Net net;
            net.load_model(NULL, "tengine", files_name[i].c_str());
            //net.set_device(device);

            net.input_tensor("data", input_tensor);
            if (cooling_down)
            {
                sleep(10);
            }
            //tengine::Tensor output_tensor;
            // warm up
            for (int i = 0; i < warmup_loop_count; i++)
            {
                net.run();
            }

            double time_min = DBL_MAX;
            double time_max = -DBL_MAX;
            double time_avg = 0;

            for (int i = 0; i < loop_count; i++)
            {
                double start = get_current_time();

                {
                    net.run();
                    //net.extract_tensor("fc7", output_tensor);
                }

                double end = get_current_time();

                double time = end - start;

                time_min = std::min(time_min, time);
                time_max = std::max(time_max, time);
                time_avg += time;
            }
            time_avg /= loop_count;

            size_t start = files_name[i].find_last_of('/') + 1;
            size_t end = files_name[i].find_last_of('.');
            std::string model_name = files_name[i].substr(start, end - start);
            fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", model_name.c_str(), time_min, time_max, time_avg);
        }
    }

    return 0;
}

Model::~Model()
{
}
