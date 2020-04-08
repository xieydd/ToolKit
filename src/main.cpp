/*
 * @Author: xieydd
 * @since: 2020-04-07 14:08:24
 * @lastTime: 2020-04-07 16:47:58
 * @LastAuthor: Do not edit
 * @message: 
 */
// SYSTEM
#include "model.h"

int main(int argc, char **argv)
{
    Model model = Model(argc, argv);
    int result;
    result = model.benchmark_ncnn();
    result = model.benchmark_mnn();
    result = model.benchmark_tengine();
    return result;
}
