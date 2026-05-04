#include <iostream>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <xclbin>" << std::endl;
        return 1;
    }

    std::cout << "=== stub host ===" << std::endl;

    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(argv[1]);
    auto kernel = xrt::kernel(device, uuid, "stub_top",
                              xrt::kernel::cu_access_mode::exclusive);

    // small buffers, just enough for the 3-word write
    size_t in_size = 4 * sizeof(uint32_t);
    size_t out_size = 3 * sizeof(uint32_t);

    auto in_bo = xrt::bo(device, in_size, kernel.group_id(0));
    auto out_bo = xrt::bo(device, out_size, kernel.group_id(1));

    // initialize output to known sentinel so we can tell if kernel wrote
    auto out_map = out_bo.map<uint32_t*>();
    out_map[0] = 0xAAAAAAAA;
    out_map[1] = 0xAAAAAAAA;
    out_map[2] = 0xAAAAAAAA;
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "launching kernel..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    auto run = kernel(in_bo, out_bo, 1);

    // poll for 10s
    bool completed = false;
    for (int i = 0; i < 1000; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto st = run.state();
        if (st == ERT_CMD_STATE_COMPLETED) {
            std::cout << "kernel completed after " << (i * 10) << " ms" << std::endl;
            completed = true;
            break;
        }
        if (i % 100 == 0) {
            std::cout << "t=" << i/100 << "s state=" << st << std::endl;
        }
    }

    if (!completed) {
        std::cout << "TIMEOUT - kernel did not complete in 10s" << std::endl;
        std::cout << "AP_CTRL = 0x" << std::hex << kernel.read_register(0x00) << std::dec << std::endl;
        return 1;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    std::cout << "elapsed: " << elapsed_us << " us" << std::endl;

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    printf("out[0] = 0x%08x  (expect 0xdeadbeef)\n", out_map[0]);
    printf("out[1] = 0x%08x  (expect 0xcafebabe)\n", out_map[1]);
    printf("out[2] = 0x%08x  (expect 0x12345678)\n", out_map[2]);

    bool ok = (out_map[0] == 0xdeadbeef &&
               out_map[1] == 0xcafebabe &&
               out_map[2] == 0x12345678);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}