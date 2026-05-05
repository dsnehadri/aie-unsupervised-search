// stub_ctrl_host.cpp - minimal XRT host for control-only stub
// compile: g++ -std=c++17 -O2 -I/usr/include/xrt stub_ctrl_host.cpp -L/usr/lib -lxrt_coreutil -lpthread -luuid -o stub_ctrl_host
// usage: ./stub_ctrl_host stub_ctrl.xclbin

#include <iostream>
#include <chrono>
#include <thread>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <xclbin>" << std::endl;
        return 1;
    }

    std::cout << "=== stub_ctrl host ===" << std::endl;

    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(argv[1]);
    auto kernel = xrt::kernel(device, uuid, "stub_ctrl",
                              xrt::kernel::cu_access_mode::exclusive);

    std::cout << "launching kernel (no buffers, single scalar arg)..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    auto run = kernel(0);  // n_events = 0, unused

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
        std::cout << "=> CONTROL-PATH ISSUE: ap_done is not being asserted" << std::endl;
        std::cout << "   even with no AXI memory transactions involved." << std::endl;
        return 1;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    std::cout << "elapsed: " << elapsed_us << " us" << std::endl;
    std::cout << "=> CONTROL PATH WORKS. Issue is specifically in m_axi/DDR." << std::endl;
    return 0;
}