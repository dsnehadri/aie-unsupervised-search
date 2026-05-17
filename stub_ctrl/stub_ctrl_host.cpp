// stub_ctrl_host.cpp
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
    std::cout.flush();

    std::cout << "[1] opening device..." << std::endl; std::cout.flush();
    auto device = xrt::device(0);
    std::cout << "[1] OK" << std::endl; std::cout.flush();

    std::cout << "[2] loading xclbin..." << std::endl; std::cout.flush();
    auto uuid = device.load_xclbin(argv[1]);
    std::cout << "[2] OK" << std::endl; std::cout.flush();

    std::cout << "[3] creating kernel handle..." << std::endl; std::cout.flush();
    auto kernel = xrt::kernel(device, uuid, "stub_ctrl");
    std::cout << "[3] OK" << std::endl; std::cout.flush();

    std::cout << "[3a] sleeping before launch..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "[4] launching kernel..." << std::endl; std::cout.flush();
    std::cout << "[4a] creating run object..." << std::endl;
    xrt::run run(kernel);

    std::cout << "[4b] setting n_events..." << std::endl;
    run.set_arg(0, 0);

    std::cout << "[4c] starting kernel..." << std::endl;
    run.start();

    std::cout << "[4d] start returned..." << std::endl;
    std::cout << "[4] OK (launched)" << std::endl; std::cout.flush();

    std::cout << "[5] polling state..." << std::endl; std::cout.flush();
    bool completed = false;
    
    for (int i = 0; i < 1000; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << "about to do run.state()..." << std::endl; std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto st = run.state();
        std::cout << "finished run.state()..." << std::endl; std::cout.flush();
        if (st == ERT_CMD_STATE_COMPLETED) {
            std::cout << "[5] kernel completed after " << (i * 10) << " ms" << std::endl;
            completed = true;
            break;
        }
        if (i % 100 == 0) {
            auto ctrl = kernel.read_register(0x00);
		    std::cout << " t=" << i/100 << "s state=" << st << " AP_CTRL=0x" << std::hex << ctrl << std::dec << std::endl;
        }
    }

    // for (int i = 0; i < 1000; i++) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //     std::cout << "about to do run.state()..." << std::endl; std::cout.flush();
    //     auto st = run.state();
    //     std::cout << "finished run.state()..." << std::endl; std::cout.flush();
    //     if (st == ERT_CMD_STATE_COMPLETED) {
    //         std::cout << "[5] kernel completed after " << (i * 10) << " ms" << std::endl;
    //         completed = true;
    //         break;
    //     }
    //     if (i % 100 == 0) {
    //         std::cout << "  t=" << i/100 << "s state=" << st << std::endl;
    //         std::cout.flush();
    //     }
    // }

    if (!completed) {
        std::cout << "[5] TIMEOUT - 10s elapsed, no completion" << std::endl;
        std::cout << "AP_CTRL = 0x" << std::hex << kernel.read_register(0x00) << std::dec << std::endl;
        return 1;
    }
    return 0;
}