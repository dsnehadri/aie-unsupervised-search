/// host.cpp - XRT host application for passwd-abc on vck190

// compile on vck190: g++ -o host host.cpp -lxrt_coreutil -I$XILINX_XRT/include -L$XILINX_XRT/lib
// usage: ./host pl_stream.xsa [input.bin] [n_events]

// if no input file is given, runs a single hardcoded test event

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <thread>

// XRT includes

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

static const int WORDS_PER_EVENT_IN = 72;
static const int WORDS_PER_EVENT_OUT = 3;

// reinterpret uint32 bits as float

static float bits_to_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

// load binary input_file: raw 32-bit words, WORDS_PER_EVENT_IN per event

static std::vector<uint32_t> load_input(const std::string &path, int n_events) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::cerr << "ERROR: cannot open input file: " << path << std::endl;
        exit(1);
    }

    size_t total_words = n_events * WORDS_PER_EVENT_IN;
    std::vector<uint32_t> data(total_words);
    fin.read(reinterpret_cast<char*>(data.data()), total_words * sizeof(uint32_t));

    if (!fin) {
        std::cerr << "ERROR: Input file too short. Expected " << total_words * 4 << " bytes for " << n_events << " events." << std::endl;
        exit(1);
    }

    return data;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <xsa_file> [input.bin] [n_events]" << std::endl;
        return 1;
    }

    std::string xsa_path = argv[1];
    std::string input_path = (argc >= 3) ? argv[2] : "";
    int n_events = (argc >= 4) ? std::atoi(argv[3]) : 1;

    std::cout << "=== passwd-abc vck190 host ===" << std::endl;
    std::cout << "xsa:    " << xsa_path << std::endl;
    std::cout << "events:    " << n_events << std::endl;

    // open device and load bitsream
    std::cout << "opening device..." << std::endl;
    auto device = xrt::device(0);

    std::cout << "loading bitstream: " << xsa_path << std::endl;
    auto uuid = device.load_xclbin(xsa_path);

    std::cout << "creating kernel handle..." << std::endl;
    auto kernel = xrt::kernel(device, uuid, "passwd_stream_top",
                          xrt::kernel::cu_access_mode::exclusive);

    // allocate device buffers

    size_t in_size = n_events * WORDS_PER_EVENT_IN * sizeof(uint32_t);
    size_t out_size = n_events * WORDS_PER_EVENT_OUT * sizeof(uint32_t);

    std::cout << "allocating buffers: in = " << in_size << " bytes, out =" << out_size << " bytes" << std::endl;

    auto in_bo = xrt::bo(device, in_size, kernel.group_id(0));
    auto out_bo = xrt::bo(device, out_size, kernel.group_id(1));

    // prepare input data

    auto in_map = in_bo.map<uint32_t*>();

    if (!input_path.empty()) {
        //load from binary file
        std::cout << "loading input from: " << input_path << std::endl;
        auto input_data = load_input(input_path, n_events);
        std::memcpy(in_map, input_data.data(), in_size);
    } else {
        // zero-fill if no input given
        std::cout << "no input file - running smoke test with zeros" << std::endl;
        std::memset(in_map, 0, in_size);
        // set all mask bits to 1
        for (int ev = 0; ev < n_events; ev++) {
            for (int i = 0; i < 12; i ++) {
                in_map[ev * WORDS_PER_EVENT_IN + 60 + i] = 1;
            }
        }
    }

    // transfer input to device

    std::cout << "transferring input to device" << std::endl;
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // run kernel

    std::cout << "running kernel (" << n_events << "events)..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();


    auto run = kernel(in_bo, out_bo, n_events);

    for (int i = 0; i < 30; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto st = run.state();
        uint32_t val = kernel.read_register(0x30);
        std::cout << "t=" << i << "s  state=" << st 
                << "  debug_stage=" << val << std::endl;
        std::cout.flush();
        if (st == ERT_CMD_STATE_COMPLETED) break;
    }

    run.wait();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "kernel finished" << std::endl;
    std::cout << "total time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "per event: " << elapsed_ms / n_events << " ms" << std::endl;
    std::cout << "throughput: " << (n_events / elapsed_ms) * 1000.0 << " events/sec" << std::endl;

    // transfer output from device

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto out_map = out_bo.map<uint32_t*>();

    // print results

    std::cout << std::endl;
    std::cout << "===results===" << std::endl;
    printf("%-8s    %12s    %12s %12s\n", "Event", "MSE loss", "MSE crossed", "latent dist");
    printf("%-8s    %12s    %12s %12s\n", "-----", "-----", "-----", "-----");

    for (int ev = 0; ev < n_events; ev++) {
        int offset = ev * WORDS_PER_EVENT_OUT;
        float mse = bits_to_float(out_map[offset + 0]);
        float crossed = bits_to_float(out_map[offset + 1]);
        float latent = bits_to_float(out_map[offset + 2]);

        printf("%-8d %12.6f %12.6f %12.6f\n", ev, mse, crossed, latent);
    }

    {
        std::string out_path = "output.bin";
        std::ofstream fout(out_path, std::ios::binary);
        fout.write(reinterpret_cast<char*>(out_map), out_size);
        std::cout << std::endl << "output saved to: " << out_path << std::endl;
    }





}