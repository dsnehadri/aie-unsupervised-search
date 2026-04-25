// loads xclbin, starts aie graph, runs events through pl which bridges to aie

// compile with:
// g++ -std=c++17 -Wall -O2 -I$XILINX_XRT/include \
//       host_aie.cpp -L$XILINX_XRT/lib -lxrt_coreutil \
//       -o host_aie

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_graph.h>

constexpr int WORDS_PER_EVENT_IN = 72; // 60 jets + 12 mask
constexpr int WORDS_PER_EVENT_OUT = 3; // mse, crossed, latent

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << "<aie_stream.xclbin> <input.bin>\n";
        return 1;
    }

    const std::string xclbin_path = argv[1];
    const std::string input_path = argv[2];

    // load input events

    std::ifstream fin(input_path, std::ios::binary);
    if (!fin) {std::cerr << "Cannot open " << input_path << "\n"; return 1;}
    fin.seekg(0, std::ios::end);
    size_t file_bytes = fin.tellg();
    fin.seekg(0);

    size_t n_events = file_bytes / (WORDS_PER_EVENT_IN * sizeof(uint32_t));
    std::cout << "Loading " << n_events << " events\n";

    std::vector<uint32_t> in_data(n_events * WORDS_PER_EVENT_IN);
    fin.read(reinterpret_cast<char*>(in_data.data()), file_bytes);

    std::vector<uint32_t> out_data(n_events * WORDS_PER_EVENT_OUT, 0);

    // open device, load xclbin

    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_path);

    // start aie graph

    auto graph = xrt::graph(device, uuid, "aie_graph");
    graph.reset();
    graph.run(-1);
    std::cout << "AIE graph started \n";

    // allocate ddr buffers

    auto bo_in = xrt::bo(device, in_data.size() * sizeof(uint32_t),
                        xrt::bo::flags::normal, 0);

    auto bo_out = xrt::bo(device, out_data.size() * sizeof(uint32_t),
                        xrt::bo::flags::normal, 1);

    std::memcpy(bo_in.map<uint32_t*>(), in_data.data(),
                in_data.size() * sizeof(uint32_t));
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // run pl kernel
    auto kernel = xrt::kernel(device, uuid, "aie_stream_top");

    auto start = std::chrono::steady_clock::now();
    auto run = kernel(bo_in, bo_out, static_cast<int>(n_events));
    run.wait();
    auto end = std::chrono::steady_clock::now();

    // retrieve results
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(out_data.data(), bo_out.map<uint32_t*>(),
                out_data.size() * sizeof(uint32_t));

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_event_us = (total_ms * 1000.0) / n_events;

    std::cout << "Processed" << n_events << " events in "
            << total_ms << " ms ("
            << per_event_us << "us/event)\n";

    // print first few results

    std::cout << "First 3 events (mse, crossed, latent dist)\n";

    for (int ev = 0; ev < std::min<size_t>(3, n_events); ev++) {
        float mse = *reinterpret_cast<float*>(&out_data[ev * 3 + 0]);
        float crossed = *reinterpret_cast<float*>(&out_data[ev * 3 + 1]);
        float latent = *reinterpret_cast<float*>(&out_data[ev * 3 + 2]);
        std::cout << " ev " << ev << ": " << mse << ", " << crossed << ", " << latent << "\n"; 
    }

    // stop aie graph
    graph.end();

    //save output

    std::ofstream fout("output.bin", std::ios::binary);
    fout.write(reinterpret_cast<char*>(out_data.data()),
                out_data.size() * sizeof(uint32_t));


    return 0;
}


