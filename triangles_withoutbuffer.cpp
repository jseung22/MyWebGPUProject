//// draw_instanced_grid_no_buffers.cpp
//#include <iostream>
//#include <chrono>
//
//#include <GLFW/glfw3.h>
//#if defined(__EMSCRIPTEN__)
//#include <emscripten/emscripten.h>
//#endif
//#include <dawn/webgpu_cpp_print.h>
//#include <webgpu/webgpu_cpp.h>
//#include <webgpu/webgpu_glfw.h>
//
//wgpu::Instance instance;
//wgpu::Adapter  adapter;
//wgpu::Device   device;
//wgpu::RenderPipeline pipeline;
//
//wgpu::Surface surface;
//wgpu::TextureFormat format;
//
//const uint32_t kWidth = 1000;
//const uint32_t kHeight = 1000;
//const uint32_t GRID = 100;               
//const uint32_t INSTANCES = GRID * GRID;
//
//// ======================================================================================
//// Surface config
//// ======================================================================================
//void ConfigureSurface() {
//    wgpu::SurfaceCapabilities capabilities;
//    surface.GetCapabilities(adapter, &capabilities);
//
//    // pick the first supported format
//    format = capabilities.formats[0];
//
//    // present mode: Fifo is always supported
//    wgpu::SurfaceConfiguration config{
//      .device = device,
//      .format = format,
//      .width = kWidth,
//      .height = kHeight,
//      .presentMode = wgpu::PresentMode::Fifo
//    };
//    surface.Configure(&config);
//}
//
//// ======================================================================================
///* Initialization: instance, adapter, device (using WaitAny to stay synchronous) */
//// ======================================================================================
//void Init() {
//    static const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
//    wgpu::InstanceDescriptor instanceDesc{ .requiredFeatureCount = 1,
//                                           .requiredFeatures = &kTimedWaitAny };
//    instance = wgpu::CreateInstance(&instanceDesc);
//
//    // Request adapter (synchronously via WaitAny)
//    wgpu::Future f1 = instance.RequestAdapter(
//        /*options*/nullptr, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message) {
//            if (status != wgpu::RequestAdapterStatus::Success) {
//                std::cerr << "RequestAdapter failed: " << message << "\n";
//                std::exit(1);
//            }
//            adapter = std::move(a);
//        });
//    instance.WaitAny(f1, UINT64_MAX);
//
//    // Create device
//    wgpu::DeviceDescriptor desc{};
//    desc.SetUncapturedErrorCallback([](const wgpu::Device&,
//        wgpu::ErrorType errorType,
//        wgpu::StringView message) {
//            std::cerr << "Device error (" << int(errorType) << "): " << message << "\n";
//        });
//
//    wgpu::Future f2 = adapter.RequestDevice(
//        &desc, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestDeviceStatus status, wgpu::Device d, wgpu::StringView message) {
//            if (status != wgpu::RequestDeviceStatus::Success) {
//                std::cerr << "RequestDevice failed: " << message << "\n";
//                std::exit(1);
//            }
//            device = std::move(d);
//        });
//    instance.WaitAny(f2, UINT64_MAX);
//}
//
//// ======================================================================================
//// WGSL: procedural vertices + per-instance offset (no buffers)
//// ======================================================================================
//static const char kShader[] = R"(
//  const GRID : u32 = 100u;
//
//  struct VSOut {
//    @builtin(position) pos : vec4f,
//    @location(0) color : vec3f
//  };
//
//  fn hash11(n: f32) -> f32 {
//    let x = fract(sin(n) * 43758.5453);
//    return x;
//  }
//
//  @vertex
//  fn vertexMain(
//    @builtin(vertex_index)  vi   : u32,
//    @builtin(instance_index) inst : u32
//  ) -> VSOut {
//    // Base triangle in local space
//    var base = array<vec2f, 3>(
//      vec2f(0.0,  1.0),
//      vec2f(-1.0, -1.0),
//      vec2f( 1.0, -1.0)
//    );
//
//    // Which cell are we?
//    let x = f32(inst % GRID);
//    let y = f32(inst / GRID);
//
//    // Map cell center to NDC [-1,1]
//    let cellSize = 2.0 / f32(GRID);
//    let offset = vec2f(
//      -1.0 + (x + 0.5) * cellSize,
//      -1.0 + (y + 0.5) * cellSize
//    );
//
//    // Scale triangle to fit within its cell
//    let scale = 0.9 * (cellSize * 0.5);
//    let p = offset + base[vi] * scale;
//
//    // Procedural color from instance id
//    let t = hash11(f32(inst));
//    let color = vec3f(0.2 + 0.8 * fract(vec3f(t, t*1.37, t*2.11)));
//
//    var out : VSOut;
//    out.pos = vec4f(p, 0.0, 1.0);
//    out.color = color;
//    return out;
//  }
//
//  @fragment
//  fn fragmentMain(@location(0) color : vec3f) -> @location(0) vec4f {
//    return vec4f(color, 1.0);
//  }
//)";
//
//// ======================================================================================
//void CreateRenderPipeline() {
//    wgpu::ShaderSourceWGSL wgsl{ {.code = kShader} };
//    wgpu::ShaderModuleDescriptor smDesc{ .nextInChain = &wgsl };
//    wgpu::ShaderModule shader = device.CreateShaderModule(&smDesc);
//
//    // Color target
//    wgpu::ColorTargetState colorTarget{
//      .format = format,
//    };
//
//    // Fragment state
//    wgpu::FragmentState fs{
//      .module = shader,
//      .targetCount = 1,
//      .targets = &colorTarget
//    };
//
//    // Pipeline
//    wgpu::RenderPipelineDescriptor rpDesc{};
//    rpDesc.vertex.module = shader;
//    rpDesc.fragment = &fs;
//
//    pipeline = device.CreateRenderPipeline(&rpDesc);
//}
//
//// ======================================================================================
//void Render() {
//    static auto lastTime = std::chrono::high_resolution_clock::now();
//    static int frameCount = 0;
//    static double totalMs = 0.0;
//    static constexpr int kReportEvery = 60;
//
//    auto t0 = std::chrono::high_resolution_clock::now();
//
//    wgpu::SurfaceTexture st{};
//    surface.GetCurrentTexture(&st);
//
//    wgpu::TextureView backbuffer = st.texture.CreateView();
//
//    wgpu::RenderPassColorAttachment colorAttachment{
//      .view = backbuffer,
//      .loadOp = wgpu::LoadOp::Clear,
//      .storeOp = wgpu::StoreOp::Store,
//      .clearValue = {0.05, 0.05, 0.06, 1.0}
//    };
//
//    wgpu::RenderPassDescriptor rpDesc{
//      .colorAttachmentCount = 1,
//      .colorAttachments = &colorAttachment
//    };
//
//    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
//    {
//        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&rpDesc);
//        pass.SetPipeline(pipeline);
//        pass.Draw(3, INSTANCES); // 3 verts, 100x100 instances. 여기서 실행되는게 아냐
//        pass.End();
//    }
//    wgpu::CommandBuffer cmd = encoder.Finish();
//    device.GetQueue().Submit(1, &cmd); //이렇게 보내야 실행됨.
//
//    // Present the acquired texture
//    surface.Present();
//
//    auto t1 = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> dt = t1 - t0;
//
//    totalMs += dt.count();
//    frameCount++;
//    if (frameCount % kReportEvery == 0) {
//        double avg = totalMs / kReportEvery;
//        double fps = 1000.0 / avg;
//        std::cout << "[perf] avg frame: " << avg << " ms   FPS: " << fps << "\n";
//        totalMs = 0.0;
//    }
//}
//
//// ======================================================================================
//void InitGraphics() {
//    ConfigureSurface();
//    CreateRenderPipeline();
//}
//
//// ======================================================================================
//void Start() {
//    if (!glfwInit()) return;
//
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "WebGPU instanced grid (no buffers)", nullptr, nullptr);
//
//    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
//    InitGraphics();
//
//#if defined(__EMSCRIPTEN__)
//    emscripten_set_main_loop([]() { Render(); }, 0, false);
//#else
//    while (!glfwWindowShouldClose(window)) {
//        glfwPollEvents();
//        Render();
//        instance.ProcessEvents(); // service callbacks/futures
//    }
//    glfwDestroyWindow(window);
//    glfwTerminate();
//#endif
//}
//
//// ======================================================================================
//int main() {
//    Init();
//    Start();
//    return 0;
//}
