//// draw_triangle_with_vbo_ibo.cpp
//#include <iostream>
//#include <chrono>
//#include <vector>
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
//wgpu::Buffer vertexBuffer;
//wgpu::Buffer indexBuffer;
//
//const uint32_t kWidth = 1000;
//const uint32_t kHeight = 1000;
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
//    wgpu::SurfaceConfiguration config;
//    config.device = device;
//    config.format = format;
//    config.width = kWidth;
//    config.height = kHeight;
//    config.presentMode = wgpu::PresentMode::Fifo;
//    surface.Configure(&config);
//}
//
//// ======================================================================================
//// Initialization: instance, adapter, device (using WaitAny to stay synchronous)
//// ======================================================================================
//void Init() {
//    const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
//
//    wgpu::InstanceDescriptor instanceDesc;
//    instanceDesc.requiredFeatureCount = 1;
//    instanceDesc.requiredFeatures = &kTimedWaitAny;
//    instance = wgpu::CreateInstance(&instanceDesc);
//
//    // Request adapter (synchronously via WaitAny)
//    wgpu::Future f1 = instance.RequestAdapter(
//        nullptr, wgpu::CallbackMode::WaitAnyOnly,
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
//    wgpu::DeviceDescriptor desc;
//    desc.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType errorType, wgpu::StringView message) {
//        std::cerr << "Device error (" << int(errorType) << "): " << message << "\n";
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
//// WGSL: vertex inputs from buffers (pos @loc0, color @loc1)
//// ======================================================================================
//static const char kShader[] = R"(
//  struct VSOut {
//    @builtin(position) pos : vec4f,
//    @location(0) color : vec3f
//  };
//
//  @vertex
//  fn vertexMain(@location(0) inPos : vec2f,
//                @location(1) inCol : vec3f) -> VSOut {
//    var out : VSOut;
//    out.pos = vec4f(inPos, 0.0, 1.0);
//    out.color = inCol;
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
//// Buffer helpers
//// ======================================================================================
//wgpu::Buffer CreateBufferFromData(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd;
//    bd.size = size;
//    bd.usage = usage;
//    bd.mappedAtCreation = true;
//    wgpu::Buffer buf = device.CreateBuffer(&bd);
//
//    std::memcpy(buf.GetMappedRange(), data, size);
//    buf.Unmap();
//    return buf;
//}
//
//void CreateTriangleBuffers() {
//    // Interleaved: vec2 pos + vec3 color
//    // Triangle in NDC
//    const float vertices[] = {
//        //   x,     y,      r,    g,    b
//         0.0f,  0.8f,    1.0f, 0.3f, 0.3f,  // v0
//        -0.8f, -0.8f,    0.3f, 1.0f, 0.3f,  // v1
//         0.8f, -0.8f,    0.3f, 0.3f, 1.0f   // v2
//    };
//    const uint32_t indices[] = { 0, 1, 2 };
//
//    vertexBuffer = CreateBufferFromData(vertices, sizeof(vertices),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//    indexBuffer = CreateBufferFromData(indices, sizeof(indices),
//        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst);
//}
//
//// ======================================================================================
//// Pipeline (with vertex buffer layout)
//// ======================================================================================
//void CreateRenderPipeline() {
//    // Shader module
//    wgpu::ShaderSourceWGSL wgsl;
//    wgsl.code = kShader;
//
//    wgpu::ShaderModuleDescriptor smDesc;
//    smDesc.nextInChain = &wgsl;
//    wgpu::ShaderModule shader = device.CreateShaderModule(&smDesc);
//
//    // Vertex attributes: pos (float2) @loc0, color (float3) @loc1
//    wgpu::VertexAttribute vattrs[2];
//    vattrs[0].format = wgpu::VertexFormat::Float32x2;
//    vattrs[0].offset = 0;
//    vattrs[0].shaderLocation = 0;
//
//    vattrs[1].format = wgpu::VertexFormat::Float32x3;
//    vattrs[1].offset = sizeof(float) * 2;
//    vattrs[1].shaderLocation = 1;
//
//    wgpu::VertexBufferLayout vbl;
//    vbl.arrayStride = sizeof(float) * (2 + 3);
//    vbl.attributeCount = 2;
//    vbl.attributes = vattrs;
//    vbl.stepMode = wgpu::VertexStepMode::Vertex;
//
//    // Color target
//    wgpu::ColorTargetState colorTarget;
//    colorTarget.format = format;
//
//    // Fragment state
//    wgpu::FragmentState fs;
//    fs.module = shader;
//    fs.entryPoint = "fragmentMain";
//    fs.targetCount = 1;
//    fs.targets = &colorTarget;
//
//    // Pipeline
//    wgpu::RenderPipelineDescriptor rpDesc;
//    rpDesc.vertex.module = shader;
//    rpDesc.vertex.entryPoint = "vertexMain";
//    rpDesc.vertex.bufferCount = 1;
//    rpDesc.vertex.buffers = &vbl;
//    rpDesc.fragment = &fs;
//
//    pipeline = device.CreateRenderPipeline(&rpDesc);
//}
//
//// ======================================================================================
//// Render loop
//// ======================================================================================
//void Render() {
//    static auto lastTime = std::chrono::high_resolution_clock::now();
//    static int frameCount = 0;
//    static double totalMs = 0.0;
//    static constexpr int kReportEvery = 60;
//
//    auto t0 = std::chrono::high_resolution_clock::now();
//
//    wgpu::SurfaceTexture st;
//    surface.GetCurrentTexture(&st);
//    wgpu::TextureView backbuffer = st.texture.CreateView();
//
//    wgpu::RenderPassColorAttachment colorAttachment;
//    colorAttachment.view = backbuffer;
//    colorAttachment.loadOp = wgpu::LoadOp::Clear;
//    colorAttachment.storeOp = wgpu::StoreOp::Store;
//    colorAttachment.clearValue = { 0.05, 0.05, 0.06, 1.0 };
//
//    wgpu::RenderPassDescriptor rpDesc;
//    rpDesc.colorAttachmentCount = 1;
//    rpDesc.colorAttachments = &colorAttachment;
//
//    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
//    {
//        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&rpDesc);
//        pass.SetPipeline(pipeline);
//        pass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize);
//        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize);
//        pass.DrawIndexed(3, 1, 0, 0, 0); // 3 indices, 1 instance
//        pass.End();
//    }
//    wgpu::CommandBuffer cmd = encoder.Finish();
//    device.GetQueue().Submit(1, &cmd);
//
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
//    CreateTriangleBuffers();
//    CreateRenderPipeline();
//}
//
//// ======================================================================================
//void Start() {
//    if (!glfwInit()) return;
//
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "WebGPU triangle (VBO+IBO)", nullptr, nullptr);
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
