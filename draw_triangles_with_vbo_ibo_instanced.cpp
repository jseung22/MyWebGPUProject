//#include <iostream>
//#include <chrono>
//#include <vector>
//#include <cmath>
//#include <cstring>
//
//#include <GLFW/glfw3.h>
//#if defined(__EMSCRIPTEN__)
//#include <emscripten/emscripten.h>
//#endif
//#include <dawn/webgpu_cpp_print.h>
//#include <webgpu/webgpu_cpp.h>
//#include <webgpu/webgpu_glfw.h>
//
//// ======================================================================================
//// WebGPU �ٽ� ��ü�� - GPU�� ����Ǵ� �ֿ� �������̽���
//// ======================================================================================
//wgpu::Instance instance;        // WebGPU �ν��Ͻ� - ��� GPU �۾��� ������
//wgpu::Adapter  adapter;         // GPU ����� - ������ GPU �ϵ��� ��Ÿ��
//wgpu::Device   device;          // GPU ����̽� - ���� GPU ���ҽ��� ��� ����
//wgpu::RenderPipeline pipeline;  // ������ ���������� - GPU ������ ���¸� ����
//
//// Surface�� ȭ�鿡 �׸� �� �ִ� ĵ������ ��Ÿ�� (������ �ý��۰� ����)
//wgpu::Surface surface;
//wgpu::TextureFormat format;
//
//// ======================================================================================
//// ���� ��ü�� - GPU �޸𸮿� ����Ǵ� ������
//// ======================================================================================
//wgpu::Buffer vertexBuffer;      // ���� ������ (��ġ + ����) - �ﰢ�� ��� ����
//wgpu::Buffer indexBuffer;       // �ε��� ������ - ���� ���� ���� ����
//wgpu::Buffer instanceBuffer;    // �ν��Ͻ� ������ - �� �ﰢ���� ��ġ ������
//
//// ======================================================================================
//// Uniform �ý��� - ���̴��� ���� ������ ����
//// ======================================================================================
//wgpu::Buffer uniformBuffer;           // uniform ������ ���� ����
//wgpu::BindGroup bindGroup;           // uniform buffer�� ���̴��� �����ϴ� �׷�
//wgpu::BindGroupLayout bindGroupLayout; // bind group�� ���̾ƿ� ����
//
//// ======================================================================================
//// ������ ���� �����
//// ======================================================================================
//const uint32_t kWidth = 1000;
//const uint32_t kHeight = 1000;
//
//static constexpr uint32_t GRID = 100;                    // 100x100 ����
//static constexpr uint32_t INSTANCE_COUNT = GRID * GRID;  // �� 10,000�� �ﰢ�� �ν��Ͻ�
//
//// ======================================================================================
//// Uniform ������ ����ü - CPU���� GPU�� ������ ������
//// ======================================================================================
//struct UniformData {
//    float grid;         // ���� ũ�� ��
//    float padding[3];   // GPU �޸� ������ ���� �е� (16����Ʈ ���� ����)
//};
//
//// ======================================================================================
//// Surface ���� - ȭ�鿡 �׸��� ���� ĵ���� ����
//// ======================================================================================
//void ConfigureSurface() {
//    // GPU�� �����ϴ� surface ��� ��ȸ
//    wgpu::SurfaceCapabilities capabilities;
//    surface.GetCapabilities(adapter, &capabilities);
//
//    // ù ��°�� �����Ǵ� ���� ��� (���� BGRA8Unorm �Ǵ� RGBA8Unorm)
//    format = capabilities.formats[0];
//
//    // Surface ���� ����ü ����
//    wgpu::SurfaceConfiguration config;
//    config.device = device;           // ����� GPU ����̽�
//    config.format = format;           // �ȼ� ����
//    config.width = kWidth;            // ȭ�� �ʺ�
//    config.height = kHeight;          // ȭ�� ����
//    config.presentMode = wgpu::PresentMode::Fifo;  // V-Sync ��� (60fps ����)
//    surface.Configure(&config);
//}
//
//// ======================================================================================
//// WebGPU �ʱ�ȭ - Instance, Adapter, Device ������ ����
//// ======================================================================================
//void Init() {
//    // TimedWaitAny ��� ��û - �񵿱� �۾� ��⸦ ���� ���
//    const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
//
//    // WebGPU Instance ���� - ��� WebGPU �۾��� ������
//    wgpu::InstanceDescriptor instanceDesc;
//    instanceDesc.requiredFeatureCount = 1;
//    instanceDesc.requiredFeatures = &kTimedWaitAny;
//    instance = wgpu::CreateInstance(&instanceDesc);
//
//    // ======================================================================================
//    // GPU Adapter ��û - �ý����� GPU �ϵ���� ã��
//    // ======================================================================================
//    wgpu::Future f1 = instance.RequestAdapter(
//        nullptr,                           // �⺻ �ɼ� ���
//        wgpu::CallbackMode::WaitAnyOnly,   // ����� ���
//        [](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message) {
//            if (status != wgpu::RequestAdapterStatus::Success) {
//                std::cerr << "RequestAdapter failed: " << message << "\n";
//                std::exit(1);
//            }
//            adapter = std::move(a);  // ���� ������ adapter ����
//        });
//    instance.WaitAny(f1, UINT64_MAX);  // �Ϸ�� ������ ���
//
//    // ======================================================================================
//    // GPU Device ��û - ���� GPU ���ҽ��� ��� ������ ���� ����̽�
//    // ======================================================================================
//    wgpu::DeviceDescriptor desc;
//    // GPU ���� �߻� �� �ݹ� �Լ� ����
//    desc.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType errorType, wgpu::StringView message) {
//        std::cerr << "Device error (" << int(errorType) << "): " << message << "\n";
//    });
//
//    wgpu::Future f2 = adapter.RequestDevice(
//        &desc, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestDeviceStatus status, wgpu::Device d, wgpu::StringView message) {
//            if (status != wgpu::RequestDeviceStatus::Success) {
//                std::cerr << "RequestDevice failed: " << message << "\n";
//                std::exit(1);
//            }
//            device = std::move(d);  // ���� ������ device ����
//        });
//    instance.WaitAny(f2, UINT64_MAX);  // �Ϸ�� ������ ���
//}
//
//// ======================================================================================
//// WGSL ���̴� - GPU���� ����Ǵ� ���α׷�
//// ======================================================================================
//static const char kShader[] = R"(
//  // ======================================================================================
//  // Uniform ����ü ���� - CPU���� ���޹��� ������
//  // ======================================================================================
//  struct Uniforms {
//    grid : f32,    // ���� ũ�� (100.0)
//  };
//
//  // @group(0) @binding(0): ù ��° bind group�� ù ��° ���ε�
//  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
//
//  // ======================================================================================
//  // ���� ���̴� ��� ����ü
//  // ======================================================================================
//  struct VSOut {
//    @builtin(position) pos : vec4f,   // Ŭ�� ���� ��ǥ (�ʼ�)
//    @location(0) color : vec3f        // ���� ������ (fragment shader�� ����)
//  };
//
//  // ======================================================================================
//  // ���� ���̴� - �� �������� �����
//  // ======================================================================================
//  @vertex
//  fn vertexMain(
//      @location(0) inPos : vec2f,        // ���� ��ġ (vertex buffer slot 0)
//      @location(1) inCol : vec3f,        // ���� ���� (vertex buffer slot 0)
//      @location(2) instOffset : vec2f    // �ν��Ͻ� ������ (vertex buffer slot 1)
//  ) -> VSOut {
//    // NDC(-1~1) �������� �� ũ�� ���
//    let step : f32 = 2.0 / uniforms.grid;    // �� ���� ũ�� (0.02)
//    let scale : f32 = 0.45 * step;           // �ﰢ�� ũ�� (���� 45% ũ��)
//
//    var out : VSOut;
//    // ���� ��ġ = �ν��Ͻ� ������ + (�⺻ �ﰢ�� * ������)
//    out.pos = vec4f(instOffset + inPos * scale, 0.0, 1.0);
//    out.color = inCol;    // ������ �״�� ����
//    return out;
//  }
//
//  // ======================================================================================
//  // �����׸�Ʈ ���̴� - �� �ȼ����� �����
//  // ======================================================================================
//  @fragment
//  fn fragmentMain(@location(0) color : vec3f) -> @location(0) vec4f {
//    return vec4f(color, 1.0);    // RGB + Alpha(������)
//  }
//)";
//
//// ======================================================================================
//// ���� ���� ���� �Լ� - �����͸� GPU �޸𸮿� ����
//// ======================================================================================
//wgpu::Buffer CreateBufferFromData(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd;
//    bd.size = size;                    // ���� ũ��
//    bd.usage = usage;                  // ���� ��� �뵵
//    bd.mappedAtCreation = true;        // ���� ������ CPU���� ���� �����ϰ� ����
//    wgpu::Buffer buf = device.CreateBuffer(&bd);
//
//    // CPU �޸��� �����͸� GPU ���۷� ����
//    std::memcpy(buf.GetMappedRange(), data, size);
//    buf.Unmap();    // CPU ���� ���� (���� GPU�� ���� ����)
//    return buf;
//}
//
//// ======================================================================================
//// �ﰢ�� ���� �� �ε��� ���� ����
//// ======================================================================================
//void CreateTriangleBuffers() {
//    // ======================================================================================
//    // ���� ������ ���� - ���͸���� ���� (��ġ + ����)
//    // ======================================================================================
//    const float vertices[] = {
//        //   x,     y,      r,    g,    b
//         0.0f,  1.0f,    1.0f, 0.3f, 0.3f,  // v0: ��� (����)
//        -1.0f, -1.0f,    0.3f, 1.0f, 0.3f,  // v1: ���ϴ� (�ʷ�)
//         1.0f, -1.0f,    0.3f, 0.3f, 1.0f   // v2: ���ϴ� (�Ķ�)
//    };
//    // �ε��� ������ - ���� ���� ���� (�ð� �ݴ� ����)
//    const uint32_t indices[] = { 0, 1, 2 };
//
//    // GPU ���� ����
//    vertexBuffer = CreateBufferFromData(vertices, sizeof(vertices),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//    indexBuffer = CreateBufferFromData(indices, sizeof(indices),
//        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst);
//}
//
//// ======================================================================================
//// �ν��Ͻ� ������ ���� ���� - 10,000�� �ﰢ���� ��ġ ����
//// ======================================================================================
//void CreateInstanceBuffer() {
//    // �� �ν��Ͻ��� ��ġ ������ ����� ����
//    std::vector<float> offsets;
//    offsets.reserve(INSTANCE_COUNT * 2);  // x, y ��ǥ�̹Ƿ� *2
//
//    const float grid = static_cast<float>(GRID);     // 100.0
//    const float step = 2.0f / grid;                  // NDC �������� �� ũ�� (0.02)
//    const float start = -1.0f + step * 0.5f;        // ù ��° ���� �߽� (-0.99)
//
//    // ======================================================================================
//    // 100x100 ������ �� �� �߽� ��ǥ ���
//    // ======================================================================================
//    for (uint32_t y = 0; y < GRID; ++y) {
//        for (uint32_t x = 0; x < GRID; ++x) {
//            // �� ���� �߽� ��ǥ ���
//            float cx = start + step * static_cast<float>(x);  // X ��ǥ
//            float cy = start + step * static_cast<float>(y);  // Y ��ǥ
//            offsets.push_back(cx);
//            offsets.push_back(cy);
//        }
//    }
//
//    // �ν��Ͻ� ���� ���� (per-instance �����Ϳ�)
//    instanceBuffer = CreateBufferFromData(
//        offsets.data(), offsets.size() * sizeof(float),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//}
//
//// ======================================================================================
//// Uniform ���� ���� - ���̴��� ������ uniform ������
//// ======================================================================================
//void CreateUniformBuffer() {
//    UniformData uniformData;
//    uniformData.grid = static_cast<float>(GRID);  // ���� ũ�� ���� (100.0)
//
//    // Uniform ���� ����
//    uniformBuffer = CreateBufferFromData(&uniformData, sizeof(UniformData),
//        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//}
//
//// ======================================================================================
//// Bind Group Layout ���� - uniform buffer�� ���̾ƿ� ����
//// ======================================================================================
//void CreateBindGroupLayout() {
//    wgpu::BindGroupLayoutEntry entry;
//    entry.binding = 0;                                     // @binding(0)�� �ش�
//    entry.visibility = wgpu::ShaderStage::Vertex;          // ���� ���̴������� ����
//    entry.buffer.type = wgpu::BufferBindingType::Uniform;  // uniform buffer Ÿ��
//    entry.buffer.minBindingSize = sizeof(UniformData);     // �ּ� ���� ũ��
//
//    wgpu::BindGroupLayoutDescriptor layoutDesc;
//    layoutDesc.entryCount = 1;
//    layoutDesc.entries = &entry;
//    bindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
//}
//
//// ======================================================================================
//// Bind Group ���� - ���� uniform buffer�� ���ε�
//// ======================================================================================
//void CreateBindGroup() {
//    wgpu::BindGroupEntry entry;
//    entry.binding = 0;                    // @binding(0)�� �ش�
//    entry.buffer = uniformBuffer;         // ���� uniform buffer
//    entry.offset = 0;                     // ���� ���� ������
//    entry.size = sizeof(UniformData);     // ���ε��� ������ ũ��
//
//    wgpu::BindGroupDescriptor bindGroupDesc;
//    bindGroupDesc.layout = bindGroupLayout;  // ������ ���� layout ���
//    bindGroupDesc.entryCount = 1;
//    bindGroupDesc.entries = &entry;
//    bindGroup = device.CreateBindGroup(&bindGroupDesc);
//}
//
//// ======================================================================================
//// ������ ���������� ���� - GPU�� ������ ���� ����
//// ======================================================================================
//void CreateRenderPipeline() {
//    // ======================================================================================
//    // ���̴� ��� ����
//    // ======================================================================================
//    wgpu::ShaderSourceWGSL wgsl;
//    wgsl.code = kShader;  // WGSL �ڵ�
//    wgpu::ShaderModuleDescriptor smDesc;
//    smDesc.nextInChain = &wgsl;
//    wgpu::ShaderModule shader = device.CreateShaderModule(&smDesc);
//
//    // ======================================================================================
//    // ���� ���� ���̾ƿ� ���� (Slot 0: per-vertex ������)
//    // ======================================================================================
//    wgpu::VertexAttribute vattrs0[2];
//    // ��ġ �Ӽ� (@location(0))
//    vattrs0[0].format = wgpu::VertexFormat::Float32x2;  // vec2f
//    vattrs0[0].offset = 0;                              // ���� ���ۺ���
//    vattrs0[0].shaderLocation = 0;                      // @location(0)
//
//    // ���� �Ӽ� (@location(1))
//    vattrs0[1].format = wgpu::VertexFormat::Float32x3;  // vec3f
//    vattrs0[1].offset = sizeof(float) * 2;              // ��ġ ������ �ں���
//    vattrs0[1].shaderLocation = 1;                      // @location(1)
//
//    wgpu::VertexBufferLayout vbl0;
//    vbl0.arrayStride = sizeof(float) * (2 + 3);         // ������ ������ ũ�� (��ġ2 + ����3)
//    vbl0.attributeCount = 2;                            // �Ӽ� ����
//    vbl0.attributes = vattrs0;
//    vbl0.stepMode = wgpu::VertexStepMode::Vertex;       // �������� ������ ����
//
//    // ======================================================================================
//    // �ν��Ͻ� ���� ���̾ƿ� ���� (Slot 1: per-instance ������)
//    // ======================================================================================
//    wgpu::VertexAttribute vattrs1[1];
//    // �ν��Ͻ� ������ �Ӽ� (@location(2))
//    vattrs1[0].format = wgpu::VertexFormat::Float32x2;  // vec2f
//    vattrs1[0].offset = 0;                              // ���� ���ۺ���
//    vattrs1[0].shaderLocation = 2;                      // @location(2)
//
//    wgpu::VertexBufferLayout vbl1;
//    vbl1.arrayStride = sizeof(float) * 2;               // �ν��Ͻ��� ������ ũ�� (x, y)
//    vbl1.attributeCount = 1;                            // �Ӽ� ����
//    vbl1.attributes = vattrs1;
//    vbl1.stepMode = wgpu::VertexStepMode::Instance;     // �ν��Ͻ����� ������ ����
//
//    // ======================================================================================
//    // ���� ��� ����
//    // ======================================================================================
//    wgpu::ColorTargetState colorTarget;
//    colorTarget.format = format;  // Surface�� ���� ���� ���
//
//    // ======================================================================================
//    // �����׸�Ʈ ���̴� ����
//    // ======================================================================================
//    wgpu::FragmentState fs;
//    fs.module = shader;              // ���̴� ���
//    fs.entryPoint = "fragmentMain";  // �����׸�Ʈ ���̴� ������
//    fs.targetCount = 1;              // ���� ��� ����
//    fs.targets = &colorTarget;
//
//    // ======================================================================================
//    // ���������� ���̾ƿ� ���� (uniform buffer ����)
//    // ======================================================================================
//    wgpu::PipelineLayoutDescriptor layoutDesc;
//    layoutDesc.bindGroupLayoutCount = 1;
//    layoutDesc.bindGroupLayouts = &bindGroupLayout;
//    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&layoutDesc);
//
//    // ======================================================================================
//    // ������ ���������� ����
//    // ======================================================================================
//    wgpu::RenderPipelineDescriptor rpDesc;
//    rpDesc.layout = pipelineLayout;                      // ���������� ���̾ƿ�
//    rpDesc.vertex.module = shader;                       // ���� ���̴� ���
//    rpDesc.vertex.entryPoint = "vertexMain";             // ���� ���̴� ������
//
//    // �� ���� ���� ���� ���� (per-vertex, per-instance)
//    wgpu::VertexBufferLayout vbuffers[2] = { vbl0, vbl1 };
//    rpDesc.vertex.bufferCount = 2;
//    rpDesc.vertex.buffers = vbuffers;
//
//    rpDesc.fragment = &fs;  // �����׸�Ʈ ���̴� ����
//
//    pipeline = device.CreateRenderPipeline(&rpDesc);
//}
//
//// ======================================================================================
//// ������ ���� - �� �����Ӹ��� ����
//// ======================================================================================
//void Render() {
//    // ======================================================================================
//    // ���� ������ ���� ������
//    // ======================================================================================
//    static auto lastTime = std::chrono::high_resolution_clock::now();
//    static int frameCount = 0;
//    static double totalMs = 0.0;
//    static constexpr int kReportEvery = 60;  // 60�����Ӹ��� ����Ʈ
//
//    auto t0 = std::chrono::high_resolution_clock::now();
//
//    // ======================================================================================
//    // ���� Ÿ�� �غ� - Surface���� ���� �������� �ؽ�ó ��������
//    // ======================================================================================
//    wgpu::SurfaceTexture st;
//    surface.GetCurrentTexture(&st);
//    wgpu::TextureView backbuffer = st.texture.CreateView();
//
//    // ======================================================================================
//    // ���� �н� ���� - ȭ�� Ŭ���� �� ������ �غ�
//    // ======================================================================================
//    wgpu::RenderPassColorAttachment colorAttachment;
//    colorAttachment.view = backbuffer;                            // ���� Ÿ��
//    colorAttachment.loadOp = wgpu::LoadOp::Clear;                // ȭ�� Ŭ����
//    colorAttachment.storeOp = wgpu::StoreOp::Store;              // ��� ����
//    colorAttachment.clearValue = { 0.05, 0.05, 0.06, 1.0 };     // Ŭ���� ���� (��ο� ȸ��)
//
//    wgpu::RenderPassDescriptor rpDesc;
//    rpDesc.colorAttachmentCount = 1;
//    rpDesc.colorAttachments = &colorAttachment;
//
//    // ======================================================================================
//    // ��� ���ڴ� ���� �� ������ ��� ���
//    // ======================================================================================
//    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
//    {
//        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&rpDesc);
//        
//        // ������ ���� ����
//        pass.SetPipeline(pipeline);                                    // ������ ����������
//        pass.SetBindGroup(0, bindGroup);                              // uniform buffer ���ε�
//        pass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize);    // ���� ���� (slot 0)
//        pass.SetVertexBuffer(1, instanceBuffer, 0, wgpu::kWholeSize);  // �ν��Ͻ� ���� (slot 1)
//        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize);
//        
//        // ���� ��ο� �� - 10,000�� �ﰢ�� �ν��Ͻ� ������
//        pass.DrawIndexed(3,              // �ε��� ���� (�ﰢ�� = 3�� �ε���)
//                        INSTANCE_COUNT,  // �ν��Ͻ� ���� (10,000��)
//                        0,               // �ε��� ���� ������
//                        0,               // ���� ���� ������
//                        0);              // �ν��Ͻ� ���� ������
//        pass.End();
//    }
//    
//    // ======================================================================================
//    // ��� ���� �� ȭ�� ǥ��
//    // ======================================================================================
//    wgpu::CommandBuffer cmd = encoder.Finish();  // ��� ���� �ϼ�
//    device.GetQueue().Submit(1, &cmd);           // GPU�� ��� ����
//    surface.Present();                           // ȭ�鿡 ��� ǥ��
//
//    // ======================================================================================
//    // ���� ���� �� ����Ʈ
//    // ======================================================================================
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
//// �׷��� �ʱ�ȭ - ��� GPU ���ҽ� �غ�
//// ======================================================================================
//void InitGraphics() {
//    ConfigureSurface();        // Surface ����
//    CreateTriangleBuffers();   // ����/�ε��� ���� ����
//    CreateInstanceBuffer();    // �ν��Ͻ� ���� ����
//    CreateBindGroupLayout();   // Bind group layout ����
//    CreateUniformBuffer();     // Uniform buffer ����
//    CreateBindGroup();         // Bind group ����
//    CreateRenderPipeline();    // ������ ���������� ����
//}
//
//// ======================================================================================
//// ������ ���� �� ���� ���� ����
//// ======================================================================================
//void Start() {
//    if (!glfwInit()) return;
//
//    // OpenGL API ��� ���� (WebGPU ���)
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "WebGPU instanced triangles (VBO+IBO)", nullptr, nullptr);
//
//    // GLFW ������� WebGPU Surface ����
//    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
//    InitGraphics();
//
//#if defined(__EMSCRIPTEN__)
//    // �� ȯ��: �������� �ִϸ��̼� ���� ���
//    emscripten_set_main_loop([]() { Render(); }, 0, false);
//#else
//    // ����Ƽ�� ȯ��: ���� ���� ����
//    while (!glfwWindowShouldClose(window)) {
//        glfwPollEvents();           // ������ �̺�Ʈ ó��
//        Render();                   // ������ ����
//        instance.ProcessEvents();   // WebGPU �̺�Ʈ ó��
//    }
//    glfwDestroyWindow(window);
//    glfwTerminate();
//#endif
//}
//
//// ======================================================================================
//// ���α׷� ������
//// ======================================================================================
//int main() {
//    Init();    // WebGPU �ʱ�ȭ (Instance, Adapter, Device)
//    Start();   // ������ ���� �� ������ ���� ����
//    return 0;
//}
