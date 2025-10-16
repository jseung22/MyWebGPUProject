//#include <iostream>
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
//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"
//
//wgpu::Texture      gTexture;
//wgpu::TextureView  gTextureView;
//wgpu::Sampler      gSampler;
//wgpu::BindGroupLayout gBGL;
//wgpu::BindGroup       gBindGroup;
//wgpu::PipelineLayout  gPipelineLayout;
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
//struct LoadedImage {
//    int w = 0, h = 0, comp = 0; // �̹��� ������(w), ���� ����(h), ���� ä�μ�(comp)
//    std::vector<uint8_t> rgba; // ���ڵ��� �ȼ� �����͸� ���� ����(RGBA 8bit x 4 channel)
//};
//
//LoadedImage LoadImageRGBA(const char* path) {
//    LoadedImage out;
//    stbi_uc* data = stbi_load(path, &out.w, &out.h, &out.comp, 4);
//    if (!data) {
//        std::cerr << "Failed to load image: " << path << "\n";
//        std::exit(1);
//    }
//    out.comp = 4;
//    out.rgba.assign(data, data + (out.w * out.h * 4));
//    stbi_image_free(data);
//    return out;
//}
//
//void CreateTextureAndBindGroup(const char* path) {
//    // �̹��� �ε�
//    LoadedImage img = LoadImageRGBA(path);
//
//    // �ؽ�ó ���� (RGBAUnorm)
//    wgpu::TextureDescriptor td;
//    td.size = { (uint32_t)img.w, (uint32_t)img.h, 1 }; //e2D�̸� depthOrArrayLayers���� ���� ° ���� �迭 ���̾� ���� �ؼ� 
//    td.format = wgpu::TextureFormat::RGBA8Unorm;
//    td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
//    td.dimension = wgpu::TextureDimension::e2D; //texture�� 1D/2D/3D �� �������� ����. ���⼱ 2D �ؽ�ó
//    td.mipLevelCount = 1;
//    td.sampleCount = 1;
//    gTexture = device.CreateTexture(&td);
//    gTextureView = gTexture.CreateView();
//
//    // 3) ���ε� (WriteTexture)
//    wgpu::ImageCopyTexture dst;
//    dst.texture = gTexture;
//    dst.mipLevel = 0;
//    dst.origin = { 0,0,0 };
//    dst.aspect = wgpu::TextureAspect::All;
//
//    wgpu::TextureDataLayout layout;
//    layout.offset = 0;
//    layout.bytesPerRow = 4 * img.w;     // RGBA8
//    layout.rowsPerImage = img.h;
//
//    wgpu::Extent3D extent = { (uint32_t)img.w, (uint32_t)img.h, 1 };
//    device.GetQueue().WriteTexture(&dst, img.rgba.data(), img.rgba.size(), &layout, &extent);
//
//    // 4) ���÷�
//    wgpu::SamplerDescriptor sd;
//    sd.magFilter = wgpu::FilterMode::Linear;
//    sd.minFilter = wgpu::FilterMode::Linear;
//    sd.mipmapFilter = wgpu::MipmapFilterMode::Linear;
//    sd.addressModeU = wgpu::AddressMode::ClampToEdge;
//    sd.addressModeV = wgpu::AddressMode::ClampToEdge;
//    gSampler = device.CreateSampler(&sd);
//
//    // 5) ���ε� �׷� ���̾ƿ�
//    wgpu::BindGroupLayoutEntry bgle[2] = {};
//    bgle[0].binding = 0;
//    bgle[0].visibility = wgpu::ShaderStage::Fragment;
//    bgle[0].sampler.type = wgpu::SamplerBindingType::Filtering;
//
//    bgle[1].binding = 1;
//    bgle[1].visibility = wgpu::ShaderStage::Fragment;
//    bgle[1].texture.sampleType = wgpu::TextureSampleType::Float;
//    bgle[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;
//    bgle[1].texture.multisampled = false;
//
//    wgpu::BindGroupLayoutDescriptor bgld;
//    bgld.entryCount = 2;
//    bgld.entries = bgle;
//    gBGL = device.CreateBindGroupLayout(&bgld);
//
//    // 6) ���ε� �׷�
//    wgpu::BindGroupEntry bge[2] = {};
//    bge[0].binding = 0;
//    bge[0].sampler = gSampler;
//
//    bge[1].binding = 1;
//    bge[1].textureView = gTextureView;
//
//    wgpu::BindGroupDescriptor bgd;
//    bgd.layout = gBGL;
//    bgd.entryCount = 2;
//    bgd.entries = bge;
//    gBindGroup = device.CreateBindGroup(&bgd);
//}
//
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
//static const char kShader[] = R"(
//  struct VSOut {
//    @builtin(position) pos : vec4f,
//    @location(0) uv : vec2f
//  };
//
//  @group(0) @binding(0) var samp : sampler;
//  @group(0) @binding(1) var tex0 : texture_2d<f32>;
//
//  @vertex
//  fn vertexMain(@location(0) inPos : vec2f,
//                @location(1) inUV : vec2f) -> VSOut {
//    var out : VSOut;
//    out.pos = vec4f(inPos, 0.0, 1.0);
//    out.uv = inUV;
//    return out;
//  }
//
//  @fragment
//  fn fragmentMain(@location(0) uv : vec2f) -> @location(0) vec4f {
//    let c = textureSample(tex0, samp, uv);
//    return c;
//  }
//)";
//
//wgpu::Buffer CreateBufferFromData(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd;
//    bd.size = size;
//    bd.usage = usage;
//    bd.mappedAtCreation = true; // ���� ������ ���ÿ� CPU���� ���� �����ϰ� ��
//    wgpu::Buffer buf = device.CreateBuffer(&bd);
//
//    std::memcpy(buf.GetMappedRange(), data, size); //������ �ʵ� �޸� ���� ����Ű�� CPU ������ ���, ���⿡ data ���� ����. 
//    buf.Unmap(); //���� ����. �ش������ʹ� ��ȿ, �����ʹ� GPU �� ���ۿ� �ݿ���
//    return buf; //�ϼ��� ���� �ڵ� ��ȯ. ���� ���� �� SetVertexBuffer / SetIndexBuffer � �Ѱܼ� ���
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
//void Render() {
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
//}
//
//void InitGraphics() {
//    ConfigureSurface();
//    CreateTriangleBuffers();
//    CreateRenderPipeline();
//}
//
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
//        surface.Present();
//        instance.ProcessEvents(); // service callbacks/futures
//    }
//    glfwDestroyWindow(window);
//    glfwTerminate();
//#endif
//}
//
//int main() {
//	Init();
//	Start();
//	return 0;
//}