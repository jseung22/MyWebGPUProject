//// ===================== WebGPU + GLFW로 3D 큐브 인스턴스 텍스처 렌더링 =====================
//// 50x50x50(총 125,000개) 큐브 인스턴싱 + 인스턴스별 텍스처 레이어 선택
//// 카메라 이동/회전, MVP, UBO, Depth Test, 마우스/키보드 입력, 인스턴스 버퍼(vec3 offset + u32 layer)
//
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <chrono>
//#include <cstring>
//#include <cstdint>
//#include <random>
//#include <GLFW/glfw3.h>
//#if defined(__EMSCRIPTEN__)
//#include <emscripten/emscripten.h>
//#endif
//#include <dawn/webgpu_cpp_print.h>
//#include <webgpu/webgpu_cpp.h>
//#include <webgpu/webgpu_glfw.h>
//
//struct Camera {
//    float pos[3] = { 0.0f, 0.0f, 150.0f };
//    float front[3] = { 0.0f, 0.0f, -1.0f };
//    float up[3] = { 0.0f, 1.0f, 0.0f };
//    float yaw = -90.0f, pitch = 0.0f;
//    float fov = 45.0f;
//    float lastX = 500.0f, lastY = 500.0f;
//    bool firstMouse = true;
//    void ProcessKeyboard(int key, float dt) {
//        float speed = 30.0f * dt;
//        float right[3]; Cross(front, up, right); Normalize(right);
//        if (key == GLFW_KEY_W) Move(front, speed);
//        if (key == GLFW_KEY_S) Move(front, -speed);
//        if (key == GLFW_KEY_A) Move(right, -speed);
//        if (key == GLFW_KEY_D) Move(right, speed);
//    }
//    void ProcessMouse(float xpos, float ypos) {
//        if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
//        float xoffset = (xpos - lastX) * 0.1f;
//        float yoffset = (lastY - ypos) * 0.1f;
//        lastX = xpos; lastY = ypos;
//        yaw += xoffset; pitch += yoffset;
//        if (pitch > 89.0f) pitch = 89.0f;
//        if (pitch < -89.0f) pitch = -89.0f;
//        UpdateFront();
//    }
//    void ProcessScroll(float yoffset) {
//        fov -= yoffset; if (fov < 1.0f) fov = 1.0f; if (fov > 45.0f) fov = 45.0f;
//    }
//    void UpdateFront() {
//        float ry = yaw * 3.14159265f / 180.0f, rp = pitch * 3.14159265f / 180.0f;
//        front[0] = cosf(ry) * cosf(rp); front[1] = sinf(rp); front[2] = sinf(ry) * cosf(rp);
//        Normalize(front);
//    }
//    void Move(const float* d, float a) { for (int i = 0; i < 3; ++i) pos[i] += d[i] * a; }
//    static void Cross(const float* a, const float* b, float* o) { o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0]; }
//    static void Normalize(float* v) { float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (l > 1e-5f) { v[0] /= l; v[1] /= l; v[2] /= l; } }
//};
//
//void Mat4Identity(float* m) { memset(m, 0, sizeof(float) * 16); m[0] = m[5] = m[10] = m[15] = 1.0f; }
//void Mat4Perspective(float* m, float fov, float aspect, float n, float f) {
//    float t = tanf(fov * 0.5f * 3.14159265f / 180.0f);
//    memset(m, 0, sizeof(float) * 16);
//    m[0] = 1.f / (aspect * t); m[5] = 1.f / t; m[10] = -(f + n) / (f - n); m[11] = -1.f; m[14] = -(2.f * f * n) / (f - n);
//}
//void Mat4LookAt(float* m, const float* e, const float* c, const float* up) {
//    float f[3] = { c[0] - e[0],c[1] - e[1],c[2] - e[2] }; Camera::Normalize(f);
//    float s[3]; Camera::Cross(f, up, s); Camera::Normalize(s);
//    float u[3]; Camera::Cross(s, f, u);
//    Mat4Identity(m);
//    m[0] = s[0]; m[1] = u[0]; m[2] = -f[0];
//    m[4] = s[1]; m[5] = u[1]; m[6] = -f[1];
//    m[8] = s[2]; m[9] = u[2]; m[10] = -f[2];
//    m[12] = -(s[0] * e[0] + s[1] * e[1] + s[2] * e[2]);
//    m[13] = -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]);
//    m[14] = f[0] * e[0] + f[1] * e[1] + f[2] * e[2];
//}
//void Mat4Mul(float* out, const float* a, const float* b) {
//    float r[16]; for (int i = 0; i < 4; i++)for (int j = 0; j < 4; j++) { r[i * 4 + j] = 0; for (int k = 0; k < 4; k++) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j]; }
//    memcpy(out, r, sizeof(float) * 16);
//}
//
//struct MVP { float model[16]; float view[16]; float proj[16]; };
//
//// ★ 정점: 24개(면별 고유 UV). 포맷: pos(3) + uv(2) = 5 floats/vertex
////   인덱스: 36개 동일
//static const float cubeVertices[] = {
//    //  +Z (front)
//    -0.5f,-0.5f, 0.5f, 0.f,0.f,
//     0.5f,-0.5f, 0.5f, 1.f,0.f,
//     0.5f, 0.5f, 0.5f, 1.f,1.f,
//    -0.5f, 0.5f, 0.5f, 0.f,1.f,
//    //  -Z (back)
//     0.5f,-0.5f,-0.5f, 0.f,0.f,
//    -0.5f,-0.5f,-0.5f, 1.f,0.f,
//    -0.5f, 0.5f,-0.5f, 1.f,1.f,
//     0.5f, 0.5f,-0.5f, 0.f,1.f,
//     //  +X (right)
//      0.5f,-0.5f, 0.5f, 0.f,0.f,
//      0.5f,-0.5f,-0.5f, 1.f,0.f,
//      0.5f, 0.5f,-0.5f, 1.f,1.f,
//      0.5f, 0.5f, 0.5f, 0.f,1.f,
//      //  -X (left)
//      -0.5f,-0.5f,-0.5f, 0.f,0.f,
//      -0.5f,-0.5f, 0.5f, 1.f,0.f,
//      -0.5f, 0.5f, 0.5f, 1.f,1.f,
//      -0.5f, 0.5f,-0.5f, 0.f,1.f,
//      //  +Y (top)
//      -0.5f, 0.5f, 0.5f, 0.f,0.f,
//       0.5f, 0.5f, 0.5f, 1.f,0.f,
//       0.5f, 0.5f,-0.5f, 1.f,1.f,
//      -0.5f, 0.5f,-0.5f, 0.f,1.f,
//      //  -Y (bottom)
//      -0.5f,-0.5f,-0.5f, 0.f,0.f,
//       0.5f,-0.5f,-0.5f, 1.f,0.f,
//       0.5f,-0.5f, 0.5f, 1.f,1.f,
//      -0.5f,-0.5f, 0.5f, 0.f,1.f,
//};
//static const uint32_t cubeIndices[] = {
//    0,1,2, 2,3,0,        // front
//    4,5,6, 6,7,4,        // back
//    8,9,10, 10,11,8,     // right
//    12,13,14, 14,15,12,  // left
//    16,17,18, 18,19,16,  // top
//    20,21,22, 22,23,20   // bottom
//};
//
//// ===================== 인스턴스 데이터 =====================
//constexpr int GRID = 50;
//constexpr int INSTANCE_COUNT = GRID * GRID * GRID;
//struct InstanceData { float offset[3]; uint32_t layer; }; // ★ stride=16B
//std::vector<InstanceData> instances;
//
//void GenerateInstances(uint32_t layerCount) {
//    instances.resize(INSTANCE_COUNT);
//    float spacing = 2.2f; float off = (GRID - 1) * spacing / 2.f;
//    uint32_t i = 0;
//    for (int x = 0; x < GRID; ++x) {
//        for (int y = 0; y < GRID; ++y) {
//            for (int z = 0; z < GRID; ++z) {
//                instances[i].offset[0] = x * spacing - off;
//                instances[i].offset[1] = y * spacing - off;
//                instances[i].offset[2] = z * spacing - off;
//                // ★ 레이어를 랜덤/규칙으로 배정 (여기서는 반복 패턴)
//                instances[i].layer = (x + y + z) % layerCount;
//                ++i;
//            }
//        }
//    }
//}
//
//// ===================== WGSL 셰이더 =====================
//static const char kShader[] = R"(
//struct MVP {
//  model : mat4x4<f32>,
//  view  : mat4x4<f32>,
//  proj  : mat4x4<f32>,
//};
//
//@group(0) @binding(0) var<uniform> mvp : MVP;
//@group(0) @binding(1) var samp : sampler;
//@group(0) @binding(2) var tex2DArray : texture_2d_array<f32>;
//
//struct VSOut {
//  @builtin(position) pos : vec4f,
//  @location(0) uv : vec2f,
//  @location(1) @interpolate(flat) layer : u32,
//};
//
//@vertex
//fn vertexMain(
//  @location(0) inPos : vec3f,
//  @location(1) inUV  : vec2f,
//  @location(2) instOffset : vec3f,
//  @location(3) instLayer  : u32
//) -> VSOut {
//  var o : VSOut;
//  let world = vec4f(inPos + instOffset, 1.0);
//  o.pos = mvp.proj * mvp.view * mvp.model * world;
//  o.uv = inUV;
//  o.layer = instLayer;
//  return o;
//}
//
//@fragment
//fn fragmentMain(@location(0) uv : vec2f, @location(1) @interpolate(flat) layer : u32) -> @location(0) vec4f {
//  // 2D array 텍스처 샘플 (인덱스는 i32)
//  let color = textureSample(tex2DArray, samp, uv, i32(layer));
//  return color;
//}
//)";
//
//// ===================== WebGPU 객체 =====================
//wgpu::Instance instanceW;
//wgpu::Adapter adapter;
//wgpu::Device device;
//wgpu::Surface surface;
//wgpu::TextureFormat format;
//wgpu::Buffer vertexBuffer, indexBuffer, uniformBuffer, instanceBuffer;
//wgpu::BindGroup bindGroup;
//wgpu::BindGroupLayout bindGroupLayout;
//wgpu::RenderPipeline pipeline;
//wgpu::Texture depthTexture;
//wgpu::TextureView depthView;
//
//// ★ 텍스처 배열 + 샘플러
//wgpu::Texture textureArray;
//wgpu::TextureView textureArrayView;
//wgpu::Sampler samplerState;
//
//const uint32_t kWidth = 1200, kHeight = 900;
//
//// 헬퍼
//wgpu::Buffer CreateBuffer(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd;
//    bd.size = size; bd.usage = usage; bd.mappedAtCreation = true;
//    wgpu::Buffer buf = device.CreateBuffer(&bd);
//    std::memcpy(buf.GetMappedRange(), data, size);
//    buf.Unmap();
//    return buf;
//}
//
//void CreateDepthTexture() {
//    wgpu::TextureDescriptor td;
//    td.size = { kWidth, kHeight, 1 };
//    td.format = wgpu::TextureFormat::Depth24Plus;
//    td.usage = wgpu::TextureUsage::RenderAttachment;
//    depthTexture = device.CreateTexture(&td);
//    depthView = depthTexture.CreateView();
//}
//
//void ConfigureSurface() {
//    wgpu::SurfaceCapabilities caps; surface.GetCapabilities(adapter, &caps);
//    format = caps.formats[0];
//    wgpu::SurfaceConfiguration cfg;
//    cfg.device = device; cfg.format = format; cfg.width = kWidth; cfg.height = kHeight;
//    cfg.presentMode = wgpu::PresentMode::Fifo;
//    surface.Configure(&cfg);
//}
//
//// ★ 텍스처 배열 생성(4 레이어, 256x256, RGBA8Unorm), 각 레이어 CPU 패턴 채우기
//constexpr uint32_t TEX_W = 256, TEX_H = 256, TEX_LAYERS = 4;
//void CreateTextureArrayAndSampler() {
//    // Texture
//    wgpu::TextureDescriptor td{};
//    td.size = { TEX_W, TEX_H, TEX_LAYERS };
//    td.mipLevelCount = 1;
//    td.sampleCount = 1;
//    td.dimension = wgpu::TextureDimension::e2D;
//    td.format = wgpu::TextureFormat::RGBA8Unorm;
//    td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
//    textureArray = device.CreateTexture(&td);
//
//    // CPU 데이터 생성
//    std::vector<uint8_t> pixels(TEX_W * TEX_H * 4);
//
//    auto uploadLayer = [&](uint32_t layer) {
//        wgpu::TexelCopyTextureInfo dst{};
//        dst.texture = textureArray;
//        dst.mipLevel = 0;
//        dst.origin = { 0, 0, layer };
//        dst.aspect = wgpu::TextureAspect::All;
//
//        wgpu::TexelCopyBufferLayout layout{};
//        layout.bytesPerRow = TEX_W * 4;
//        layout.rowsPerImage = TEX_H;
//
//        wgpu::Extent3D writeSize = { TEX_W, TEX_H, 1 };
//        device.GetQueue().WriteTexture(&dst, pixels.data(), pixels.size(), &layout, &writeSize);
//    };
//
//    // L0: 체커보드
//    for (uint32_t y = 0; y < TEX_H; ++y) {
//        for (uint32_t x = 0; x < TEX_W; ++x) {
//            bool c = ((x / 16) + (y / 16)) & 1;
//            uint8_t r = c ? 230 : 30, g = c ? 230 : 30, b = c ? 230 : 30;
//            uint32_t i = (y * TEX_W + x) * 4; 
//            pixels[i] = r; 
//            pixels[i + 1] = g; 
//            pixels[i + 2] = b; 
//            pixels[i + 3] = 255;
//        }
//    }
//    uploadLayer(0);
//
//    // L1: 세로 스트라이프
//    for (uint32_t y = 0; y < TEX_H; ++y) {
//        for (uint32_t x = 0; x < TEX_W; ++x) {
//            bool s = (x / 8) & 1;
//            uint8_t r = s ? 20 : 200, g = s ? 200 : 20, b = s ? 20 : 200;
//            uint32_t i = (y * TEX_W + x) * 4; 
//            pixels[i] = r; 
//            pixels[i + 1] = g; 
//            pixels[i + 2] = b; 
//            pixels[i + 3] = 255;
//        }
//    }
//    uploadLayer(1);
//
//    // L2: 그라디언트
//    for (uint32_t y = 0; y < TEX_H; ++y) {
//        for (uint32_t x = 0; x < TEX_W; ++x) {
//            uint8_t r = uint8_t((x * 255) / TEX_W);
//            uint8_t g = uint8_t((y * 255) / TEX_H);
//            uint8_t b = 180;
//            uint32_t i = (y * TEX_W + x) * 4; 
//            pixels[i] = r; 
//            pixels[i + 1] = g; 
//            pixels[i + 2] = b; 
//            pixels[i + 3] = 255;
//        }
//    }
//    uploadLayer(2);
//
//    // L3: 도트
//    for (uint32_t y = 0; y < TEX_H; ++y) {
//        for (uint32_t x = 0; x < TEX_W; ++x) {
//            int dx = int(x % 32) - 16, dy = int(y % 32) - 16;
//            int d2 = dx * dx + dy * dy;
//            bool dot = d2 < 40;
//            uint8_t r = dot ? 255 : 40, g = dot ? 80 : 40, b = dot ? 80 : 200;
//            uint32_t i = (y * TEX_W + x) * 4; 
//            pixels[i] = r; 
//            pixels[i + 1] = g; 
//            pixels[i + 2] = b; 
//            pixels[i + 3] = 255;
//        }
//    }
//    uploadLayer(3);
//
//    // View (2D Array)
//    wgpu::TextureViewDescriptor tvd{};
//    tvd.dimension = wgpu::TextureViewDimension::e2DArray;
//    tvd.format = wgpu::TextureFormat::RGBA8Unorm;
//    tvd.mipLevelCount = 1;
//    tvd.arrayLayerCount = TEX_LAYERS;
//    textureArrayView = textureArray.CreateView(&tvd);
//
//    // Sampler
//    wgpu::SamplerDescriptor sd{};
//    sd.minFilter = wgpu::FilterMode::Linear;
//    sd.magFilter = wgpu::FilterMode::Linear;
//    sd.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
//    sd.addressModeU = wgpu::AddressMode::Repeat;
//    sd.addressModeV = wgpu::AddressMode::Repeat;
//    samplerState = device.CreateSampler(&sd);
//}
//
//void Init() {
//    const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
//    wgpu::InstanceDescriptor id{}; id.requiredFeatureCount = 1; id.requiredFeatures = &kTimedWaitAny;
//    instanceW = wgpu::CreateInstance(&id);
//
//    wgpu::Future f1 = instanceW.RequestAdapter(nullptr, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView m) {
//            if (s != wgpu::RequestAdapterStatus::Success) { std::cerr << "RequestAdapter failed: " << m << "\n"; std::exit(1); }
//            adapter = std::move(a);
//        });
//    instanceW.WaitAny(f1, UINT64_MAX);
//
//    wgpu::DeviceDescriptor dd{};
//    dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView msg) {
//        std::cerr << "Device error(" << int(t) << "): " << msg << "\n";
//        });
//    wgpu::Future f2 = adapter.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView m) {
//            if (s != wgpu::RequestDeviceStatus::Success) { std::cerr << "RequestDevice failed: " << m << "\n"; std::exit(1); }
//            device = std::move(d);
//        });
//    instanceW.WaitAny(f2, UINT64_MAX);
//}
//
//wgpu::BindGroupLayout makeBGL() {
//    // ★ binding(0): MVP UBO
//    wgpu::BindGroupLayoutEntry e0{};
//    e0.binding = 0;
//    e0.visibility = wgpu::ShaderStage::Vertex;
//    e0.buffer.type = wgpu::BufferBindingType::Uniform;
//    e0.buffer.minBindingSize = sizeof(MVP);
//
//    // ★ binding(1): sampler
//    wgpu::BindGroupLayoutEntry e1{};
//    e1.binding = 1;
//    e1.visibility = wgpu::ShaderStage::Fragment;
//    e1.sampler.type = wgpu::SamplerBindingType::Filtering;
//
//    // ★ binding(2): texture 2D array
//    wgpu::BindGroupLayoutEntry e2{};
//    e2.binding = 2;
//    e2.visibility = wgpu::ShaderStage::Fragment;
//    e2.texture.sampleType = wgpu::TextureSampleType::Float;
//    e2.texture.viewDimension = wgpu::TextureViewDimension::e2DArray;
//    e2.texture.multisampled = false;
//
//    wgpu::BindGroupLayoutEntry entries[3] = { e0,e1,e2 };
//    wgpu::BindGroupLayoutDescriptor d{}; d.entryCount = 3; d.entries = entries;
//    return device.CreateBindGroupLayout(&d);
//}
//
//void InitGraphics() {
//    ConfigureSurface();
//    CreateDepthTexture();
//
//    // ★ 텍스처 배열/샘플러 생성
//    CreateTextureArrayAndSampler();
//
//    // ★ 인스턴스 생성 (레이어 수에 맞춰)
//    GenerateInstances(TEX_LAYERS);
//
//    // 버퍼 생성
//    vertexBuffer = CreateBuffer(cubeVertices, sizeof(cubeVertices),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//    indexBuffer = CreateBuffer(cubeIndices, sizeof(cubeIndices),
//        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst);
//    instanceBuffer = CreateBuffer(instances.data(), instances.size() * sizeof(InstanceData),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//
//    MVP mvp{}; uniformBuffer = CreateBuffer(&mvp, sizeof(MVP),
//        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//
//    bindGroupLayout = makeBGL();
//
//    // BindGroup (UBO + Sampler + TextureView)
//    wgpu::BindGroupEntry b0{}; b0.binding = 0; b0.buffer = uniformBuffer; b0.offset = 0; b0.size = sizeof(MVP);
//    wgpu::BindGroupEntry b1{}; b1.binding = 1; b1.sampler = samplerState;
//    wgpu::BindGroupEntry b2{}; b2.binding = 2; b2.textureView = textureArrayView;
//    wgpu::BindGroupEntry bentries[3] = { b0,b1,b2 };
//    wgpu::BindGroupDescriptor bgd{}; bgd.layout = bindGroupLayout; bgd.entryCount = 3; bgd.entries = bentries;
//    bindGroup = device.CreateBindGroup(&bgd);
//
//    // Shader with error checking
//    wgpu::ShaderSourceWGSL wgsl{}; wgsl.code = kShader;
//    wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &wgsl;
//    wgpu::ShaderModule shader = device.CreateShaderModule(&smd);
//    
//    // Add compilation checking
//    std::cout << "Shader module created successfully\n";
//
//    // Vertex Layouts
//    // Slot 0: vertex buffer (pos:location=0, uv:location=1)
//    wgpu::VertexAttribute va[2]{};
//    va[0].format = wgpu::VertexFormat::Float32x3; va[0].offset = 0; va[0].shaderLocation = 0;
//    va[1].format = wgpu::VertexFormat::Float32x2; va[1].offset = sizeof(float) * 3; va[1].shaderLocation = 1;
//
//    wgpu::VertexBufferLayout vbl0{};
//    vbl0.arrayStride = sizeof(float) * 5;
//    vbl0.attributeCount = 2;
//    vbl0.attributes = va;
//    vbl0.stepMode = wgpu::VertexStepMode::Vertex;
//
//    // Slot 1: instance buffer (offset:location=2(vec3f), layer:location=3(u32))
//    wgpu::VertexAttribute ia[2]{};
//    ia[0].format = wgpu::VertexFormat::Float32x3; ia[0].offset = 0; ia[0].shaderLocation = 2;
//    ia[1].format = wgpu::VertexFormat::Uint32;    ia[1].offset = sizeof(float) * 3; ia[1].shaderLocation = 3;
//
//    wgpu::VertexBufferLayout vbl1{};
//    vbl1.arrayStride = sizeof(InstanceData); // 16 bytes
//    vbl1.attributeCount = 2;
//    vbl1.attributes = ia;
//    vbl1.stepMode = wgpu::VertexStepMode::Instance;
//
//    // Color/depth
//    wgpu::ColorTargetState color{}; 
//    color.format = format;
//    color.writeMask = wgpu::ColorWriteMask::All;
//    
//    wgpu::FragmentState fs{}; 
//    fs.module = shader; 
//    fs.entryPoint = "fragmentMain"; 
//    fs.targetCount = 1; 
//    fs.targets = &color;
//
//    wgpu::DepthStencilState ds{};
//    ds.format = wgpu::TextureFormat::Depth24Plus;
//    ds.depthWriteEnabled = true;
//    ds.depthCompare = wgpu::CompareFunction::Less;
//
//    // Pipeline Layout
//    wgpu::PipelineLayoutDescriptor pld{}; 
//    pld.bindGroupLayoutCount = 1; 
//    pld.bindGroupLayouts = &bindGroupLayout;
//    wgpu::PipelineLayout pl = device.CreatePipelineLayout(&pld);
//
//    // Pipeline
//    wgpu::RenderPipelineDescriptor rpd{};
//    rpd.label = "CubeInstancePipeline";  // Add label for debugging
//    rpd.layout = pl;
//    rpd.vertex.module = shader;
//    rpd.vertex.entryPoint = "vertexMain";
//    wgpu::VertexBufferLayout layouts[2] = { vbl0, vbl1 };
//    rpd.vertex.bufferCount = 2;
//    rpd.vertex.buffers = layouts;
//    rpd.fragment = &fs;
//    rpd.depthStencil = &ds;
//    rpd.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
//    rpd.primitive.frontFace = wgpu::FrontFace::CCW;
//    rpd.primitive.cullMode = wgpu::CullMode::Back;
//    rpd.multisample.count = 1;
//    rpd.multisample.mask = 0xFFFFFFFF;
//    rpd.multisample.alphaToCoverageEnabled = false;
//
//    pipeline = device.CreateRenderPipeline(&rpd);
//    
//    if (!pipeline) {
//        std::cerr << "Failed to create render pipeline!\n";
//        std::exit(1);
//    }
//    
//    std::cout << "Render pipeline created successfully\n";
//}
//
//Camera camera;
//float deltaTime = 0.0f, lastFrame = 0.0f;
//bool keys[1024]{};
//
//void KeyCallback(GLFWwindow*, int key, int, int action, int) {
//    if (key >= 0 && key < 1024) { if (action == GLFW_PRESS) keys[key] = true; else if (action == GLFW_RELEASE) keys[key] = false; }
//}
//void MouseCallback(GLFWwindow*, double x, double y) { camera.ProcessMouse((float)x, (float)y); }
//void ScrollCallback(GLFWwindow*, double, double y) { camera.ProcessScroll((float)y); }
//void MouseButtonCallback(GLFWwindow* w, int b, int a, int) {
//    if (b == GLFW_MOUSE_BUTTON_RIGHT && a == GLFW_PRESS) glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
//    if (b == GLFW_MOUSE_BUTTON_LEFT && a == GLFW_PRESS) glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//}
//
//void Render() {
//    float now = (float)glfwGetTime(); deltaTime = now - lastFrame; lastFrame = now;
//    for (int i = 0; i < 1024; ++i) if (keys[i]) camera.ProcessKeyboard(i, deltaTime);
//
//    MVP mvp{};
//    Mat4Identity(mvp.model);
//    float center[3] = { camera.pos[0] + camera.front[0], camera.pos[1] + camera.front[1], camera.pos[2] + camera.front[2] };
//    Mat4LookAt(mvp.view, camera.pos, center, camera.up);
//    Mat4Perspective(mvp.proj, camera.fov, (float)kWidth / (float)kHeight, 0.1f, 1000.f);
//    device.GetQueue().WriteBuffer(uniformBuffer, 0, &mvp, sizeof(MVP));
//
//    wgpu::SurfaceTexture st{}; surface.GetCurrentTexture(&st);
//    wgpu::TextureView backbuffer = st.texture.CreateView();
//
//    wgpu::RenderPassColorAttachment ca{};
//    ca.view = backbuffer; ca.loadOp = wgpu::LoadOp::Clear; ca.storeOp = wgpu::StoreOp::Store;
//    ca.clearValue = { 0.08,0.08,0.12,1.0 };
//
//    wgpu::RenderPassDepthStencilAttachment da{};
//    da.view = depthView; da.depthLoadOp = wgpu::LoadOp::Clear; da.depthStoreOp = wgpu::StoreOp::Store; da.depthClearValue = 1.0f;
//
//    wgpu::RenderPassDescriptor rp{}; rp.colorAttachmentCount = 1; rp.colorAttachments = &ca; rp.depthStencilAttachment = &da;
//
//    wgpu::CommandEncoder enc = device.CreateCommandEncoder();
//    {
//        wgpu::RenderPassEncoder pass = enc.BeginRenderPass(&rp);
//        pass.SetPipeline(pipeline);
//        pass.SetBindGroup(0, bindGroup);
//        pass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize);
//        pass.SetVertexBuffer(1, instanceBuffer, 0, wgpu::kWholeSize);
//        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize);
//        pass.DrawIndexed(36, INSTANCE_COUNT, 0, 0, 0);
//        pass.End();
//    }
//    wgpu::CommandBuffer cmd = enc.Finish();
//    device.GetQueue().Submit(1, &cmd);
//}
//
//int main() {
//    Init();
//    if (!glfwInit()) return -1;
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "WebGPU 3D Cube Instanced (Textured)", nullptr, nullptr);
//    surface = wgpu::glfw::CreateSurfaceForWindow(instanceW, window);
//
//    InitGraphics();
//    glfwSetKeyCallback(window, KeyCallback);
//    glfwSetCursorPosCallback(window, MouseCallback);
//    glfwSetScrollCallback(window, ScrollCallback);
//    glfwSetMouseButtonCallback(window, MouseButtonCallback);
//    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//
//#if defined(__EMSCRIPTEN__)
//    emscripten_set_main_loop(Render, 0, false);
//#else
//    while (!glfwWindowShouldClose(window)) {
//        glfwPollEvents();
//        Render();
//        surface.Present();
//        instanceW.ProcessEvents();
//    }
//    glfwDestroyWindow(window);
//    glfwTerminate();
//#endif
//    return 0;
//}