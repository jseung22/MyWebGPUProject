//// ===================== WebGPU + GLFW: Instanced Cubes w/ Single Texture Atlas (4x4) =====================
//// - Uses a single atlas: floatplane\textures\particle_atlas.png  (fallback: floatplane/textures/particle_atlas.png)
//// - Randomly assigns one of 16 tiles (4x4) to each cube instance
//// - Per-instance data: offset(vec3) + atlasIndex(u32)
//// Build: link GLFW, Dawn/WebGPU C++ libs, C++17 (for <filesystem>)
//
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <chrono>
//#include <cstring>
//#include <cstdint>
//#include <string>
//#include <random>
//#include <filesystem>
//
//#include <GLFW/glfw3.h>
//#if defined(__EMSCRIPTEN__)
//#include <emscripten/emscripten.h>
//#endif
//
//#include <dawn/webgpu_cpp_print.h>
//#include <webgpu/webgpu_cpp.h>
//#include <webgpu/webgpu_glfw.h>
//
//// ===================== stb_image (PNG loader) =====================
//#define STB_IMAGE_IMPLEMENTATION
//#define STBI_ONLY_PNG
//#include "stb_image.h"
//
//// ===================== Camera / Math =====================
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
//    void ProcessScroll(float yoffset) { fov -= yoffset; if (fov < 1.0f) fov = 1.0f; if (fov > 45.0f) fov = 45.0f; }
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
//
//struct MVP { float model[16]; float view[16]; float proj[16]; };
//
//// ===================== Cube geometry (pos + uv) =====================
//static const float cubeVertices[] = {
//    // +Z
//    -0.5f,-0.5f, 0.5f, 0.f,0.f,  0.5f,-0.5f, 0.5f, 1.f,0.f,  0.5f, 0.5f, 0.5f, 1.f,1.f, -0.5f, 0.5f, 0.5f, 0.f,1.f,
//    // -Z
//     0.5f,-0.5f,-0.5f, 0.f,0.f, -0.5f,-0.5f,-0.5f, 1.f,0.f, -0.5f, 0.5f,-0.5f, 1.f,1.f,  0.5f, 0.5f,-0.5f, 0.f,1.f,
//     // +X
//      0.5f,-0.5f, 0.5f, 0.f,0.f,  0.5f,-0.5f,-0.5f, 1.f,0.f,  0.5f, 0.5f,-0.5f, 1.f,1.f,  0.5f, 0.5f, 0.5f, 0.f,1.f,
//      // -X
//      -0.5f,-0.5f,-0.5f, 0.f,0.f, -0.5f,-0.5f, 0.5f, 1.f,0.f, -0.5f, 0.5f, 0.5f, 1.f,1.f, -0.5f, 0.5f,-0.5f, 0.f,1.f,
//      // +Y
//      -0.5f, 0.5f, 0.5f, 0.f,0.f,  0.5f, 0.5f, 0.5f, 1.f,0.f,  0.5f, 0.5f,-0.5f, 1.f,1.f, -0.5f, 0.5f,-0.5f, 0.f,1.f,
//      // -Y
//      -0.5f,-0.5f,-0.5f, 0.f,0.f,  0.5f,-0.5f,-0.5f, 1.f,0.f,  0.5f,-0.5f, 0.5f, 1.f,1.f, -0.5f,-0.5f, 0.5f, 0.f,1.f,
//};
//static const uint32_t cubeIndices[] = {
//    0,1,2,2,3,0, 4,5,6,6,7,4, 8,9,10,10,11,8,
//    12,13,14,14,15,12, 16,17,18,18,19,16, 20,21,22,22,23,20
//};
//
//// ===================== Instances =====================
//constexpr int GRID = 50;
//constexpr int INSTANCE_COUNT = GRID * GRID * GRID;
//
//// 16-byte instance (std140-friendly): offset.xyz + atlasIndex
//struct InstanceData { float offset[3]; uint32_t atlasIndex; };
//std::vector<InstanceData> instances;
//
//// Atlas layout (4x4)
//static constexpr uint32_t ATLAS_COLS = 4;
//static constexpr uint32_t ATLAS_ROWS = 4;
//static constexpr uint32_t ATLAS_TILES = ATLAS_COLS * ATLAS_ROWS;
//
//void GenerateInstances() {
//    instances.resize(INSTANCE_COUNT);
//    float spacing = 2.2f;
//    float off = (GRID - 1) * spacing / 2.f;
//
//    // random index for visual variety; deterministic seed if desired
//    std::mt19937 rng(42);
//    std::uniform_int_distribution<uint32_t> dist(0, ATLAS_TILES - 1);
//
//    uint32_t i = 0;
//    for (int x = 0; x < GRID; ++x) {
//        for (int y = 0; y < GRID; ++y) {
//            for (int z = 0; z < GRID; ++z) {
//                instances[i].offset[0] = x * spacing - off;
//                instances[i].offset[1] = y * spacing - off;
//                instances[i].offset[2] = z * spacing - off;
//                instances[i].atlasIndex = dist(rng); // random [0..15]
//                ++i;
//            }
//        }
//    }
//}
//
//// ===================== WGSL =====================
//static const char kShader[] = R"(
//struct MVP { model:mat4x4<f32>, view:mat4x4<f32>, proj:mat4x4<f32>, };
//
//@group(0) @binding(0) var<uniform> mvp : MVP;
//@group(0) @binding(1) var samp : sampler;
//@group(0) @binding(2) var atlasTex : texture_2d<f32>;
//
//struct VSOut {
//  @builtin(position) pos : vec4f,
//  @location(0) uv : vec2f,
//  @location(1) @interpolate(flat) atlasIndex : u32,
//};
//
//@vertex
//fn vertexMain(
//  @location(0) inPos : vec3f,
//  @location(1) inUV  : vec2f,
//  @location(2) instOffset : vec3f,
//  @location(3) instAtlasIndex : u32
//) -> VSOut {
//  var o : VSOut;
//  let world = vec4f(inPos + instOffset, 1.0);
//  o.pos = mvp.proj * mvp.view * mvp.model * world;
//  o.uv = inUV;
//  o.atlasIndex = instAtlasIndex;
//  return o;
//}
//
//@fragment
//fn fragmentMain(@location(0) uv : vec2f,
//                @location(1) @interpolate(flat) atlasIndex : u32) -> @location(0) vec4f {
//  // Atlas params (4x4). If you want runtime-configurable, move to a uniform.
//  let cols : u32 = 4u;
//  let rows : u32 = 4u;
//  let scale = vec2f(1.0 / f32(cols), 1.0 / f32(rows));
//
//  let col = f32(atlasIndex % cols);
//  let row = f32(atlasIndex / cols);
//
//  // note: WebGPU UV (0,0) is top-left; this matches our PNG layout.
//  let tileOffset = vec2f(col, row) * scale;
//  let uvAtlas = uv * scale + tileOffset;
//
//  return textureSample(atlasTex, samp, uvAtlas);
//}
//)";
//
//// ===================== WebGPU globals =====================
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
//// atlas texture + sampler
//wgpu::Texture atlasTexture;
//wgpu::TextureView atlasView;
//wgpu::Sampler samplerState;
//
//const uint32_t kWidth = 1200, kHeight = 900;
//
//// ===================== Buffer helper =====================
//wgpu::Buffer CreateBuffer(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd; bd.size = size; bd.usage = usage; bd.mappedAtCreation = true;
//    auto buf = device.CreateBuffer(&bd);
//    std::memcpy(buf.GetMappedRange(), data, size);
//    buf.Unmap();
//    return buf;
//}
//
//// ===================== Depth/Surface =====================
//void CreateDepthTexture() {
//    wgpu::TextureDescriptor td{};
//    td.size = { kWidth, kHeight, 1 };
//    td.format = wgpu::TextureFormat::Depth24Plus;
//    td.usage = wgpu::TextureUsage::RenderAttachment;
//    depthTexture = device.CreateTexture(&td);
//    depthView = depthTexture.CreateView();
//}
//void ConfigureSurface() {
//    wgpu::SurfaceCapabilities caps; surface.GetCapabilities(adapter, &caps);
//    format = caps.formats[0];
//    wgpu::SurfaceConfiguration cfg{};
//    cfg.device = device; cfg.format = format; cfg.width = kWidth; cfg.height = kHeight;
//    cfg.presentMode = wgpu::PresentMode::Fifo;
//    surface.Configure(&cfg);
//}
//
//// ===================== PNG load (atlas) & upload =====================
//struct LoadedImage { int w = 0, h = 0; std::vector<uint8_t> rgba; };
//
//LoadedImage LoadPNG_RGBA(const std::filesystem::path& p) {
//    LoadedImage out;
//    int w, h, n;
//    
//    std::cout << "Attempting to load PNG: " << p << std::endl;
//    
//    stbi_uc* data = stbi_load(p.string().c_str(), &w, &h, &n, 4); // RGBA
//    if (!data) {
//        std::cerr << "Failed to load PNG: " << p << std::endl;
//        std::cerr << "stbi_failure_reason: " << stbi_failure_reason() << std::endl;
//        return out;
//    }
//    
//    std::cout << "PNG loaded successfully: " << w << "x" << h << ", channels: " << n << std::endl;
//    
//    out.w = w; out.h = h;
//    out.rgba.assign(data, data + (w * h * 4));
//    stbi_image_free(data);
//    return out;
//}
//
//void CreateAtlasTextureFromPNG(const std::filesystem::path& preferredPath) {
//    namespace fs = std::filesystem;
//    
//    // Try multiple possible paths, including relative to source directory
//    std::vector<fs::path> possiblePaths = {
//        preferredPath,
//        "floatplane/textures/particle_atlas.png",
//        "floatplane\\textures\\particle_atlas.png",
//        // Look in parent directories to find the source
//        "../../../floatplane/textures/particle_atlas.png",
//        "../../../floatplane\\textures\\particle_atlas.png",
//        "../../floatplane/textures/particle_atlas.png", 
//        "../../floatplane\\textures\\particle_atlas.png",
//        "../floatplane/textures/particle_atlas.png",
//        "../floatplane\\textures\\particle_atlas.png",
//        // Absolute paths
//        fs::current_path() / "floatplane" / "textures" / "particle_atlas.png",
//        fs::current_path().parent_path() / "floatplane" / "textures" / "particle_atlas.png",
//        fs::current_path().parent_path().parent_path() / "floatplane" / "textures" / "particle_atlas.png",
//        fs::current_path().parent_path().parent_path().parent_path() / "floatplane" / "textures" / "particle_atlas.png"
//    };
//    
//    fs::path path;
//    bool found = false;
//    
//    // Debug: print current working directory
//    std::cout << "Current working directory: " << fs::current_path() << std::endl;
//    
//    for (const auto& testPath : possiblePaths) {
//        std::cout << "Trying path: " << testPath << " -> exists: " << fs::exists(testPath) << std::endl;
//        if (fs::exists(testPath)) {
//            path = testPath;
//            found = true;
//            std::cout << "Found atlas at: " << path << std::endl;
//            break;
//        }
//    }
//    
//    if (!found) {
//        std::cerr << "Atlas image not found. Tried paths:\n";
//        for (const auto& testPath : possiblePaths) {
//            std::cerr << "  - " << testPath << "\n";
//        }
//        std::exit(1);
//    }
//    
//    auto img = LoadPNG_RGBA(path);
//    if (img.w == 0 || img.h == 0) {
//        std::cerr << "Failed to load atlas image from: " << path << "\n";
//        std::exit(1);
//    }
//    
//    std::cout << "Successfully loaded atlas: " << img.w << "x" << img.h << " pixels from " << path << std::endl;
//
//    // Create texture
//    wgpu::TextureDescriptor td{};
//    td.size = { (uint32_t)img.w, (uint32_t)img.h, 1 };
//    td.mipLevelCount = 1;
//    td.sampleCount = 1;
//    td.dimension = wgpu::TextureDimension::e2D;
//    td.format = wgpu::TextureFormat::RGBA8Unorm;
//    td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
//    atlasTexture = device.CreateTexture(&td);
//
//    // Upload
//    wgpu::TexelCopyTextureInfo dst{};
//    dst.texture = atlasTexture;
//    dst.mipLevel = 0;
//    dst.origin = { 0,0,0 };
//    dst.aspect = wgpu::TextureAspect::All;
//
//    wgpu::TexelCopyBufferLayout layout{};
//    layout.bytesPerRow = (uint32_t)img.w * 4;
//    layout.rowsPerImage = (uint32_t)img.h;
//
//    wgpu::Extent3D size{ (uint32_t)img.w, (uint32_t)img.h, 1 };
//    device.GetQueue().WriteTexture(&dst, img.rgba.data(), img.rgba.size(), &layout, &size);
//
//    // View
//    wgpu::TextureViewDescriptor tvd{};
//    tvd.dimension = wgpu::TextureViewDimension::e2D;
//    tvd.format = wgpu::TextureFormat::RGBA8Unorm;
//    tvd.mipLevelCount = 1;
//    tvd.arrayLayerCount = 1;
//    atlasView = atlasTexture.CreateView(&tvd);
//
//    // Sampler
//    wgpu::SamplerDescriptor sd{};
//    sd.minFilter = wgpu::FilterMode::Linear;
//    sd.magFilter = wgpu::FilterMode::Linear;
//    sd.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
//    sd.addressModeU = wgpu::AddressMode::ClampToEdge; // atlas 경계 보존
//    sd.addressModeV = wgpu::AddressMode::ClampToEdge;
//    samplerState = device.CreateSampler(&sd);
//}
//
//// ===================== Init / Pipeline =====================
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
//    wgpu::BindGroupLayoutEntry e0{}; // UBO
//    e0.binding = 0; e0.visibility = wgpu::ShaderStage::Vertex; e0.buffer.type = wgpu::BufferBindingType::Uniform; e0.buffer.minBindingSize = sizeof(MVP);
//    wgpu::BindGroupLayoutEntry e1{}; // sampler
//    e1.binding = 1; e1.visibility = wgpu::ShaderStage::Fragment; e1.sampler.type = wgpu::SamplerBindingType::Filtering;
//    wgpu::BindGroupLayoutEntry e2{}; // texture2D
//    e2.binding = 2; e2.visibility = wgpu::ShaderStage::Fragment; e2.texture.sampleType = wgpu::TextureSampleType::Float; e2.texture.viewDimension = wgpu::TextureViewDimension::e2D;
//
//    wgpu::BindGroupLayoutEntry entries[3] = { e0,e1,e2 };
//    wgpu::BindGroupLayoutDescriptor d{}; d.entryCount = 3; d.entries = entries;
//    return device.CreateBindGroupLayout(&d);
//}
//
//wgpu::RenderPipeline buildPipeline(wgpu::ShaderModule shader, wgpu::PipelineLayout pl) {
//    // Vertex layouts
//    wgpu::VertexAttribute va[2]{};
//    va[0].format = wgpu::VertexFormat::Float32x3; va[0].offset = 0; va[0].shaderLocation = 0;
//    va[1].format = wgpu::VertexFormat::Float32x2; va[1].offset = sizeof(float) * 3; va[1].shaderLocation = 1;
//    wgpu::VertexBufferLayout vbl0{}; vbl0.arrayStride = sizeof(float) * 5; vbl0.attributeCount = 2; vbl0.attributes = va; vbl0.stepMode = wgpu::VertexStepMode::Vertex;
//
//    // Instance buffer: offset(vec3), atlasIndex(u32)
//    wgpu::VertexAttribute ia[2]{};
//    ia[0].format = wgpu::VertexFormat::Float32x3; ia[0].offset = 0;                  ia[0].shaderLocation = 2;
//    ia[1].format = wgpu::VertexFormat::Uint32;    ia[1].offset = sizeof(float) * 3;    ia[1].shaderLocation = 3;
//    wgpu::VertexBufferLayout vbl1{}; vbl1.arrayStride = sizeof(InstanceData); vbl1.attributeCount = 2; vbl1.attributes = ia; vbl1.stepMode = wgpu::VertexStepMode::Instance;
//
//    wgpu::ColorTargetState color{}; color.format = format; color.writeMask = wgpu::ColorWriteMask::All;
//    wgpu::FragmentState fs{}; fs.module = shader; fs.entryPoint = "fragmentMain"; fs.targetCount = 1; fs.targets = &color;
//
//    wgpu::DepthStencilState ds{}; ds.format = wgpu::TextureFormat::Depth24Plus; ds.depthWriteEnabled = true; ds.depthCompare = wgpu::CompareFunction::Less;
//
//    wgpu::RenderPipelineDescriptor rpd{};
//    rpd.label = "InstancedCubes_Atlas";
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
//    rpd.multisample.count = 1; rpd.multisample.mask = 0xFFFFFFFF; rpd.multisample.alphaToCoverageEnabled = false;
//
//    return device.CreateRenderPipeline(&rpd);
//}
//
//void InitGraphics() {
//    ConfigureSurface();
//    CreateDepthTexture();
//
//    // Load atlas from disk
//    CreateAtlasTextureFromPNG(std::filesystem::path("floatplane\\textures\\particle_atlas.png"));
//
//    // Instances
//    GenerateInstances();
//
//    // Buffers
//    vertexBuffer = CreateBuffer(cubeVertices, sizeof(cubeVertices), wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//    indexBuffer = CreateBuffer(cubeIndices, sizeof(cubeIndices), wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst);
//    instanceBuffer = CreateBuffer(instances.data(), instances.size() * sizeof(InstanceData), wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//    MVP mvp{};     uniformBuffer = CreateBuffer(&mvp, sizeof(MVP), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//
//    // Bindings
//    bindGroupLayout = makeBGL();
//    wgpu::BindGroupEntry b0{}; b0.binding = 0; b0.buffer = uniformBuffer; b0.offset = 0; b0.size = sizeof(MVP);
//    wgpu::BindGroupEntry b1{}; b1.binding = 1; b1.sampler = samplerState;
//    wgpu::BindGroupEntry b2{}; b2.binding = 2; b2.textureView = atlasView;
//    wgpu::BindGroupEntry ben[3] = { b0,b1,b2 };
//    wgpu::BindGroupDescriptor bgd{}; bgd.layout = bindGroupLayout; bgd.entryCount = 3; bgd.entries = ben;
//    bindGroup = device.CreateBindGroup(&bgd);
//
//    // Shader/Pipeline
//    wgpu::ShaderSourceWGSL wgsl{}; wgsl.code = kShader;
//    wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &wgsl;
//    wgpu::ShaderModule shader = device.CreateShaderModule(&smd);
//    wgpu::PipelineLayoutDescriptor pld{}; pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bindGroupLayout;
//    wgpu::PipelineLayout pl = device.CreatePipelineLayout(&pld);
//    pipeline = buildPipeline(shader, pl);
//
//    if (!pipeline) { std::cerr << "Failed to create pipeline\n"; std::exit(1); }
//}
//
//// ===================== Input & Render =====================
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
//    MVP mvp{}; Mat4Identity(mvp.model);
//    float center[3] = { camera.pos[0] + camera.front[0], camera.pos[1] + camera.front[1], camera.pos[2] + camera.front[2] };
//    Mat4LookAt(mvp.view, camera.pos, center, camera.up);
//    Mat4Perspective(mvp.proj, camera.fov, (float)kWidth / (float)kHeight, 0.1f, 1000.f);
//    device.GetQueue().WriteBuffer(uniformBuffer, 0, &mvp, sizeof(MVP));
//
//    wgpu::SurfaceTexture st{}; surface.GetCurrentTexture(&st);
//    auto backbuffer = st.texture.CreateView();
//
//    wgpu::RenderPassColorAttachment ca{}; ca.view = backbuffer; ca.loadOp = wgpu::LoadOp::Clear; ca.storeOp = wgpu::StoreOp::Store; ca.clearValue = { 0.08,0.08,0.12,1.0 };
//    wgpu::RenderPassDepthStencilAttachment da{}; da.view = depthView; da.depthLoadOp = wgpu::LoadOp::Clear; da.depthStoreOp = wgpu::StoreOp::Store; da.depthClearValue = 1.0f;
//
//    wgpu::RenderPassDescriptor rp{}; rp.colorAttachmentCount = 1; rp.colorAttachments = &ca; rp.depthStencilAttachment = &da;
//
//    auto enc = device.CreateCommandEncoder();
//    {
//        auto pass = enc.BeginRenderPass(&rp);
//        pass.SetPipeline(pipeline);
//        pass.SetBindGroup(0, bindGroup);
//        pass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize);
//        pass.SetVertexBuffer(1, instanceBuffer, 0, wgpu::kWholeSize);
//        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize);
//        pass.DrawIndexed(36, INSTANCE_COUNT, 0, 0, 0);
//        pass.End();
//    }
//    auto cmd = enc.Finish();
//    device.GetQueue().Submit(1, &cmd);
//}
//
//// ===================== Main =====================
//int main() {
//    Init();
//    if (!glfwInit()) return -1;
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(1200, 900, "Instanced Cubes (Single Texture Atlas 4x4)", nullptr, nullptr);
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
