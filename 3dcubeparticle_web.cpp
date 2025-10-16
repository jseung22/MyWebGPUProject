//// ===================== WebGPU + GLFW: 1,000,000 Cube Particles via SSBO + Compute (fixed UBO layout) =====================
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
//#define STB_IMAGE_IMPLEMENTATION
//#define STBI_ONLY_PNG
//#include "stb_image.h"
//
//// ===================== Camera / Math =====================
//struct Camera {
//    float pos[3] = { 0.0f, 0.0f, 220.0f };
//    float front[3] = { 0.0f, 0.0f, -1.0f };
//    float up[3] = { 0.0f, 1.0f, 0.0f };
//    float yaw = -90.0f, pitch = 0.0f;
//    float fov = 45.0f;
//    float lastX = 500.0f, lastY = 500.0f;
//    bool firstMouse = true;
//    static void Cross(const float* a, const float* b, float* o) { o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0]; }
//    static void Normalize(float* v) { float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (l > 1e-5f) { v[0] /= l; v[1] /= l; v[2] /= l; } }
//    void Move(const float* d, float a) { for (int i = 0; i < 3; ++i) pos[i] += d[i] * a; }
//    void UpdateFront() {
//        float ry = yaw * 3.14159265f / 180.0f, rp = pitch * 3.14159265f / 180.0f;
//        front[0] = cosf(ry) * cosf(rp); front[1] = sinf(rp); front[2] = sinf(ry) * cosf(rp);
//        Normalize(front);
//    }
//    void ProcessKeyboard(int key, float dt) {
//        float speed = 90.0f * dt;
//        float right[3]; Cross(front, up, right); Normalize(right);
//        if (key == GLFW_KEY_W) Move(front, speed);
//        if (key == GLFW_KEY_S) Move(front, -speed);
//        if (key == GLFW_KEY_A) Move(right, -speed);
//        if (key == GLFW_KEY_D) Move(right, speed);
//        if (key == GLFW_KEY_Q) pos[1] -= speed;
//        if (key == GLFW_KEY_E) pos[1] += speed;
//    }
//    void ProcessMouse(float xpos, float ypos) {
//        if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
//        float xoffset = (xpos - lastX) * 0.1f, yoffset = (lastY - ypos) * 0.1f;
//        lastX = xpos; lastY = ypos;
//        yaw += xoffset; pitch += yoffset;
//        if (pitch > 89.0f) pitch = 89.0f;
//        if (pitch < -89.0f) pitch = -89.0f;
//        UpdateFront();
//    }
//    void ProcessScroll(float yoffset) {
//        fov -= yoffset; if (fov < 1.0f) fov = 1.0f; if (fov > 70.0f) fov = 70.0f;
//    }
//};
//
//void Mat4Identity(float* m) { std::memset(m, 0, sizeof(float) * 16); m[0] = m[5] = m[10] = m[15] = 1.0f; }
//void Mat4Perspective(float* m, float fov, float aspect, float n, float f) {
//    float t = tanf(fov * 0.5f * 3.14159265f / 180.0f);
//    std::memset(m, 0, sizeof(float) * 16);
//    m[0] = 1.f / (aspect * t); m[5] = 1.f / t; m[10] = -(f + n) / (f - n); m[11] = -1.f; m[14] = -(2.f * f * n) / (f - n);
//}
//void Mat4LookAt(float* m, const float* e, const float* c, const float* up) {
//    float fwd[3] = { c[0] - e[0], c[1] - e[1], c[2] - e[2] }; Camera::Normalize(fwd);
//    float s[3]; Camera::Cross(fwd, up, s); Camera::Normalize(s);
//    float u[3]; Camera::Cross(s, fwd, u);
//    Mat4Identity(m);
//    m[0] = s[0]; m[1] = u[0]; m[2] = -fwd[0];
//    m[4] = s[1]; m[5] = u[1]; m[6] = -fwd[1];
//    m[8] = s[2]; m[9] = u[2]; m[10] = -fwd[2];
//    m[12] = -(s[0] * e[0] + s[1] * e[1] + s[2] * e[2]);
//    m[13] = -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]);
//    m[14] = fwd[0] * e[0] + fwd[1] * e[1] + fwd[2] * e[2];
//}
//
//struct MVP {
//    float model[16];
//    float view[16];
//    float proj[16];
//    float camPos[4];
//};
//
//// ======= SimUBO: std140 정렬을 위한 vec4 묶음 (총 48B, 16B 정렬) =======
//struct alignas(16) SimUBO {
//    float dt_time_bounds_damping[4]; // (dt, time, bounds, damping)
//    float accel_jitter[4];           // (accel.x, accel.y, accel.z, jitter)
//    uint32_t count;                  // 파티클 개수
//    uint32_t _pad0 = 0;              // 패딩으로 16B 정렬 보장
//    uint32_t _pad1 = 0;
//    uint32_t _pad2 = 0;
//};
//
//// ===================== Cube geometry =====================
//static const float cubeVertices[] = {
//    -0.5f,-0.5f, 0.5f, 0,0,1,  0,0,
//     0.5f,-0.5f, 0.5f, 0,0,1,  1,0,
//     0.5f, 0.5f, 0.5f, 0,0,1,  1,1,
//    -0.5f, 0.5f, 0.5f, 0,0,1,  0,1,
//     0.5f,-0.5f,-0.5f, 0,0,-1, 0,0,
//    -0.5f,-0.5f,-0.5f, 0,0,-1, 1,0,
//    -0.5f, 0.5f,-0.5f, 0,0,-1, 1,1,
//     0.5f, 0.5f,-0.5f, 0,0,-1, 0,1,
//     0.5f,-0.5f, 0.5f, 1,0,0,  0,0,
//     0.5f,-0.5f,-0.5f,1,0,0,  1,0,
//     0.5f, 0.5f,-0.5f,1,0,0,  1,1,
//     0.5f, 0.5f, 0.5f,1,0,0,  0,1,
//    -0.5f,-0.5f,-0.5f,-1,0,0, 0,0,
//    -0.5f,-0.5f, 0.5f,-1,0,0, 1,0,
//    -0.5f, 0.5f, 0.5f,-1,0,0, 1,1,
//    -0.5f, 0.5f,-0.5f,-1,0,0, 0,1,
//    -0.5f, 0.5f, 0.5f,0,1,0,  0,0,
//     0.5f, 0.5f, 0.5f,0,1,0,  1,0,
//     0.5f, 0.5f,-0.5f,0,1,0,  1,1,
//    -0.5f, 0.5f,-0.5f,0,1,0,  0,1,
//    -0.5f,-0.5f,-0.5f,0,-1,0, 0,0,
//     0.5f,-0.5f,-0.5f,0,-1,0, 1,0,
//     0.5f,-0.5f, 0.5f,0,-1,0,  1,1,
//    -0.5f,-0.5f, 0.5f,0,-1,0,  0,1,
//};
//static const uint32_t cubeIndices[] = {
//    0,1,2,2,3,0, 4,5,6,6,7,4, 8,9,10,10,11,8,
//    12,13,14,14,15,12, 16,17,18,18,19,16, 20,21,22,22,23,20
//};
//
//// ===================== Particles (1,000,000) =====================
//constexpr uint32_t PARTICLE_COUNT = 1'000'000;
//
//struct ParticleCPU {
//    float pos[4];       // xyz + randSeed
//    float vel[4];       // xyz + speedScale
//    float lightDir[4];  // xyz + pad
//    float lightColor[4];// rgb + pad
//    float atlasShiny[4];// rand, atlasIndex(float), shininess, pad
//};
//std::vector<ParticleCPU> gParticles;
//
//// ===================== WebGPU globals =====================
//wgpu::Instance instanceW;
//wgpu::Adapter adapter;
//wgpu::Device device;
//wgpu::Surface surface;
//wgpu::TextureFormat format;
//wgpu::Texture depthTexture;
//wgpu::TextureView depthView;
//
//wgpu::Texture atlasTexture;
//wgpu::TextureView atlasView;
//wgpu::Sampler samplerState;
//
//wgpu::Buffer vertexBuffer, indexBuffer, uniformBuffer, simUniformBuffer;
//wgpu::Buffer particleBuffer;
//
//wgpu::BindGroupLayout renderBGL, computeBGL;
//wgpu::BindGroup renderBG, computeBG;
//wgpu::RenderPipeline renderPipeline;
//wgpu::ComputePipeline computePipeline;
//
//const uint32_t kWidth = 1280, kHeight = 900;
//
//// ===================== Buffer helpers =====================
//wgpu::Buffer CreateBuffer(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd; bd.size = size; bd.usage = usage; bd.mappedAtCreation = true;
//    auto buf = device.CreateBuffer(&bd);
//    if (data) std::memcpy(buf.GetMappedRange(), data, size);
//    buf.Unmap();
//    return buf;
//}
//
//// ===================== Depth/Surface =====================
//void CreateDepthTexture() {
//    wgpu::TextureDescriptor td{}; td.size = { kWidth,kHeight,1 }; td.format = wgpu::TextureFormat::Depth24Plus;
//    td.usage = wgpu::TextureUsage::RenderAttachment;
//    depthTexture = device.CreateTexture(&td);
//    depthView = depthTexture.CreateView();
//}
//void ConfigureSurface() {
//    wgpu::SurfaceCapabilities caps; surface.GetCapabilities(adapter, &caps);
//    format = caps.formats[0];
//    wgpu::SurfaceConfiguration cfg{}; cfg.device = device; cfg.format = format; cfg.width = kWidth; cfg.height = kHeight; cfg.presentMode = wgpu::PresentMode::Fifo;
//    surface.Configure(&cfg);
//}
//
//// ===================== PNG load =====================
//struct LoadedImage { int w = 0, h = 0; std::vector<uint8_t> rgba; };
//
//LoadedImage LoadPNG_RGBA(const std::filesystem::path& p) {
//    LoadedImage out; int w, h, n;
//    stbi_uc* data = stbi_load(p.string().c_str(), &w, &h, &n, 4);
//    if (!data) { std::cerr << "Failed to load PNG: " << p << "\n"; return out; }
//    out.w = w; out.h = h; out.rgba.assign(data, data + (w * h * 4)); stbi_image_free(data); return out;
//}
//void CreateAtlasTextureFromPNG(const std::filesystem::path& preferred) {
//    namespace fs = std::filesystem;
//    fs::path path;
//
//#if defined(__EMSCRIPTEN__)
//    // 웹에서는 가상 FS의 고정 경로만 사용
//    path = "/assets/particle_atlas.png";
//#else
//    // 네이티브에서는 기존 후보 경로들 탐색
//    std::vector<fs::path> candidates = {
//        preferred,
//        "floatplane/textures/particle_atlas.png",
//        "floatplane\\textures\\particle_atlas.png",
//        "../../../floatplane/textures/particle_atlas.png",
//        "../../../floatplane\\textures\\particle_atlas.png",
//        "../../floatplane/textures/particle_atlas.png",
//        "../../floatplane\\textures\\particle_atlas.png",
//        "../floatplane/textures/particle_atlas.png",
//        "../floatplane\\textures\\particle_atlas.png",
//        fs::current_path() / "floatplane" / "textures" / "particle_atlas.png",
//        fs::current_path().parent_path() / "floatplane" / "textures" / "particle_atlas.png"
//    };
//    for (auto& c : candidates) { if (fs::exists(c)) { path = c; break; } }
//#endif
//    if (path.empty()) { std::cerr << "Atlas image not found.\n"; std::exit(1); }
//
//    auto img = LoadPNG_RGBA(path);
//    if (img.w == 0 || img.h == 0) { std::cerr << "Invalid atlas image.\n"; std::exit(1); }
//
//    wgpu::TextureDescriptor td{}; td.size = { (uint32_t)img.w,(uint32_t)img.h,1 }; td.mipLevelCount = 1; td.sampleCount = 1;
//    td.dimension = wgpu::TextureDimension::e2D; td.format = wgpu::TextureFormat::RGBA8Unorm;
//    td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
//    atlasTexture = device.CreateTexture(&td);
//
//    wgpu::TexelCopyTextureInfo dst{}; dst.texture = atlasTexture; dst.mipLevel = 0; dst.origin = { 0,0,0 }; dst.aspect = wgpu::TextureAspect::All;
//    wgpu::TexelCopyBufferLayout layout{}; layout.bytesPerRow = (uint32_t)img.w * 4; layout.rowsPerImage = (uint32_t)img.h;
//    wgpu::Extent3D size{ (uint32_t)img.w,(uint32_t)img.h,1 };
//    device.GetQueue().WriteTexture(&dst, img.rgba.data(), img.rgba.size(), &layout, &size);
//
//    wgpu::TextureViewDescriptor tvd{}; tvd.dimension = wgpu::TextureViewDimension::e2D; tvd.format = wgpu::TextureFormat::RGBA8Unorm;
//    tvd.mipLevelCount = 1; tvd.arrayLayerCount = 1;
//    atlasView = atlasTexture.CreateView(&tvd);
//
//    wgpu::SamplerDescriptor sd{}; sd.minFilter = wgpu::FilterMode::Linear; sd.magFilter = wgpu::FilterMode::Linear; sd.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
//    sd.addressModeU = wgpu::AddressMode::ClampToEdge; sd.addressModeV = wgpu::AddressMode::ClampToEdge;
//    samplerState = device.CreateSampler(&sd);
//}
//
//// ===================== WGSL =====================
//static const char kComputeWGSL[] = R"(
//struct Particle {
//  pos        : vec4<f32>,
//  vel        : vec4<f32>,
//  lightDir   : vec4<f32>,
//  lightColor : vec4<f32>,
//  atlasShiny : vec4<f32>,
//}
//
//struct Particles { 
//  data : array<Particle>,
//}
//
//@group(0) @binding(0) var<storage, read_write> particles : Particles;
//
//// ---- std140-friendly UBO (총 48B) ----
//struct SimUBO {
//  dt_time_bounds_damping : vec4<f32>, // (dt, time, bounds, damping)
//  accel_jitter           : vec4<f32>, // (accel.xyz, jitter)
//  count                  : u32,
//  _pad0                  : u32,
//  _pad1                  : u32,
//  _pad2                  : u32,
//}
//
//@group(0) @binding(1) var<uniform> sim : SimUBO;
//
//@compute @workgroup_size(256)
//fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
//  let i = gid.x;
//  if (i >= sim.count) { 
//    return; 
//  }
//
//  var P = particles.data[i];
//
//  let dt = sim.dt_time_bounds_damping.x;
//  let bounds = sim.dt_time_bounds_damping.z;
//
//  // 단순한 직선등속 운동: 위치 = 위치 + 속도 * 시간
//  var pos = P.pos.xyz + P.vel.xyz * dt;
//
//  // 화면 경계에서 바운스 처리
//  for (var a: i32 = 0; a < 3; a++) {
//    if (pos[a] > bounds) { 
//      pos[a] = bounds;  
//      P.vel[a] = -abs(P.vel[a]); // 음의 방향으로 바운스
//    }
//    if (pos[a] < -bounds) { 
//      pos[a] = -bounds; 
//      P.vel[a] = abs(P.vel[a]); // 양의 방향으로 바운스
//    }
//  }
//
//  // 위치 업데이트 (랜덤 시드는 유지)
//  P.pos = vec4<f32>(pos, P.pos.w);
//
//  particles.data[i] = P;
//}
//)";
//
//static const char kRenderWGSL[] = R"(
//struct MVP {
//  model : mat4x4<f32>,
//  view  : mat4x4<f32>,
//  proj  : mat4x4<f32>,
//  camPos: vec4<f32>,
//}
//
//struct Particle {
//  pos        : vec4<f32>,
//  vel        : vec4<f32>,
//  lightDir   : vec4<f32>,
//  lightColor : vec4<f32>,
//  atlasShiny : vec4<f32>,
//}
//
//struct Particles { 
//  data : array<Particle>,
//}
//
//@group(0) @binding(0) var<uniform> mvp : MVP;
//@group(0) @binding(1) var samp : sampler;
//@group(0) @binding(2) var atlasTex : texture_2d<f32>;
//@group(0) @binding(3) var<storage, read> particles : Particles;
//
//struct VSOut {
//  @builtin(position) pos : vec4<f32>,
//  @location(0) uv : vec2<f32>,
//  @location(1) normalWS : vec3<f32>,
//  @location(2) worldPos : vec3<f32>,
//  @location(3) @interpolate(flat) atlasIndex : u32,
//  @location(4) @interpolate(flat) lightDir : vec3<f32>,
//  @location(5) @interpolate(flat) lightColor : vec3<f32>,
//  @location(6) @interpolate(flat) shininess : f32,
//}
//
//@vertex
//fn vs(
//  @location(0) inPos : vec3<f32>,
//  @location(1) inNormal : vec3<f32>,
//  @location(2) inUV : vec2<f32>,
//  @builtin(instance_index) iid : u32
//) -> VSOut {
//  var o : VSOut;
//  let P = particles.data[iid];
//
//  let worldPos = inPos + P.pos.xyz;
//  o.pos = mvp.proj * mvp.view * mvp.model * vec4<f32>(worldPos, 1.0);
//
//  o.worldPos = worldPos;
//  o.normalWS = normalize(inNormal);
//  o.uv = inUV;
//
//  o.atlasIndex = u32(P.atlasShiny.y);
//  o.lightDir   = normalize(P.lightDir.xyz);
//  o.lightColor = P.lightColor.rgb;
//  o.shininess  = max(P.atlasShiny.z, 1.0);
//  return o;
//}
//
//@fragment
//fn fs(
//  @location(0) uv : vec2<f32>,
//  @location(1) normalWS : vec3<f32>,
//  @location(2) worldPos : vec3<f32>,
//  @location(3) @interpolate(flat) atlasIndex : u32,
//  @location(4) @interpolate(flat) lightDir : vec3<f32>,
//  @location(5) @interpolate(flat) lightColor : vec3<f32>,
//  @location(6) @interpolate(flat) shininess : f32
//) -> @location(0) vec4<f32> {
//  let cols:u32 = 4u;
//  let rows:u32 = 4u;
//  let scale = vec2<f32>(1.0/f32(cols), 1.0/f32(rows));
//  let col = f32(atlasIndex % cols);
//  let row = f32(atlasIndex / cols);
//  let tileOffset = vec2<f32>(col,row) * scale;
//  let inset = 0.002;
//  let uvAtlas = uv * (scale - 2.0*vec2<f32>(inset*scale.x, inset*scale.y)) + tileOffset + vec2<f32>(inset*scale.x, inset*scale.y);
//  let baseColor = textureSample(atlasTex, samp, uvAtlas).rgb;
//
//  let N = normalize(normalWS);
//  let L = normalize(-lightDir);
//  let V = normalize(mvp.camPos.xyz - worldPos);
//  let H = normalize(L + V);
//
//  let ambient = 0.10;
//  let ndotl = max(dot(N,L), 0.0);
//  let diffuse = ndotl;
//  let specular = pow(max(dot(N,H), 0.0), shininess);
//  let color = baseColor * (ambient + diffuse * lightColor) + specular * lightColor;
//  return vec4<f32>(color, 1.0);
//}
//)";
//
//// ===================== Init =====================
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
//        const char* errorType = "Unknown";
//        switch(t) {
//            case wgpu::ErrorType::Validation: errorType = "Validation"; break;
//            case wgpu::ErrorType::OutOfMemory: errorType = "OutOfMemory"; break;
//            case wgpu::ErrorType::Internal: errorType = "Internal"; break;
//            case wgpu::ErrorType::Unknown: errorType = "Unknown"; break;
//            default: errorType = "Other"; break;
//        }
//        std::cerr << "Device error [" << errorType << "]: " << msg << "\n";
//        });
//    wgpu::Future f2 = adapter.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView m) {
//            if (s != wgpu::RequestDeviceStatus::Success) { std::cerr << "RequestDevice failed: " << m << "\n"; std::exit(1); }
//            device = std::move(d);
//        });
//    instanceW.WaitAny(f2, UINT64_MAX);
//}
//
//// ===================== Particles init =====================
//static inline void norm3(float v[3]) {
//    float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
//    if (l > 1e-6f) { v[0] /= l; v[1] /= l; v[2] /= l; }
//}
//void InitParticlesCPU() {
//    gParticles.resize(PARTICLE_COUNT);
//    std::mt19937 rng(42);
//    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
//    std::uniform_real_distribution<float> U01(0.0f, 1.0f);
//    std::uniform_int_distribution<int> AT(0, 15);
//
//    const float spawn = 100.0f;
//    for (uint32_t i = 0; i < PARTICLE_COUNT; ++i) {
//        auto& P = gParticles[i];
//        
//        // 랜덤 초기 위치
//        P.pos[0] = U(rng) * spawn; 
//        P.pos[1] = U(rng) * spawn; 
//        P.pos[2] = U(rng) * spawn; 
//        P.pos[3] = U01(rng) * 1000.0f; // 랜덤 시드
//        
//        // 랜덤 직선 속도 (각 축별로 독립적인 속도)
//        float speedScale = 20.0f + 40.0f * U01(rng); // 20~60 사이의 랜덤 속도
//        P.vel[0] = U(rng) * speedScale; // X축 속도 (-60 ~ +60)
//        P.vel[1] = U(rng) * speedScale; // Y축 속도 (-60 ~ +60)
//        P.vel[2] = U(rng) * speedScale; // Z축 속도 (-60 ~ +60)
//        P.vel[3] = speedScale; // 속도 스케일 저장 (사용하지 않지만 호환성 유지)
//
//        // 라이트 방향 (고정)
//        float ld[3] = { U(rng), U(rng) * 0.5f - 0.2f, U(rng) }; 
//        norm3(ld);
//        P.lightDir[0] = ld[0]; P.lightDir[1] = ld[1]; P.lightDir[2] = ld[2]; P.lightDir[3] = 0.0f;
//
//        // 라이트 색상
//        float rc = 0.6f + 0.4f * U01(rng), gc = 0.6f + 0.4f * U01(rng), bc = 0.6f + 0.4f * U01(rng);
//        P.lightColor[0] = rc; P.lightColor[1] = gc; P.lightColor[2] = bc; P.lightColor[3] = 0.0f;
//
//        // 아틀라스 및 광택
//        P.atlasShiny[0] = U01(rng) * 1000.0f;
//        P.atlasShiny[1] = (float)AT(rng);
//        P.atlasShiny[2] = 8.0f + 64.0f * U01(rng);
//        P.atlasShiny[3] = 0.0f;
//    }
//}
//
//// ===================== Pipeline setups =====================
//wgpu::BindGroupLayout MakeRenderBGL(uint64_t particleBytes) {
//    wgpu::BindGroupLayoutEntry e0{}; e0.binding = 0; e0.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
//    e0.buffer.type = wgpu::BufferBindingType::Uniform; e0.buffer.minBindingSize = sizeof(MVP);
//    wgpu::BindGroupLayoutEntry e1{}; e1.binding = 1; e1.visibility = wgpu::ShaderStage::Fragment; e1.sampler.type = wgpu::SamplerBindingType::Filtering;
//    wgpu::BindGroupLayoutEntry e2{}; e2.binding = 2; e2.visibility = wgpu::ShaderStage::Fragment; e2.texture.sampleType = wgpu::TextureSampleType::Float; e2.texture.viewDimension = wgpu::TextureViewDimension::e2D;
//    wgpu::BindGroupLayoutEntry e3{}; e3.binding = 3; e3.visibility = wgpu::ShaderStage::Vertex; e3.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage; e3.buffer.minBindingSize = particleBytes;
//    wgpu::BindGroupLayoutEntry entries[4] = { e0,e1,e2,e3 };
//    wgpu::BindGroupLayoutDescriptor d{}; d.entryCount = 4; d.entries = entries;
//    return device.CreateBindGroupLayout(&d);
//}
//wgpu::BindGroupLayout MakeComputeBGL(uint64_t particleBytes) {
//    wgpu::BindGroupLayoutEntry e0{}; e0.binding = 0; e0.visibility = wgpu::ShaderStage::Compute; e0.buffer.type = wgpu::BufferBindingType::Storage; e0.buffer.minBindingSize = particleBytes;
//    wgpu::BindGroupLayoutEntry e1{}; e1.binding = 1; e1.visibility = wgpu::ShaderStage::Compute; e1.buffer.type = wgpu::BufferBindingType::Uniform; e1.buffer.minBindingSize = sizeof(SimUBO);
//    wgpu::BindGroupLayoutEntry entries[2] = { e0,e1 };
//    wgpu::BindGroupLayoutDescriptor d{}; d.entryCount = 2; d.entries = entries;
//    return device.CreateBindGroupLayout(&d);
//}
//
//wgpu::RenderPipeline BuildRenderPipeline(wgpu::ShaderModule shader, wgpu::PipelineLayout pl) {
//    wgpu::VertexAttribute va[3]{}; // pos3, normal3, uv2
//    va[0].format = wgpu::VertexFormat::Float32x3; va[0].offset = 0;               va[0].shaderLocation = 0;
//    va[1].format = wgpu::VertexFormat::Float32x3; va[1].offset = sizeof(float) * 3; va[1].shaderLocation = 1;
//    va[2].format = wgpu::VertexFormat::Float32x2; va[2].offset = sizeof(float) * 6; va[2].shaderLocation = 2;
//    wgpu::VertexBufferLayout vbl0{}; vbl0.arrayStride = sizeof(float) * 8; vbl0.attributeCount = 3; vbl0.attributes = va; vbl0.stepMode = wgpu::VertexStepMode::Vertex;
//
//    wgpu::ColorTargetState color{}; color.format = format; color.writeMask = wgpu::ColorWriteMask::All;
//    wgpu::FragmentState fs{}; fs.module = shader; fs.entryPoint = "fs"; fs.targetCount = 1; fs.targets = &color;
//
//    wgpu::DepthStencilState ds{}; ds.format = wgpu::TextureFormat::Depth24Plus; ds.depthWriteEnabled = true; ds.depthCompare = wgpu::CompareFunction::Less;
//
//    wgpu::RenderPipelineDescriptor rpd{};
//    rpd.label = "Render_1M_Particles_Cubes";
//    rpd.layout = pl;
//    rpd.vertex.module = shader; rpd.vertex.entryPoint = "vs";
//    rpd.vertex.bufferCount = 1; rpd.vertex.buffers = &vbl0;
//    rpd.fragment = &fs;
//    rpd.depthStencil = &ds;
//    rpd.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
//    rpd.primitive.frontFace = wgpu::FrontFace::CCW;
//    rpd.primitive.cullMode = wgpu::CullMode::Back;
//    rpd.multisample.count = 1; rpd.multisample.mask = 0xFFFFFFFF; rpd.multisample.alphaToCoverageEnabled = false;
//    return device.CreateRenderPipeline(&rpd);
//}
//
//wgpu::ComputePipeline BuildComputePipeline(wgpu::ShaderModule shader, wgpu::PipelineLayout pl) {
//    wgpu::ComputePipelineDescriptor cpd{}; cpd.layout = pl; cpd.compute.module = shader; cpd.compute.entryPoint = "cs_main";
//    cpd.label = "ComputePipeline(UpdateParticles)";
//    return device.CreateComputePipeline(&cpd);
//}
//
//// ===== Shader compile log helper =====
//void PrintModuleMessages(const wgpu::ShaderModule& mod, const char* tag) {
//    // Note: In current WebGPU C++ API, GetCompilationInfo might not be available
//    // or might have a different signature. For now, we'll skip this functionality
//    // and rely on device error callbacks for shader compilation errors.
//    std::cout << "[WGSL] Module compiled: " << tag << std::endl;
//    
//    // Check if the module is valid by checking if it's non-null
//    if (!mod) {
//        std::cerr << "[ERROR] Failed to create shader module: " << tag << std::endl;
//    }
//}
//
//// ===================== Graphics Init =====================
//void InitGraphics() {
//    ConfigureSurface();
//    CreateDepthTexture();
//
//    CreateAtlasTextureFromPNG(std::filesystem::path("floatplane\\textures\\particle_atlas.png"));
//
//    vertexBuffer = CreateBuffer(cubeVertices, sizeof(cubeVertices), wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//    indexBuffer = CreateBuffer(cubeIndices, sizeof(cubeIndices), wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst);
//
//    MVP mvp{}; uniformBuffer = CreateBuffer(&mvp, sizeof(MVP), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//
//    InitParticlesCPU();
//    const uint64_t particleBytes = (uint64_t)gParticles.size() * sizeof(ParticleCPU);
//    particleBuffer = CreateBuffer(gParticles.data(), (size_t)particleBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
//
//    SimUBO sim{};
//    sim.dt_time_bounds_damping[0] = 0.016f; // dt
//    sim.dt_time_bounds_damping[1] = 0.0f;   // time (사용하지 않음)
//    sim.dt_time_bounds_damping[2] = 140.0f; // bounds
//    sim.dt_time_bounds_damping[3] = 0.0f;   // damping (사용하지 않음)
//    sim.accel_jitter[0] = 0.0f; sim.accel_jitter[1] = 0.0f; sim.accel_jitter[2] = 0.0f; sim.accel_jitter[3] = 0.0f; // 가속도와 지터 사용하지 않음
//    sim.count = PARTICLE_COUNT;
//    simUniformBuffer = CreateBuffer(&sim, sizeof(SimUBO), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//
//    renderBGL = MakeRenderBGL(particleBytes);
//    computeBGL = MakeComputeBGL(particleBytes);
//
//    { // render pipeline
//        wgpu::ShaderSourceWGSL rsrc{}; rsrc.code = kRenderWGSL;
//        wgpu::ShaderModuleDescriptor rmd{}; rmd.nextInChain = &rsrc; rmd.label = "RenderWGSL";
//        wgpu::ShaderModule rmod = device.CreateShaderModule(&rmd);
//        PrintModuleMessages(rmod, "render");
//
//        wgpu::PipelineLayoutDescriptor rpld{}; wgpu::BindGroupLayout rLayouts[1] = { renderBGL };
//        rpld.bindGroupLayoutCount = 1; rpld.bindGroupLayouts = rLayouts; rpld.label = "PL_Render";
//        wgpu::PipelineLayout rpl = device.CreatePipelineLayout(&rpld);
//        renderPipeline = BuildRenderPipeline(rmod, rpl);
//    }
//    { // compute pipeline
//        wgpu::ShaderSourceWGSL csrc{}; csrc.code = kComputeWGSL;
//        wgpu::ShaderModuleDescriptor cmd{}; cmd.nextInChain = &csrc; cmd.label = "ComputeWGSL";
//        wgpu::ShaderModule cmod = device.CreateShaderModule(&cmd);
//        PrintModuleMessages(cmod, "compute");
//
//        wgpu::PipelineLayoutDescriptor cpld{}; wgpu::BindGroupLayout cLayouts[1] = { computeBGL };
//        cpld.bindGroupLayoutCount = 1; cpld.bindGroupLayouts = cLayouts; cpld.label = "PL_Compute";
//        wgpu::PipelineLayout cpl = device.CreatePipelineLayout(&cpld);
//        computePipeline = BuildComputePipeline(cmod, cpl);
//    }
//
//    // BindGroups
//    {
//        wgpu::BindGroupEntry b0{}; b0.binding = 0; b0.buffer = uniformBuffer; b0.offset = 0; b0.size = sizeof(MVP);
//        wgpu::BindGroupEntry b1{}; b1.binding = 1; b1.sampler = samplerState;
//        wgpu::BindGroupEntry b2{}; b2.binding = 2; b2.textureView = atlasView;
//        wgpu::BindGroupEntry b3{}; b3.binding = 3; b3.buffer = particleBuffer; b3.offset = 0; b3.size = particleBytes;
//        wgpu::BindGroupEntry entries[4] = { b0,b1,b2,b3 };
//        wgpu::BindGroupDescriptor bgd{}; bgd.layout = renderBGL; bgd.entryCount = 4; bgd.entries = entries; bgd.label = "BG_Render(g0)";
//        renderBG = device.CreateBindGroup(&bgd);
//    }
//    {
//        wgpu::BindGroupEntry c0{}; c0.binding = 0; c0.buffer = particleBuffer; c0.offset = 0; c0.size = particleBytes;
//        wgpu::BindGroupEntry c1{}; c1.binding = 1; c1.buffer = simUniformBuffer; c1.offset = 0; c1.size = sizeof(SimUBO);
//        wgpu::BindGroupEntry entries[2] = { c0,c1 };
//        wgpu::BindGroupDescriptor bgd{}; bgd.layout = computeBGL; bgd.entryCount = 2; bgd.entries = entries; bgd.label = "BG_Compute(g0)";
//        computeBG = device.CreateBindGroup(&bgd);
//    }
//}
//
//// ===================== Input & Render Loop =====================
//Camera camera;
//float deltaTime = 0.0f, lastFrame = 0.0f;
//bool keys[1024]{};
//
//void KeyCallback(GLFWwindow*, int key, int, int action, int) { if (key >= 0 && key < 1024) { if (action == GLFW_PRESS) keys[key] = true; else if (action == GLFW_RELEASE) keys[key] = false; } }
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
//    Mat4Perspective(mvp.proj, camera.fov, (float)kWidth / (float)kHeight, 0.1f, 2000.f);
//    mvp.camPos[0] = camera.pos[0]; mvp.camPos[1] = camera.pos[1]; mvp.camPos[2] = camera.pos[2]; mvp.camPos[3] = 0.f;
//    device.GetQueue().WriteBuffer(uniformBuffer, 0, &mvp, sizeof(MVP));
//
//    static float simTime = 0.0f; simTime += deltaTime;
//    SimUBO sim{};
//    sim.dt_time_bounds_damping[0] = (deltaTime > 0 ? deltaTime : 0.016f);
//    sim.dt_time_bounds_damping[1] = 0.0f; // time (사용하지 않음)
//    sim.dt_time_bounds_damping[2] = 140.0f; // bounds
//    sim.dt_time_bounds_damping[3] = 0.0f; // damping (사용하지 않음)
//    sim.accel_jitter[0] = 0.0f; sim.accel_jitter[1] = 0.0f; sim.accel_jitter[2] = 0.0f; sim.accel_jitter[3] = 0.0f; // 가속도와 지터 사용하지 않음
//    sim.count = PARTICLE_COUNT;
//    device.GetQueue().WriteBuffer(simUniformBuffer, 0, &sim, sizeof(SimUBO));
//
//    wgpu::SurfaceTexture st{}; surface.GetCurrentTexture(&st);
//    auto backbuffer = st.texture.CreateView();
//
//    wgpu::RenderPassColorAttachment ca{}; ca.view = backbuffer; ca.loadOp = wgpu::LoadOp::Clear; ca.storeOp = wgpu::StoreOp::Store; ca.clearValue = { 0.06,0.06,0.10,1.0 };
//    wgpu::RenderPassDepthStencilAttachment da{}; da.view = depthView; da.depthLoadOp = wgpu::LoadOp::Clear; da.depthStoreOp = wgpu::StoreOp::Store; da.depthClearValue = 1.0f;
//    wgpu::RenderPassDescriptor rp{}; rp.colorAttachmentCount = 1; rp.colorAttachments = &ca; rp.depthStencilAttachment = &da;
//
//    auto enc = device.CreateCommandEncoder();
//
//    { // Compute
//        auto cpass = enc.BeginComputePass();
//        cpass.SetPipeline(computePipeline);
//        cpass.SetBindGroup(0, computeBG);
//        const uint32_t WG = 256;
//        const uint32_t numGroups = (PARTICLE_COUNT + WG - 1) / WG;
//        cpass.DispatchWorkgroups(numGroups);
//        cpass.End();
//    }
//    { // Render
//        auto pass = enc.BeginRenderPass(&rp);
//        pass.SetPipeline(renderPipeline);
//        pass.SetBindGroup(0, renderBG);
//        pass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize);
//        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize);
//        pass.DrawIndexed(36, PARTICLE_COUNT, 0, 0, 0);
//        pass.End();
//    }
//
//    auto cmd = enc.Finish();
//    device.GetQueue().Submit(1, &cmd);
//}
//
//// ===================== Main =====================
//int main() {
//    Init();
//    if (!glfwInit()) return -1;
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "1M Cube Particles (SSBO + Compute + Atlas)", nullptr, nullptr);
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
