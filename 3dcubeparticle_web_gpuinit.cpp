//// ===================== WebGPU + GLFW: 1,000,000 Cube Particles via SSBO + Compute (GPU init) =====================
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <chrono>
//#include <cstring>
//#include <cstdint>
//#include <string>
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
//// ======= SimUBO: std140 16B 정렬 =======
//struct alignas(16) SimUBO {
//    float dt_time_bounds_damping[4]; // (dt, time, bounds, damping)
//    float accel_jitter[4];           // (accel.x, accel.y, accel.z, jitter)
//    uint32_t count;
//    uint32_t _pad0 = 0;
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
//// ===================== Particles =====================
//#if defined(__EMSCRIPTEN__)
//constexpr uint32_t PARTICLE_COUNT = 1'000'000;
//#else
//constexpr uint32_t PARTICLE_COUNT = 1'000'000;
//#endif
//
//struct ParticleCPU {
//    float pos[4];
//    float vel[4];
//    float lightDir[4];
//    float lightColor[4];
//    float atlasShiny[4];
//};
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
//wgpu::ComputePipeline computePipeline; // update
//wgpu::ComputePipeline initPipeline;    // init
//
//const uint32_t kWidth = 1280, kHeight = 900;
//
//// ===================== Buffer helpers =====================
//wgpu::Buffer CreateBuffer(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd{}; bd.size = size; bd.usage = usage; bd.mappedAtCreation = (data != nullptr);
//    auto buf = device.CreateBuffer(&bd);
//    if (data) { std::memcpy(buf.GetMappedRange(), data, size); buf.Unmap(); }
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
//#if defined(__EMSCRIPTEN__)
//    path = "/assets/particle_atlas.png";
//#else
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
//// ===================== WGSL (init / update / render) =====================
//static const char kInitWGSL[] = R"(
//struct Particle { pos:vec4<f32>, vel:vec4<f32>, lightDir:vec4<f32>, lightColor:vec4<f32>, atlasShiny:vec4<f32>, }
//struct Particles { data: array<Particle>, }
//
//@group(0) @binding(0) var<storage, read_write> particles : Particles;
//
//struct SimUBO {
//  dt_time_bounds_damping: vec4<f32>,
//  accel_jitter: vec4<f32>,
//  count: u32, _pad0: u32, _pad1: u32, _pad2: u32,
//}
//@group(0) @binding(1) var<uniform> sim : SimUBO;
//
//fn hash32(x:u32)->u32{
//  var v = x * 747796405u + 2891336453u;
//  v ^= v >> 16u; v *= 2246822519u; v ^= v >> 13u; v *= 3266489917u; v ^= v >> 16u; return v;
//}
//fn rnd01(seed:u32)->f32{ return f32(hash32(seed))*(1.0/4294967295.0); }
//fn rnd11(seed:u32)->f32{ return rnd01(seed)*2.0 - 1.0; }
//fn normalize3(v:vec3<f32>)->vec3<f32>{ let len=max(length(v),1e-6); return v/len; }
//
//@compute @workgroup_size(256)
//fn cs_init(@builtin(global_invocation_id) gid: vec3<u32>){
//  let i = gid.x; if (i >= sim.count){ return; }
//  let spawn:f32 = 100.0;
//  let speedMin:f32 = 20.0; let speedMax:f32 = 60.0;
//  
//  // Fix: Declare variables one by one with semicolons
//  let s0=i*1664525u+1013904223u;
//  let s1=hash32(s0+1u);
//  let s2=hash32(s0+2u);
//  let s3=hash32(s0+3u);
//  let s4=hash32(s0+4u);
//  let s5=hash32(s0+5u);
//  let s6=hash32(s0+6u);
//  let s7=hash32(s0+7u);
//  let s8=hash32(s0+8u);
//
//  let pos = vec3<f32>( rnd11(s1)*spawn, rnd11(s2)*spawn, rnd11(s3)*spawn );
//  let randSeed = rnd01(s4)*1000.0;
//
//  let speedScale = mix(speedMin, speedMax, rnd01(s5));
//  let vel = vec3<f32>( rnd11(s6)*speedScale, rnd11(s7)*speedScale, rnd11(s8)*speedScale );
//
//  let ld = normalize3(vec3<f32>( rnd11(hash32(s1+77u)),
//                                 rnd11(hash32(s2+77u))*0.5-0.2,
//                                 rnd11(hash32(s3+77u)) ));
//  let col = vec3<f32>( 0.6+0.4*rnd01(hash32(s4+99u)),
//                       0.6+0.4*rnd01(hash32(s5+99u)),
//                       0.6+0.4*rnd01(hash32(s6+99u)) );
//  let atlasIndex = floor(rnd01(hash32(s7+123u))*16.0);
//  let shininess  = 8.0 + 64.0 * rnd01(hash32(s8+123u));
//
//  var P:Particle;
//  P.pos=vec4<f32>(pos, randSeed);
//  P.vel=vec4<f32>(vel, speedScale);
//  P.lightDir=vec4<f32>(ld,0.0);
//  P.lightColor=vec4<f32>(col,0.0);
//  P.atlasShiny=vec4<f32>(rnd01(s0)*1000.0, atlasIndex, shininess, 0.0);
//
//  particles.data[i]=P;
//}
//)";
//
//static const char kComputeWGSL[] = R"(
//struct Particle { pos:vec4<f32>, vel:vec4<f32>, lightDir:vec4<f32>, lightColor:vec4<f32>, atlasShiny:vec4<f32>, }
//struct Particles { data: array<Particle>, }
//
//@group(0) @binding(0) var<storage, read_write> particles : Particles;
//
//struct SimUBO {
//  dt_time_bounds_damping: vec4<f32>,
//  accel_jitter: vec4<f32>,
//  count: u32, _pad0: u32, _pad1: u32, _pad2: u32,
//}
//@group(0) @binding(1) var<uniform> sim : SimUBO;
//
//@compute @workgroup_size(256)
//fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
//  let i = gid.x;
//  if (i >= sim.count) { return; }
//  var P = particles.data[i];
//  let dt = sim.dt_time_bounds_damping.x;
//  let bounds = sim.dt_time_bounds_damping.z;
//  var pos = P.pos.xyz + P.vel.xyz * dt;
//  for (var a:i32=0; a<3; a++){
//    if (pos[a] > bounds) { pos[a] = bounds;  P.vel[a] = -abs(P.vel[a]); }
//    if (pos[a] < -bounds){ pos[a] = -bounds; P.vel[a] = abs(P.vel[a]);  }
//  }
//  P.pos = vec4<f32>(pos, P.pos.w);
//  particles.data[i] = P;
//}
//)";
//
//static const char kRenderWGSL[] = R"(
//struct MVP { model:mat4x4<f32>, view:mat4x4<f32>, proj:mat4x4<f32>, camPos:vec4<f32>, }
//struct Particle { pos:vec4<f32>, vel:vec4<f32>, lightDir:vec4<f32>, lightColor:vec4<f32>, atlasShiny:vec4<f32>, }
//struct Particles { data: array<Particle>, }
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
//fn vs(@location(0) inPos:vec3<f32>, @location(1) inNormal:vec3<f32>, @location(2) inUV:vec2<f32>, @builtin(instance_index) iid:u32) -> VSOut {
//  var o:VSOut;
//  let P = particles.data[iid];
//  let worldPos = inPos + P.pos.xyz;
//  o.pos = mvp.proj * mvp.view * mvp.model * vec4<f32>(worldPos, 1.0);
//  o.worldPos = worldPos;
//  o.normalWS = normalize(inNormal);
//  o.uv = inUV;
//  o.atlasIndex = u32(P.atlasShiny.y);
//  o.lightDir   = normalize(P.lightDir.xyz);
//  o.lightColor = P.lightColor.rgb;
//  o.shininess  = max(P.atlasShiny.z, 1.0);
//  return o;
//}
//
//@fragment
//fn fs(@location(0) uv:vec2<f32>, @location(1) normalWS:vec3<f32>, @location(2) worldPos:vec3<f32>,
//      @location(3) @interpolate(flat) atlasIndex:u32,
//      @location(4) @interpolate(flat) lightDir:vec3<f32>,
//      @location(5) @interpolate(flat) lightColor:vec3<f32>,
//      @location(6) @interpolate(flat) shininess:f32) -> @location(0) vec4<f32> {
//  let cols:u32 = 4u; let rows:u32 = 4u;
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
//// ===================== Init Device =====================
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
//        switch (t) {
//        case wgpu::ErrorType::Validation: errorType = "Validation"; break;
//        case wgpu::ErrorType::OutOfMemory: errorType = "OutOfMemory"; break;
//        case wgpu::ErrorType::Internal: errorType = "Internal"; break;
//        case wgpu::ErrorType::Unknown: errorType = "Unknown"; break;
//        default: errorType = "Other"; break;
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
//// ===================== BGL / Pipelines =====================
//wgpu::BindGroupLayout MakeRenderBGL(uint64_t particleBytes) {
//    wgpu::BindGroupLayoutEntry e0{}; e0.binding = 0; e0.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment; e0.buffer.type = wgpu::BufferBindingType::Uniform; e0.buffer.minBindingSize = sizeof(MVP);
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
//    wgpu::VertexAttribute va[3]{};
//    va[0].format = wgpu::VertexFormat::Float32x3; va[0].offset = 0; va[0].shaderLocation = 0;
//    va[1].format = wgpu::VertexFormat::Float32x3; va[1].offset = sizeof(float) * 3; va[1].shaderLocation = 1;
//    va[2].format = wgpu::VertexFormat::Float32x2; va[2].offset = sizeof(float) * 6; va[2].shaderLocation = 2;
//
//    wgpu::VertexBufferLayout vbl0{}; vbl0.arrayStride = sizeof(float) * 8; vbl0.attributeCount = 3; vbl0.attributes = va; vbl0.stepMode = wgpu::VertexStepMode::Vertex;
//
//    wgpu::ColorTargetState color{}; color.format = format; color.writeMask = wgpu::ColorWriteMask::All;
//    wgpu::FragmentState fs{}; fs.module = shader; fs.entryPoint = "fs"; fs.targetCount = 1; fs.targets = &color;
//
//    wgpu::DepthStencilState ds{}; ds.format = wgpu::TextureFormat::Depth24Plus; ds.depthWriteEnabled = true; ds.depthCompare = wgpu::CompareFunction::Less;
//
//    wgpu::RenderPipelineDescriptor rpd{};
//    rpd.label = "Render_1M_Particles_Cubes"; rpd.layout = pl;
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
//wgpu::ComputePipeline BuildComputePipeline(wgpu::ShaderModule shader, wgpu::PipelineLayout pl, const char* entry) {
//    wgpu::ComputePipelineDescriptor cpd{}; cpd.layout = pl; cpd.compute.module = shader; cpd.compute.entryPoint = entry;
//    return device.CreateComputePipeline(&cpd);
//}
//
//// ===================== Graphics Init =====================
//void InitGraphics() {
//    ConfigureSurface();
//    CreateDepthTexture();
//    CreateAtlasTextureFromPNG(std::filesystem::path("floatplane\\textures\\particle_atlas.png"));
//
//    vertexBuffer = CreateBuffer(cubeVertices, sizeof(cubeVertices), wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//    indexBuffer = CreateBuffer(cubeIndices, sizeof(cubeIndices), wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst);
//
//    MVP mvp{}; uniformBuffer = CreateBuffer(&mvp, sizeof(MVP), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//
//    // 파티클 SSBO: 빈 버퍼
//    const uint64_t particleBytes = uint64_t(PARTICLE_COUNT) * sizeof(ParticleCPU);
//    particleBuffer = CreateBuffer(nullptr, (size_t)particleBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
//
//    // SimUBO (count 필수)
//    SimUBO sim{};
//    sim.dt_time_bounds_damping[0] = 0.016f; // dt
//    sim.dt_time_bounds_damping[1] = 0.0f;   // time
//    sim.dt_time_bounds_damping[2] = 140.0f; // bounds
//    sim.dt_time_bounds_damping[3] = 0.0f;   // damping
//    sim.accel_jitter[0] = 0.0f; sim.accel_jitter[1] = 0.0f; sim.accel_jitter[2] = 0.0f; sim.accel_jitter[3] = 0.0f;
//    sim.count = PARTICLE_COUNT;
//    simUniformBuffer = CreateBuffer(&sim, sizeof(SimUBO), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//
//    renderBGL = MakeRenderBGL(particleBytes);
//    computeBGL = MakeComputeBGL(particleBytes);
//
//    // ===== 초기화용 compute 파이프라인 =====
//    {
//        wgpu::ShaderSourceWGSL src{}; src.code = kInitWGSL;
//        wgpu::ShaderModuleDescriptor md{}; md.nextInChain = &src; md.label = "ComputeInitWGSL";
//        auto mod = device.CreateShaderModule(&md);
//
//        wgpu::PipelineLayoutDescriptor pld{}; wgpu::BindGroupLayout layouts[1] = { computeBGL };
//        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = layouts; pld.label = "PL_Compute_Init";
//        auto pl = device.CreatePipelineLayout(&pld);
//
//        initPipeline = BuildComputePipeline(mod, pl, "cs_init");
//    }
//
//    // ===== 업데이트용 compute 파이프라인 =====
//    {
//        wgpu::ShaderSourceWGSL src{}; src.code = kComputeWGSL;
//        wgpu::ShaderModuleDescriptor md{}; md.nextInChain = &src; md.label = "ComputeUpdateWGSL";
//        auto mod = device.CreateShaderModule(&md);
//
//        wgpu::PipelineLayoutDescriptor pld{}; wgpu::BindGroupLayout layouts[1] = { computeBGL };
//        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = layouts; pld.label = "PL_Compute_Update";
//        auto pl = device.CreatePipelineLayout(&pld);
//
//        computePipeline = BuildComputePipeline(mod, pl, "cs_main");
//    }
//
//    // ===== 렌더 파이프라인 =====
//    {
//        wgpu::ShaderSourceWGSL src{}; src.code = kRenderWGSL;
//        wgpu::ShaderModuleDescriptor md{}; md.nextInChain = &src; md.label = "RenderWGSL";
//        auto mod = device.CreateShaderModule(&md);
//
//        wgpu::PipelineLayoutDescriptor pld{}; wgpu::BindGroupLayout layouts[1] = { renderBGL };
//        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = layouts; pld.label = "PL_Render";
//        auto pl = device.CreatePipelineLayout(&pld);
//
//        renderPipeline = BuildRenderPipeline(mod, pl);
//    }
//
//    // ===== BindGroups =====
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
//
//    // ===== ★ GPU 초기화 패스 한 번 실행 ★ =====
//    {
//        auto enc = device.CreateCommandEncoder();
//        auto cpass = enc.BeginComputePass();
//        cpass.SetPipeline(initPipeline);
//        cpass.SetBindGroup(0, computeBG);
//        const uint32_t WG = 256;
//        const uint32_t numGroups = (PARTICLE_COUNT + WG - 1) / WG;
//        cpass.DispatchWorkgroups(numGroups);
//        cpass.End();
//        auto cmd = enc.Finish();
//        device.GetQueue().Submit(1, &cmd);
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
//    // sim UBO 갱신 (dt/bounds 등 필요시)
//    SimUBO sim{};
//    sim.dt_time_bounds_damping[0] = (deltaTime > 0 ? deltaTime : 0.016f);
//    sim.dt_time_bounds_damping[1] = 0.0f;
//    sim.dt_time_bounds_damping[2] = 140.0f;
//    sim.dt_time_bounds_damping[3] = 0.0f;
//    sim.accel_jitter[0] = 0.0f; sim.accel_jitter[1] = 0.0f; sim.accel_jitter[2] = 0.0f; sim.accel_jitter[3] = 0.0f;
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
//    { // Compute Update
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
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "Cube Particles (SSBO + Compute + GPU init)", nullptr, nullptr);
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
