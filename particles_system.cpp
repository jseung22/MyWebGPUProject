//// ===================== WebGPU + GLFW: 1,000,000 Cube Particles (SSBO + Compute) =====================
//// - Compute shader: Sine Wave jitter + bounce inside AABB
//// - Render: instanced cubes; vertex pulls per-instance position from SSBO
//// - Camera: WASD move, mouse look, wheel zoom
//// - Build: same as your previous Dawn/WebGPU + GLFW setup
//
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <chrono>
//#include <cstring>
//#include <random>
//#include <GLFW/glfw3.h>
//#include <dawn/webgpu_cpp_print.h>
//#include <webgpu/webgpu_cpp.h>
//#include <webgpu/webgpu_glfw.h>
//
//static constexpr uint32_t kWidth = 1280;
//static constexpr uint32_t kHeight = 800;
//
//// ===================== Camera =====================
//struct Camera {
//    float pos[3] = { 0.0f, 0.0f, 200.0f };
//    float front[3] = { 0.0f, 0.0f,-1.0f };
//    float up[3] = { 0.0f, 1.0f, 0.0f };
//    float yaw = -90.0f, pitch = 0.0f, fov = 45.0f;
//    float lastX = kWidth / 2.0f, lastY = kHeight / 2.0f;
//    bool firstMouse = true;
//    static void Normalize(float* v) { float L = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (L > 1e-6f) { v[0] /= L; v[1] /= L; v[2] /= L; } }
//    static void Cross(const float* a, const float* b, float* o) { o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0]; }
//    void UpdateFront() {
//        float ry = yaw * 3.14159265f / 180.f, rp = pitch * 3.14159265f / 180.f;
//        front[0] = std::cos(ry) * std::cos(rp); front[1] = std::sin(rp); front[2] = std::sin(ry) * std::cos(rp); Normalize(front);
//    }
//    void Move(const float* d, float s) { pos[0] += d[0] * s; pos[1] += d[1] * s; pos[2] += d[2] * s; }
//    void ProcessKeyboard(int key, float dt) {
//        float sp = 80.f * dt, right[3]; Cross(front, up, right); Normalize(right);
//        if (key == GLFW_KEY_W) Move(front, sp);
//        if (key == GLFW_KEY_S) Move(front, -sp);
//        if (key == GLFW_KEY_A) Move(right, -sp);
//        if (key == GLFW_KEY_D) Move(right, sp);
//    }
//    void ProcessMouse(float x, float y) {
//        if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }
//        float xo = (x - lastX) * 0.1f, yo = (lastY - y) * 0.1f; lastX = x; lastY = y;
//        yaw += xo; pitch += yo; if (pitch > 89) pitch = 89; if (pitch < -89) pitch = -89; UpdateFront();
//    }
//    void ProcessScroll(float yo) { fov -= yo; if (fov < 1) fov = 1; if (fov > 60) fov = 60; }
//};
//
//// ===================== Math =====================
//void Mat4Identity(float* m) { std::memset(m, 0, 16 * sizeof(float)); m[0] = m[5] = m[10] = m[15] = 1; }
//void Mat4Perspective(float* m, float fov, float aspect, float n, float f) {
//    float t = std::tan(fov * 0.5f * 3.14159265f / 180.f);
//    std::memset(m, 0, 16 * sizeof(float));
//    m[0] = 1.f / (aspect * t); m[5] = 1.f / t; m[10] = -(f + n) / (f - n); m[11] = -1.f; m[14] = -(2.f * f * n) / (f - n);
//}
//void Mat4LookAt(float* m, const float* e, const float* c, const float* up) {
//    float fwd[3] = { c[0] - e[0],c[1] - e[1],c[2] - e[2] }; Camera::Normalize(fwd);
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
//// ===================== GPU Globals =====================
//wgpu::Instance instance;
//wgpu::Adapter  adapter;
//wgpu::Device   device;
//wgpu::Surface  surface;
//wgpu::TextureFormat colorFormat;
//wgpu::Texture  depthTex;
//wgpu::TextureView depthView;
//
//wgpu::Buffer vertexBuffer, indexBuffer;
//wgpu::Buffer particleBuffer;    // SSBO (read/write in compute, read in vertex)
//wgpu::Buffer simUniform;        // compute params
//wgpu::Buffer mvpUniform;        // render MVP
//wgpu::BindGroup simBG, renderBG;
//wgpu::BindGroupLayout simBGL, renderBGL;
//wgpu::RenderPipeline renderPipeline;
//wgpu::ComputePipeline computePipeline;
//
//static constexpr uint32_t WORKGROUP_SIZE = 256u;
//static constexpr uint32_t NUM_PARTICLES = 1'000'000u; // 1e6
//
//// ============ Cube mesh (6 faces * 2 tris * 3 idx) ============
//const float cubeVerts[] = {
//    // pos (xyz)
//    -0.5f,-0.5f,-0.5f,  0.5f,-0.5f,-0.5f,  0.5f, 0.5f,-0.5f, -0.5f, 0.5f,-0.5f,
//    -0.5f,-0.5f, 0.5f,  0.5f,-0.5f, 0.5f,  0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f
//};
//const uint32_t cubeIdx[] = {
//    0,1,2, 2,3,0,  4,5,6, 6,7,4,
//    0,4,7, 7,3,0,  1,5,6, 6,2,1,
//    3,2,6, 6,7,3,  0,1,5, 5,4,0
//};
//static constexpr uint32_t kCubeIndexCount = 36;
//
//// ===================== WGSL Shaders =====================
//// -- Compute: simple linear motion with bounce
//static const char kComputeWGSL[] = R"(
//struct Particle {
//  pos   : vec4<f32>,   // xyz = position, w = size
//  vel   : vec4<f32>,   // xyz = velocity
//  color : vec4<f32>,   // rgb = color
//  misc  : vec4<f32>,   // x=age
//};
//
//struct Sim {
//  dt_time  : vec2<f32>, // x=dt, y=time
//  _pad0    : vec2<f32>,
//  bounds   : vec4<f32>, // xyz = half-extent (AABB), w unused
//  _pad1    : vec4<f32>,
//};
//
//@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
//@group(0) @binding(1) var<uniform> sim : Sim;
//
//@compute @workgroup_size(256)
//fn csMain(@builtin(global_invocation_id) gid : vec3<u32>) {
//  let i = gid.x;
//  if (i >= arrayLength(&particles)) { return; }
//
//  var p = particles[i];
//  
//  let dt = sim.dt_time.x;
//  let B = sim.bounds.xyz;  // half-extent
//
//  // Linear motion: new_position = position + velocity * dt
//  var pos = p.pos.xyz + p.vel.xyz * dt;
//  var vel = p.vel.xyz;
//
//  // Bounce against AABB [-B, +B]
//  // X axis
//  if (pos.x > B.x) {
//    pos.x = B.x;
//    vel.x = -vel.x;
//  }
//  if (pos.x < -B.x) {
//    pos.x = -B.x;
//    vel.x = -vel.x;
//  }
//  
//  // Y axis
//  if (pos.y > B.y) {
//    pos.y = B.y;
//    vel.y = -vel.y;
//  }
//  if (pos.y < -B.y) {
//    pos.y = -B.y;
//    vel.y = -vel.y;
//  }
//  
//  // Z axis
//  if (pos.z > B.z) {
//    pos.z = B.z;
//    vel.z = -vel.z;
//  }
//  if (pos.z < -B.z) {
//    pos.z = -B.z;
//    vel.z = -vel.z;
//  }
//
//  // Update age
//  p.misc.x += dt;
//
//  // Write back to GPU memory
//  p.pos = vec4<f32>(pos, p.pos.w);
//  p.vel = vec4<f32>(vel, 0.0);
//  particles[i] = p;
//}
//)";
//
//// -- Render: read particle SSBO by instance index, compose small cube
//static const char kRenderWGSL[] = R"(
//struct Particle {
//  pos   : vec4<f32>,   // xyz = position, w = size
//  vel   : vec4<f32>,
//  color : vec4<f32>,   // rgb = color
//  misc  : vec4<f32>,
//};
//
//struct MVP {
//  model : mat4x4<f32>,
//  view  : mat4x4<f32>,
//  proj  : mat4x4<f32>,
//};
//
//@group(0) @binding(0) var<storage, read> particles : array<Particle>;
//@group(0) @binding(1) var<uniform> mvp : MVP;
//
//struct VSOut {
//  @builtin(position) pos : vec4<f32>,
//  @location(0) color : vec3<f32>
//};
//
//@vertex
//fn vsMain(@location(0) inPos : vec3<f32>, @builtin(instance_index) inst : u32) -> VSOut {
//  let p = particles[inst];
//  let size = max(0.02, p.pos.w);
//  let world = vec4<f32>(inPos * size + p.pos.xyz, 1.0);
//  var out : VSOut;
//  out.pos = mvp.proj * mvp.view * mvp.model * world;
//  out.color = p.color.rgb;
//  return out;
//}
//
//@fragment
//fn fsMain(@location(0) color : vec3<f32>) -> @location(0) vec4<f32> {
//  return vec4<f32>(color, 1.0);
//}
//)";
//
//// ===================== Helpers =====================
//wgpu::Buffer CreateBuffer(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd;
//    bd.size = size;
//    bd.usage = usage;
//    bd.mappedAtCreation = true;
//    auto buf = device.CreateBuffer(&bd);
//    if (data) std::memcpy(buf.GetMappedRange(), data, size);
//    buf.Unmap();
//    return buf;
//}
//wgpu::Buffer CreateZeroedBuffer(size_t size, wgpu::BufferUsage usage) {
//    std::vector<uint8_t> zeros(size, 0);
//    return CreateBuffer(zeros.data(), size, usage);
//}
//
//void CreateDepth() {
//    wgpu::TextureDescriptor td{};
//    td.size = { kWidth, kHeight, 1 };
//    td.format = wgpu::TextureFormat::Depth24Plus;
//    td.usage = wgpu::TextureUsage::RenderAttachment;
//    depthTex = device.CreateTexture(&td);
//    depthView = depthTex.CreateView();
//}
//
//void ConfigureSurface() {
//    wgpu::SurfaceCapabilities caps{};
//    surface.GetCapabilities(adapter, &caps);
//    colorFormat = caps.formats[0];
//    wgpu::SurfaceConfiguration cfg{};
//    cfg.device = device;
//    cfg.format = colorFormat;
//    cfg.width = kWidth;
//    cfg.height = kHeight;
//    cfg.presentMode = wgpu::PresentMode::Fifo;
//    surface.Configure(&cfg);
//}
//
//// ===================== Particle CPU Init =====================
//struct CPU_Particle {
//    float pos[4];   // xyz + size
//    float vel[4];   // xyz + pad
//    float color[4]; // rgb + pad
//    float misc[4];  // age + pad
//};
//
//void InitParticles(std::vector<CPU_Particle>& out, float halfX, float halfY, float halfZ) {
//    out.resize(NUM_PARTICLES);
//    std::mt19937 rng(42);
//    std::uniform_real_distribution<float> U(-1.f, 1.f);
//    std::uniform_real_distribution<float> U01(0.f, 1.f);
//
//    for (uint32_t i = 0; i < NUM_PARTICLES; ++i) {
//        auto& p = out[i];
//        
//        // Random starting position within bounds (80% of full size)
//        p.pos[0] = U(rng) * halfX * 0.8f;
//        p.pos[1] = U(rng) * halfY * 0.8f;
//        p.pos[2] = U(rng) * halfZ * 0.8f;
//        p.pos[3] = 0.05f; // cube size
//
//        // Random velocity direction and speed
//        float theta = U01(rng) * 6.28318f;  // angle in XY plane
//        float phi = U01(rng) * 3.14159f;    // angle from Z axis
//        float speed = 5.0f + U01(rng) * 15.0f; // speed 5~20 units/sec
//        
//        p.vel[0] = speed * std::sin(phi) * std::cos(theta);
//        p.vel[1] = speed * std::sin(phi) * std::sin(theta);
//        p.vel[2] = speed * std::cos(phi);
//        p.vel[3] = 0.f;
//
//        // Random color
//        p.color[0] = 0.2f + U01(rng) * 0.8f;
//        p.color[1] = 0.2f + U01(rng) * 0.8f;
//        p.color[2] = 0.2f + U01(rng) * 0.8f;
//        p.color[3] = 1.0f;
//
//        // Age starts at 0
//        p.misc[0] = 0.0f;
//        p.misc[1] = p.misc[2] = p.misc[3] = 0.f;
//    }
//}
//
//// ===================== Uniform structs =====================
//struct MVP { float model[16], view[16], proj[16]; };
//struct SimData {
//    float dt, time, _p0, _p1;
//    float bounds[4];    // xyz half-extent
//    float _pad1[4];
//};
//
//// ===================== Input / Timing =====================
//Camera cam;
//bool keys[1024]{};
//float deltaTime = 0.f, lastFrame = 0.f, globalTime = 0.f;
//
//void KeyCB(GLFWwindow*, int key, int, int action, int) {
//    if (key >= 0 && key < 1024) { if (action == GLFW_PRESS) keys[key] = true; else if (action == GLFW_RELEASE) keys[key] = false; }
//}
//void MouseCB(GLFWwindow*, double x, double y) { cam.ProcessMouse((float)x, (float)y); }
//void ScrollCB(GLFWwindow*, double, double y) { cam.ProcessScroll((float)y); }
//void MouseBtnCB(GLFWwindow* w, int btn, int action, int) {
//    if (btn == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
//    if (btn == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//}
//
//// ===================== Init WebGPU =====================
//void InitWebGPU() {
//    const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
//    wgpu::InstanceDescriptor id{}; id.requiredFeatureCount = 1; id.requiredFeatures = &kTimedWaitAny;
//    instance = wgpu::CreateInstance(&id);
//
//    auto f1 = instance.RequestAdapter(nullptr, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView msg) {
//            if (s != wgpu::RequestAdapterStatus::Success) { std::cerr << "RequestAdapter failed: " << msg << "\n"; std::exit(1); }
//            adapter = std::move(a);
//        });
//    instance.WaitAny(f1, UINT64_MAX);
//
//    wgpu::DeviceDescriptor dd{};
//    dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView m) {
//        std::cerr << "Device error(" << (int)t << "): " << m << "\n";
//        });
//    auto f2 = adapter.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView msg) {
//            if (s != wgpu::RequestDeviceStatus::Success) { std::cerr << "RequestDevice failed: " << msg << "\n"; std::exit(1); }
//            device = std::move(d);
//        });
//    instance.WaitAny(f2, UINT64_MAX);
//}
//
//// ===================== Pipelines & Resources =====================
//void CreatePipelinesAndResources() {
//    ConfigureSurface();
//    CreateDepth();
//
//    // cube buffers
//    vertexBuffer = CreateBuffer(cubeVerts, sizeof(cubeVerts), wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//    indexBuffer = CreateBuffer(cubeIdx, sizeof(cubeIdx), wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst);
//
//    // particles
//    std::vector<CPU_Particle> init;
//    const float halfX = 120.f, halfY = 80.f, halfZ = 120.f; // AABB half-extent
//    InitParticles(init, halfX, halfY, halfZ);
//    particleBuffer = CreateBuffer(init.data(), init.size() * sizeof(CPU_Particle),
//        wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//
//    // uniforms
//    simUniform = CreateZeroedBuffer(sizeof(SimData), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//    mvpUniform = CreateZeroedBuffer(sizeof(MVP), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//
//    // ----- Compute pipeline -----
//    // BGL: storage + uniform
//    wgpu::BindGroupLayoutEntry cEntries[2]{};
//    cEntries[0].binding = 0; cEntries[0].visibility = wgpu::ShaderStage::Compute; cEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
//    cEntries[1].binding = 1; cEntries[1].visibility = wgpu::ShaderStage::Compute; cEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
//    wgpu::BindGroupLayoutDescriptor cBGLd{ .entryCount = 2,.entries = cEntries };
//    simBGL = device.CreateBindGroupLayout(&cBGLd);
//    wgpu::BindGroupEntry cBGEs[2]{};
//    cBGEs[0].binding = 0; cBGEs[0].buffer = particleBuffer; cBGEs[0].offset = 0; cBGEs[0].size = wgpu::kWholeSize;
//    cBGEs[1].binding = 1; cBGEs[1].buffer = simUniform;     cBGEs[1].offset = 0; cBGEs[1].size = sizeof(SimData);
//    wgpu::BindGroupDescriptor cBGd{ .layout = simBGL,.entryCount = 2,.entries = cBGEs };
//    simBG = device.CreateBindGroup(&cBGd);
//
//    wgpu::ShaderSourceWGSL cWGSL;
//    cWGSL.code = kComputeWGSL;
//    wgpu::ShaderModuleDescriptor cSMD{}; 
//    cSMD.nextInChain = &cWGSL;
//    auto cSM = device.CreateShaderModule(&cSMD);
//
//    wgpu::PipelineLayoutDescriptor cPLd{};
//    cPLd.bindGroupLayoutCount = 1; cPLd.bindGroupLayouts = &simBGL;
//    auto cPL = device.CreatePipelineLayout(&cPLd);
//
//    wgpu::ComputePipelineDescriptor cPD{};
//    cPD.layout = cPL; cPD.compute.module = cSM; cPD.compute.entryPoint = "csMain";
//    computePipeline = device.CreateComputePipeline(&cPD);
//
//    // ----- Render pipeline -----
//    // BGL: storage(read) + uniform(MVP)
//    wgpu::BindGroupLayoutEntry rEntries[2]{};
//    rEntries[0].binding = 0; rEntries[0].visibility = wgpu::ShaderStage::Vertex; rEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
//    rEntries[1].binding = 1; rEntries[1].visibility = wgpu::ShaderStage::Vertex; rEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
//    wgpu::BindGroupLayoutDescriptor rBGLd{ .entryCount = 2,.entries = rEntries };
//    renderBGL = device.CreateBindGroupLayout(&rBGLd);
//    wgpu::BindGroupEntry rBGEs[2]{};
//    rBGEs[0].binding = 0; rBGEs[0].buffer = particleBuffer; rBGEs[0].offset = 0; rBGEs[0].size = wgpu::kWholeSize;
//    rBGEs[1].binding = 1; rBGEs[1].buffer = mvpUniform;     rBGEs[1].offset = 0; rBGEs[1].size = sizeof(MVP);
//    wgpu::BindGroupDescriptor rBGd{ .layout = renderBGL,.entryCount = 2,.entries = rBGEs };
//    renderBG = device.CreateBindGroup(&rBGd);
//
//    wgpu::ShaderSourceWGSL rWGSL;
//    rWGSL.code = kRenderWGSL;
//    wgpu::ShaderModuleDescriptor rSMD{}; 
//    rSMD.nextInChain = &rWGSL;
//    auto rSM = device.CreateShaderModule(&rSMD);
//
//    wgpu::VertexAttribute vAttr{};
//    vAttr.shaderLocation = 0; vAttr.format = wgpu::VertexFormat::Float32x3; vAttr.offset = 0;
//    wgpu::VertexBufferLayout vbl{};
//    vbl.arrayStride = sizeof(float) * 3; vbl.attributeCount = 1; vbl.attributes = &vAttr; vbl.stepMode = wgpu::VertexStepMode::Vertex;
//
//    wgpu::ColorTargetState cts{}; cts.format = colorFormat;
//
//    wgpu::FragmentState fs{};
//    fs.module = rSM; fs.entryPoint = "fsMain"; fs.targetCount = 1; fs.targets = &cts;
//
//    wgpu::DepthStencilState ds{};
//    ds.format = wgpu::TextureFormat::Depth24Plus; ds.depthWriteEnabled = true; ds.depthCompare = wgpu::CompareFunction::Less;
//
//    wgpu::PipelineLayoutDescriptor rPLd{};
//    wgpu::BindGroupLayout rLayouts[1] = { renderBGL };
//    rPLd.bindGroupLayoutCount = 1; rPLd.bindGroupLayouts = rLayouts;
//    auto rPL = device.CreatePipelineLayout(&rPLd);
//
//    wgpu::RenderPipelineDescriptor rp{};
//    rp.layout = rPL;
//    rp.vertex.module = rSM; rp.vertex.entryPoint = "vsMain"; rp.vertex.bufferCount = 1; rp.vertex.buffers = &vbl;
//    rp.fragment = &fs;
//    rp.depthStencil = &ds;
//
//    renderPipeline = device.CreateRenderPipeline(&rp);
//}
//
//// ===================== Frame Update =====================
//void Frame() {
//    float now = (float)glfwGetTime();
//    deltaTime = now - lastFrame; lastFrame = now; globalTime += deltaTime;
//
//    for (int i = 0; i < 1024; ++i) if (keys[i]) cam.ProcessKeyboard(i, deltaTime);
//
//    // --- write uniforms ---
//    // Sim
//    SimData sd{};
//    sd.dt = std::min(deltaTime, 1.f / 30.f); // clamp for stability
//    sd.time = globalTime;
//    sd.bounds[0] = 120.f; sd.bounds[1] = 80.f; sd.bounds[2] = 120.f; sd.bounds[3] = 0.f;
//    device.GetQueue().WriteBuffer(simUniform, 0, &sd, sizeof(sd));
//
//    // MVP
//    MVP mvp{};
//    Mat4Identity(mvp.model);
//    float center[3] = { cam.pos[0] + cam.front[0], cam.pos[1] + cam.front[1], cam.pos[2] + cam.front[2] };
//    Mat4LookAt(mvp.view, cam.pos, center, cam.up);
//    Mat4Perspective(mvp.proj, cam.fov, (float)kWidth / (float)kHeight, 0.1f, 2000.f);
//    device.GetQueue().WriteBuffer(mvpUniform, 0, &mvp, sizeof(mvp));
//
//    // --- acquire surface ---
//    wgpu::SurfaceTexture st{}; surface.GetCurrentTexture(&st);
//    auto backView = st.texture.CreateView();
//
//    // --- encode commands: compute then render ---
//    auto encoder = device.CreateCommandEncoder();
//
//    // Compute: dispatch ceil(N/WORKGROUP_SIZE)
//    {
//        auto cpass = encoder.BeginComputePass();
//        cpass.SetPipeline(computePipeline);
//        cpass.SetBindGroup(0, simBG);
//        uint32_t groups = (NUM_PARTICLES + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
//        cpass.DispatchWorkgroups(groups);
//        cpass.End();
//    }
//
//    // Render
//    wgpu::RenderPassColorAttachment ca{};
//    ca.view = backView; ca.loadOp = wgpu::LoadOp::Clear; ca.storeOp = wgpu::StoreOp::Store;
//    ca.clearValue = { 0.06,0.07,0.10,1.0 };
//
//    wgpu::RenderPassDepthStencilAttachment da{};
//    da.view = depthView; da.depthLoadOp = wgpu::LoadOp::Clear; da.depthClearValue = 1.0f; da.depthStoreOp = wgpu::StoreOp::Store;
//
//    wgpu::RenderPassDescriptor rp{};
//    rp.colorAttachmentCount = 1; rp.colorAttachments = &ca; rp.depthStencilAttachment = &da;
//
//    {
//        auto rpass = encoder.BeginRenderPass(&rp);
//        rpass.SetPipeline(renderPipeline);
//        rpass.SetBindGroup(0, renderBG);
//        rpass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize);
//        rpass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize);
//        rpass.DrawIndexed(kCubeIndexCount, NUM_PARTICLES, 0, 0, 0);
//        rpass.End();
//    }
//
//    auto cmd = encoder.Finish();
//    device.GetQueue().Submit(1, &cmd);
//    surface.Present();
//}
//
//// ===================== Main =====================
//int main() {
//    InitWebGPU();
//
//    if (!glfwInit()) return -1;
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "1M Particles - Linear Motion", nullptr, nullptr);
//
//    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
//    CreatePipelinesAndResources();
//
//    glfwSetKeyCallback(window, KeyCB);
//    glfwSetCursorPosCallback(window, MouseCB);
//    glfwSetScrollCallback(window, ScrollCB);
//    glfwSetMouseButtonCallback(window, MouseBtnCB);
//    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//
//    while (!glfwWindowShouldClose(window)) {
//        glfwPollEvents();
//        Frame();
//        instance.ProcessEvents();
//    }
//
//    glfwDestroyWindow(window);
//    glfwTerminate();
//    return 0;
//}