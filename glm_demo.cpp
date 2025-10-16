//// ===================== WebGPU + GLFW + GLM + Vertex구조체로 3D 큐브 =====================
//// 좌클릭: 카메라 컨트롤(커서 숨김), 우클릭: 커서 표시
//
//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
//#include <glm/glm.hpp>
//#include <glm/gtc/type_ptr.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/ext/matrix_clip_space.hpp>  // perspectiveRH_ZO, lookAtRH
//#include <algorithm>
//
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <chrono>
//#include <cstring>
//#include <cstddef>  // offsetof
//#include <GLFW/glfw3.h>
//#if defined(__EMSCRIPTEN__)
//#include <emscripten/emscripten.h>
//#endif
//#include <dawn/webgpu_cpp_print.h>
//#include <webgpu/webgpu_cpp.h>
//#include <webgpu/webgpu_glfw.h>
//
//// ===================== 카메라 =====================
//struct Camera {
//    glm::vec3 pos = { 0.0f, 0.0f, 3.0f };
//    glm::vec3 front = { 0.0f, 0.0f,-1.0f };
//    glm::vec3 up = { 0.0f, 1.0f, 0.0f };
//    float yaw = -90.0f, pitch = 0.0f;
//    float fov = 45.0f;
//    float lastX = 500.0f, lastY = 500.0f;
//    bool firstMouse = true;
//
//    void ProcessKeyboard(int key, float dt) {
//        float speed = 2.5f * dt;
//        glm::vec3 right = glm::normalize(glm::cross(front, up));
//        if (key == GLFW_KEY_W) pos += front * speed;
//        if (key == GLFW_KEY_S) pos -= front * speed;
//        if (key == GLFW_KEY_A) pos -= right * speed;
//        if (key == GLFW_KEY_D) pos += right * speed;
//    }
//    void ProcessMouse(float xpos, float ypos) {
//        if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
//        float xoffset = (xpos - lastX) * 0.1f;
//        float yoffset = (lastY - ypos) * 0.1f;
//        lastX = xpos; lastY = ypos;
//        yaw += xoffset; pitch += yoffset;
//        pitch = std::clamp(pitch, -89.0f, 89.0f);
//        UpdateFront();
//    }
//    void ProcessScroll(float yoffset) {
//        fov -= yoffset; fov = std::clamp(fov, 1.0f, 45.0f);
//    }
//    void UpdateFront() {
//        float ry = glm::radians(yaw), rp = glm::radians(pitch);
//        glm::vec3 f{ std::cos(ry) * std::cos(rp), std::sin(rp), std::sin(ry) * std::cos(rp) };
//        front = glm::normalize(f);
//    }
//};
//
//// ===================== UBO =====================
//struct MVP {
//    glm::mat4 model;
//    glm::mat4 view;
//    glm::mat4 proj;
//};
//
//// ===================== 정점 구조체/데이터 =====================
//struct Vertex {
//    glm::vec3 pos;
//    glm::vec3 color;
//};
//
//static const std::vector<Vertex> kVertices = {
//    {{-0.5f,-0.5f,-0.5f}, {1,0,0}},
//    {{ 0.5f,-0.5f,-0.5f}, {0,1,0}},
//    {{ 0.5f, 0.5f,-0.5f}, {0,0,1}},
//    {{-0.5f, 0.5f,-0.5f}, {1,1,0}},
//    {{-0.5f,-0.5f, 0.5f}, {1,0,1}},
//    {{ 0.5f,-0.5f, 0.5f}, {0,1,1}},
//    {{ 0.5f, 0.5f, 0.5f}, {1,1,1}},
//    {{-0.5f, 0.5f, 0.5f}, {0,0,0}},
//};
//
//static const uint32_t kIndices[] = {
//    0,1,2, 2,3,0,
//    4,5,6, 6,7,4,
//    0,4,7, 7,3,0,
//    1,5,6, 6,2,1,
//    3,2,6, 6,7,3,
//    0,1,5, 5,4,0
//};
//
//// ===================== WGSL =====================
//static const char kShader[] = R"(
//struct MVP {
//  model : mat4x4<f32>,
//  view  : mat4x4<f32>,
//  proj  : mat4x4<f32>,
//};
//@group(0) @binding(0) var<uniform> mvp : MVP;
//
//struct VSOut {
//  @builtin(position) pos : vec4f,
//  @location(0) color : vec3f
//};
//
//@vertex
//fn vertexMain(@location(0) inPos : vec3f, @location(1) inCol : vec3f) -> VSOut {
//  var out : VSOut;
//  out.pos = mvp.proj * mvp.view * mvp.model * vec4f(inPos, 1.0);
//  out.color = inCol;
//  return out;
//}
//
//@fragment
//fn fragmentMain(@location(0) color : vec3f) -> @location(0) vec4f {
//  return vec4f(color, 1.0);
//}
//)";
//
//// ===================== WebGPU 전역 =====================
//wgpu::Instance instance;
//wgpu::Adapter adapter;
//wgpu::Device device;
//wgpu::Surface surface;
//wgpu::TextureFormat format;
//wgpu::Buffer vertexBuffer, indexBuffer, uniformBuffer;
//wgpu::BindGroup bindGroup;
//wgpu::BindGroupLayout bindGroupLayout;
//wgpu::RenderPipeline pipeline;
//wgpu::Texture depthTexture;
//wgpu::TextureView depthView;
//wgpu::ShaderModule shader;
//
//const uint32_t kWidth = 1000, kHeight = 1000;
//
//// ===================== 헬퍼 =====================
//wgpu::Buffer CreateBuffer(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd{};
//    bd.size = size;
//    bd.usage = usage;
//    bd.mappedAtCreation = true;
//    wgpu::Buffer buf = device.CreateBuffer(&bd);
//    if (data && size) { std::memcpy(buf.GetMappedRange(), data, size); }
//    buf.Unmap();
//    return buf;
//}
//
//void CreateDepthTexture() {
//    wgpu::TextureDescriptor td{};
//    td.size = { kWidth, kHeight, 1 };
//    td.format = wgpu::TextureFormat::Depth24Plus;
//    td.usage = wgpu::TextureUsage::RenderAttachment;
//    depthTexture = device.CreateTexture(&td);
//    depthView = depthTexture.CreateView();
//}
//
//void ConfigureSurface() {
//    wgpu::SurfaceCapabilities caps{};
//    surface.GetCapabilities(adapter, &caps);
//    format = caps.formats[0];
//    wgpu::SurfaceConfiguration cfg{};
//    cfg.device = device;
//    cfg.format = format;
//    cfg.width = kWidth;
//    cfg.height = kHeight;
//    cfg.presentMode = wgpu::PresentMode::Fifo;
//    surface.Configure(&cfg);
//}
//
//void Init() {
//    const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
//    wgpu::InstanceDescriptor id{};
//    id.requiredFeatureCount = 1;
//    id.requiredFeatures = &kTimedWaitAny;
//    instance = wgpu::CreateInstance(&id);
//
//    wgpu::Future f1 = instance.RequestAdapter(nullptr, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView msg) {
//            if (s != wgpu::RequestAdapterStatus::Success) {
//                std::cerr << "RequestAdapter failed: " << msg << "\n"; std::exit(1);
//            }
//            adapter = std::move(a);
//        });
//    instance.WaitAny(f1, UINT64_MAX);
//
//    wgpu::DeviceDescriptor dd{};
//    dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView m) {
//        std::cerr << "Device error(" << int(t) << "): " << m << "\n";
//        });
//    wgpu::Future f2 = adapter.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView msg) {
//            if (s != wgpu::RequestDeviceStatus::Success) {
//                std::cerr << "RequestDevice failed: " << msg << "\n"; std::exit(1);
//            }
//            device = std::move(d);
//        });
//    instance.WaitAny(f2, UINT64_MAX);
//}
//
//// ===================== 리소스 생성 =====================
//void CreateBuffers() {
//    vertexBuffer = CreateBuffer(
//        kVertices.data(),
//        kVertices.size() * sizeof(Vertex),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst
//    );
//    indexBuffer = CreateBuffer(
//        kIndices,
//        sizeof(kIndices),
//        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst
//    );
//
//    MVP mvp{}; // 초기값 (첫 프레임에 업데이트)
//    uniformBuffer = CreateBuffer(
//        &mvp,
//        sizeof(MVP),
//        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst
//    );
//}
//
//void CreateBindGroupLayout() {
//    wgpu::BindGroupLayoutEntry e{};
//    e.binding = 0;
//    e.visibility = wgpu::ShaderStage::Vertex;
//    e.buffer.type = wgpu::BufferBindingType::Uniform;
//    e.buffer.minBindingSize = sizeof(MVP);
//
//    wgpu::BindGroupLayoutDescriptor d{};
//    d.entryCount = 1;
//    d.entries = &e;
//    bindGroupLayout = device.CreateBindGroupLayout(&d);
//}
//
//void CreateBindGroup() {
//    wgpu::BindGroupEntry be{};
//    be.binding = 0;
//    be.buffer = uniformBuffer;
//    be.offset = 0;
//    be.size = sizeof(MVP);
//
//    wgpu::BindGroupDescriptor bgd{};
//    bgd.layout = bindGroupLayout;
//    bgd.entryCount = 1;
//    bgd.entries = &be;
//    bindGroup = device.CreateBindGroup(&bgd);
//}
//
//void CreateShaderModule() {
//    wgpu::ShaderSourceWGSL wgsl{};
//    wgsl.code = kShader;
//    wgpu::ShaderModuleDescriptor smd{};
//    smd.nextInChain = &wgsl;
//    shader = device.CreateShaderModule(&smd);
//}
//
//void CreateRenderPipeline() {
//    // 정점 레이아웃: pos(vec3) + color(vec3)
//    wgpu::VertexAttribute attrs[2]{};
//    attrs[0].format = wgpu::VertexFormat::Float32x3;
//    attrs[0].offset = 0;
//    attrs[0].shaderLocation = 0;
//
//    attrs[1].format = wgpu::VertexFormat::Float32x3;
//    attrs[1].offset = static_cast<uint64_t>(offsetof(Vertex, color));
//    attrs[1].shaderLocation = 1;
//
//    wgpu::VertexBufferLayout vbl{};
//    vbl.arrayStride = sizeof(Vertex);
//    vbl.attributeCount = 2;
//    vbl.attributes = attrs;
//    vbl.stepMode = wgpu::VertexStepMode::Vertex;
//
//    wgpu::ColorTargetState ct{};
//    ct.format = format;
//
//    wgpu::FragmentState fs{};
//    fs.module = shader;
//    fs.entryPoint = "fragmentMain";
//    fs.targetCount = 1;
//    fs.targets = &ct;
//
//    wgpu::DepthStencilState ds{};
//    ds.format = wgpu::TextureFormat::Depth24Plus;
//    ds.depthWriteEnabled = true;
//    ds.depthCompare = wgpu::CompareFunction::Less;
//
//    wgpu::PipelineLayoutDescriptor pld{};
//    pld.bindGroupLayoutCount = 1;
//    pld.bindGroupLayouts = &bindGroupLayout;
//    wgpu::PipelineLayout pl = device.CreatePipelineLayout(&pld);
//
//    wgpu::RenderPipelineDescriptor rpd{};
//    rpd.layout = pl;
//    rpd.vertex.module = shader;
//    rpd.vertex.entryPoint = "vertexMain";
//    rpd.vertex.bufferCount = 1;
//    rpd.vertex.buffers = &vbl;
//    rpd.fragment = &fs;
//    rpd.depthStencil = &ds;
//
//    pipeline = device.CreateRenderPipeline(&rpd);
//}
//
//// ===================== 초기화 묶음 =====================
//void InitGraphics() {
//    ConfigureSurface();
//    CreateDepthTexture();
//    CreateBuffers();
//    CreateBindGroupLayout();
//    CreateBindGroup();
//    CreateShaderModule();
//    CreateRenderPipeline();
//}
//
//// ===================== 입력/루프 =====================
//Camera camera;
//float deltaTime = 0.0f, lastFrame = 0.0f;
//bool keys[1024]{};
//
//void KeyCallback(GLFWwindow*, int key, int, int action, int) {
//    if (key >= 0 && key < 1024) {
//        if (action == GLFW_PRESS) keys[key] = true;
//        else if (action == GLFW_RELEASE) keys[key] = false;
//    }
//}
//void MouseCallback(GLFWwindow*, double x, double y) { camera.ProcessMouse((float)x, (float)y); }
//void ScrollCallback(GLFWwindow*, double, double yoff) { camera.ProcessScroll((float)yoff); }
//void MouseButtonCallback(GLFWwindow* w, int button, int action, int) {
//    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
//        glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
//    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
//        glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//}
//
//void Render() {
//    float now = (float)glfwGetTime();
//    deltaTime = now - lastFrame; lastFrame = now;
//    for (int i = 0; i < 1024; ++i) if (keys[i]) camera.ProcessKeyboard(i, deltaTime);
//
//    MVP mvp{};
//    mvp.model = glm::mat4(1.0f);
//    glm::vec3 center = camera.pos + camera.front;
//    mvp.view = glm::lookAtRH(camera.pos, center, camera.up);
//    float aspect = float(kWidth) / float(kHeight);
//    mvp.proj = glm::perspectiveRH_ZO(glm::radians(camera.fov), aspect, 0.1f, 100.0f);
//
//    device.GetQueue().WriteBuffer(uniformBuffer, 0, &mvp, sizeof(MVP));
//
//    wgpu::SurfaceTexture st{};
//    surface.GetCurrentTexture(&st);
//    wgpu::TextureView backbuffer = st.texture.CreateView();
//
//    wgpu::RenderPassColorAttachment ca{};
//    ca.view = backbuffer;
//    ca.loadOp = wgpu::LoadOp::Clear;
//    ca.storeOp = wgpu::StoreOp::Store;
//    ca.clearValue = { 0.1,0.1,0.15,1.0 };
//
//    wgpu::RenderPassDepthStencilAttachment da{};
//    da.view = depthView;
//    da.depthLoadOp = wgpu::LoadOp::Clear;
//    da.depthStoreOp = wgpu::StoreOp::Store;
//    da.depthClearValue = 1.0f;
//
//    wgpu::RenderPassDescriptor rp{};
//    rp.colorAttachmentCount = 1;
//    rp.colorAttachments = &ca;
//    rp.depthStencilAttachment = &da;
//
//    wgpu::CommandEncoder enc = device.CreateCommandEncoder();
//    {
//        wgpu::RenderPassEncoder pass = enc.BeginRenderPass(&rp);
//        pass.SetPipeline(pipeline);
//        pass.SetBindGroup(0, bindGroup);
//        pass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize);
//        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize);
//        pass.DrawIndexed(36, 1, 0, 0, 0);
//        pass.End();
//    }
//    wgpu::CommandBuffer cmd = enc.Finish();
//    device.GetQueue().Submit(1, &cmd);
//}
//
//void Start() {
//    if (!glfwInit()) return;
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "WebGPU 3D Cube (GLM+Vertex)", nullptr, nullptr);
//    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
//
//    InitGraphics();
//    glfwSetKeyCallback(window, KeyCallback);
//    glfwSetCursorPosCallback(window, MouseCallback);
//    glfwSetScrollCallback(window, ScrollCallback);
//    glfwSetMouseButtonCallback(window, MouseButtonCallback);
//    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//
//    #if defined(__EMSCRIPTEN__)
//      emscripten_set_main_loop(Render, 0, false);
//    #else
//      while (!glfwWindowShouldClose(window)) {
//        glfwPollEvents();
//        Render();
//        surface.Present();
//        instance.ProcessEvents();
//      }
//      glfwDestroyWindow(window);
//      glfwTerminate();
//    #endif
//}
//
//int main() {
//    Init();
//    Start();
//    return 0;
//}
