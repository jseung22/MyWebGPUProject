//// ===================== WebGPU + GLFW�� 3D ť�� �ν��Ͻ� ������ =====================
//// 50x50x50(�� 125,000��) ť�긦 �ν��Ͻ����� ��ġ
//// ī�޶� �̵�/ȸ��, MVP ���, UBO, Depth Test, ���콺/Ű���� �Է�, �ν��Ͻ� ����
//
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <chrono>
//#include <cstring>
//#include <GLFW/glfw3.h>
//#if defined(__EMSCRIPTEN__)
//#include <emscripten/emscripten.h>
//#endif
//#include <dawn/webgpu_cpp_print.h>
//#include <webgpu/webgpu_cpp.h>
//#include <webgpu/webgpu_glfw.h>
//
//// ===================== ī�޶� Ŭ���� =====================
//// ī�޶� ��ġ, ����, �þ߰�(FOV) �� �Է� ó�� ��� ���
//struct Camera {
//    float pos[3] = {0.0f, 0.0f, 150.0f}; // ī�޶� �ʱ� ��ġ (�ָ��� ��ü ť�� ����)
//    float front[3] = {0.0f, 0.0f, -1.0f}; // ī�޶� �ٶ󺸴� ����
//    float up[3] = {0.0f, 1.0f, 0.0f}; // ī�޶��� ���� ����
//    float yaw = -90.0f, pitch = 0.0f; // ���콺 ȸ�� ����
//    float fov = 45.0f; // �þ߰�
//    float lastX = 500.0f, lastY = 500.0f; // ���콺 ��ġ ����
//    bool firstMouse = true; // ù ���콺 �Է� ����
//
//    // Ű���� �Է¿� ���� ī�޶� �̵� (WASD)
//    void ProcessKeyboard(int key, float deltaTime) {
//        float speed = 30.0f * deltaTime; // ���� 10.0f -> 30.0f�� �� ������
//        float right[3];
//        Cross(front, up, right); // ������ ���� ���
//        Normalize(right);
//        if (key == GLFW_KEY_W) Move(front, speed);      // ������ �̵�
//        if (key == GLFW_KEY_S) Move(front, -speed);     // �ڷ� �̵�
//        if (key == GLFW_KEY_A) Move(right, -speed);     // ���� �̵�
//        if (key == GLFW_KEY_D) Move(right, speed);      // ������ �̵�
//    }
//
//    // ���콺 �̵��� ���� ī�޶� ȸ��
//    void ProcessMouse(float xpos, float ypos) {
//        if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
//        float xoffset = xpos - lastX;
//        float yoffset = lastY - ypos;
//        lastX = xpos; lastY = ypos;
//        float sensitivity = 0.1f;
//        xoffset *= sensitivity; yoffset *= sensitivity;
//        yaw += xoffset; pitch += yoffset;
//        if (pitch > 89.0f) pitch = 89.0f;
//        if (pitch < -89.0f) pitch = -89.0f;
//        UpdateFront(); // ���� ���� ����
//    }
//
//    // ���콺 �ٷ� FOV(��) ����
//    void ProcessScroll(float yoffset) {
//        fov -= yoffset;
//        if (fov < 1.0f) fov = 1.0f;
//        if (fov > 45.0f) fov = 45.0f;
//    }
//
//    // yaw/pitch ������ front ���� ����
//    void UpdateFront() {
//        float radYaw = yaw * 3.14159265f / 180.0f;
//        float radPitch = pitch * 3.14159265f / 180.0f;
//        front[0] = cos(radYaw) * cos(radPitch);
//        front[1] = sin(radPitch);
//        front[2] = sin(radYaw) * cos(radPitch);
//        Normalize(front);
//    }
//
//    // ���� ����(dir)�� ī�޶� ��ġ �̵�
//    void Move(const float* dir, float amt) {
//        for (int i = 0; i < 3; ++i)
//            pos[i] += dir[i] * amt;
//    }
//
//    // ���� ����
//    static void Cross(const float* a, const float* b, float* out) {
//        out[0] = a[1]*b[2] - a[2]*b[1];
//        out[1] = a[2]*b[0] - a[0]*b[2];
//        out[2] = a[0]*b[1] - a[1]*b[0];
//    }
//
//    // ���� ����ȭ
//    static void Normalize(float* v) {
//        float len = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
//        if (len > 0.00001f) {
//            v[0]/=len;
//            v[1]/=len;
//            v[2]/=len;
//        }
//    }
//};
//
//// ===================== ��� ��ƿ =====================
//// 4x4 ��� ���� �Լ�: �������, ��������, LookAt, ��İ�
//void Mat4Identity(float* m) {
//    memset(m, 0, sizeof(float)*16);
//    m[0]=m[5]=m[10]=m[15]=1.0f;
//}
//
//void Mat4Perspective(float* m, float fov, float aspect, float near, float far) {
//    float tanHalfFov = tan(fov/2.0f*3.14159265f/180.0f);
//    memset(m, 0, sizeof(float)*16);
//    m[0] = 1.0f/(aspect*tanHalfFov); // x�� ������
//    m[5] = 1.0f/tanHalfFov;          // y�� ������
//    m[10] = -(far+near)/(far-near);  // z�� ���� ��ȯ
//    m[11] = -1.0f;
//    m[14] = -(2.0f*far*near)/(far-near); // z�� ���� ��ȯ
//}
//
//void Mat4LookAt(float* m, const float* eye, const float* center, const float* up) {
//    float f[3] = {center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]};
//    Camera::Normalize(f);
//    float s[3];
//    Camera::Cross(f, up, s);
//    Camera::Normalize(s);
//    float u[3];
//    Camera::Cross(s, f, u);
//    Mat4Identity(m);
//    m[0]=s[0]; m[1]=u[0]; m[2]=-f[0];
//    m[4]=s[1]; m[5]=u[1]; m[6]=-f[1];
//    m[8]=s[2]; m[9]=u[2]; m[10]=-f[2];
//    m[12]=-(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]);
//    m[13]=-(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]);
//    m[14]=f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2];
//}
//
//void Mat4Mul(float* out, const float* a, const float* b) {
//    float r[16];
//    for(int i=0;i<4;i++)
//        for(int j=0;j<4;j++) {
//            r[i*4+j]=0;
//            for(int k=0;k<4;k++)
//                r[i*4+j]+=a[i*4+k]*b[k*4+j];
//        }
//    memcpy(out,r,sizeof(float)*16);
//}
//
//// ===================== UBO ����ü =====================
//// ���̴��� ������ ��/��/�������� ���
//struct MVP {
//    float model[16]; // �� ��� (���� ��ȯ)
//    float view[16];  // �� ��� (ī�޶� ��ȯ)
//    float proj[16];  // �������� ��� (���� ����)
//};
//
//// ===================== ť�� ����/�ε��� ������ =====================
//// 8�� ����(x,y,z,r,g,b), 12�� ��(36�� �ε���)
//const float cubeVertices[] = {
//    -0.5f,-0.5f,-0.5f, 1,0,0,
//     0.5f,-0.5f,-0.5f, 0,1,0,
//     0.5f, 0.5f,-0.5f, 0,0,1,
//    -0.5f, 0.5f,-0.5f, 1,1,0,
//    -0.5f,-0.5f, 0.5f, 1,0,1,
//     0.5f,-0.5f, 0.5f, 0,1,1,
//     0.5f, 0.5f, 0.5f, 1,1,1,
//    -0.5f, 0.5f, 0.5f, 0,0,0
//};
//const uint32_t cubeIndices[] = {
//    0,1,2, 2,3,0,
//    4,5,6, 6,7,4,
//    0,4,7, 7,3,0,
//    1,5,6, 6,2,1,
//    3,2,6, 6,7,3,
//    0,1,5, 5,4,0
//};
//
//// ===================== �ν��Ͻ� ������ ���� =====================
//// 50x50x50 ���� ���·� �� ť���� ��ġ ������(x, y, z) ����
//constexpr int GRID = 50;
//constexpr int INSTANCE_COUNT = GRID * GRID * GRID;
//std::vector<float> instanceOffsets;
//
//void GenerateInstanceOffsets() {
//    instanceOffsets.resize(INSTANCE_COUNT * 3);
//    int idx = 0;
//    float spacing = 2.2f; // ť�� ����
//    float offset = (GRID-1)*spacing/2.0f; // �߾� ����
//    for (int x=0; x<GRID; ++x) {
//        for (int y=0; y<GRID; ++y) {
//            for (int z=0; z<GRID; ++z) {
//                instanceOffsets[idx++] = x*spacing - offset;
//                instanceOffsets[idx++] = y*spacing - offset;
//                instanceOffsets[idx++] = z*spacing - offset;
//            }
//        }
//    }
//}
//
//// ===================== WGSL ���̴� =====================
//// vertex: ��/��/�������� ��ķ� ��ȯ, �ν��Ͻ� ��ġ ����, ���� ����
//// fragment: ���� ���
//static const char kShader[] = R"(
//struct MVP {
//  model : mat4x4<f32>,
//  view : mat4x4<f32>,
//  proj : mat4x4<f32>,
//};
//@group(0) @binding(0) var<uniform> mvp : MVP;
//struct VSOut {
//  @builtin(position) pos : vec4f,
//  @location(0) color : vec3f
//};
//@vertex
//fn vertexMain(
//    @location(0) inPos : vec3f,
//    @location(1) inCol : vec3f,
//    @location(2) instanceOffset : vec3f
//) -> VSOut {
//  var out : VSOut;
//  let worldPos = inPos + instanceOffset; // �ν��Ͻ� ��ġ ����
//  out.pos = mvp.proj * mvp.view * mvp.model * vec4f(worldPos, 1.0);
//  out.color = inCol;
//  return out;
//}
//@fragment
//fn fragmentMain(@location(0) color : vec3f) -> @location(0) vec4f {
//  return vec4f(color, 1.0);
//}
//)";
//
//// ===================== WebGPU �ٽ� ��ü =====================
//// GPU�� ����Ǵ� �ֿ� ��ü�� (�ν��Ͻ�, �����, ����̽�, ���������� ��)
//wgpu::Instance instance;
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
//const uint32_t kWidth = 1200, kHeight = 900; // ������ ũ��
//
//// ===================== ���� ���� ���� =====================
//// CPU �����͸� GPU ���۷� ����
//wgpu::Buffer CreateBuffer(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd;
//    bd.size = size;
//    bd.usage = usage;
//    bd.mappedAtCreation = true;
//    wgpu::Buffer buf = device.CreateBuffer(&bd);
//    std::memcpy(buf.GetMappedRange(), data, size);
//    buf.Unmap();
//    return buf;
//}
//
//// ===================== Depth Texture ���� =====================
//// ���� �׽�Ʈ�� �ؽ�ó ���� (���ٽ� ����)
//void CreateDepthTexture() {
//    wgpu::TextureDescriptor td;
//    td.size = {kWidth, kHeight, 1};
//    td.format = wgpu::TextureFormat::Depth24Plus;
//    td.usage = wgpu::TextureUsage::RenderAttachment;
//    depthTexture = device.CreateTexture(&td);
//    depthView = depthTexture.CreateView();
//}
//
//// ===================== Surface ���� =====================
//// ȭ�鿡 �׸� Surface(ĵ����) ����
//void ConfigureSurface() {
//    wgpu::SurfaceCapabilities capabilities;
//    surface.GetCapabilities(adapter, &capabilities);
//    format = capabilities.formats[0];
//    wgpu::SurfaceConfiguration config;
//    config.device = device;
//    config.format = format;
//    config.width = kWidth;
//    config.height = kHeight;
//    config.presentMode = wgpu::PresentMode::Fifo;
//    surface.Configure(&config);
//}
//
//// ===================== WebGPU �ʱ�ȭ =====================
//// �ν��Ͻ�, �����, ����̽� ����
//void Init() {
//    const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
//    wgpu::InstanceDescriptor instanceDesc;
//    instanceDesc.requiredFeatureCount = 1;
//    instanceDesc.requiredFeatures = &kTimedWaitAny;
//    instance = wgpu::CreateInstance(&instanceDesc);
//
//    wgpu::Future f1 = instance.RequestAdapter(nullptr, wgpu::CallbackMode::WaitAnyOnly,
//        [](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message) {
//            if (status != wgpu::RequestAdapterStatus::Success) {
//                std::cerr << "RequestAdapter failed: " << message << "\n";
//                std::exit(1);
//            }
//            adapter = std::move(a);
//        });
//    instance.WaitAny(f1, UINT64_MAX);
//
//    wgpu::DeviceDescriptor desc;
//    desc.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType errorType, wgpu::StringView message) {
//        std::cerr << "Device error (" << int(errorType) << "): " << message << "\n";
//    });
//    wgpu::Future f2 = adapter.RequestDevice(&desc, wgpu::CallbackMode::WaitAnyOnly,
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
//// ===================== �׷��� ���ҽ� �ʱ�ȭ =====================
//// Surface, Depth, ����, ���������� �� GPU ���ҽ� ����
//void InitGraphics() {
//    // Surface �� Depth �ؽ�ó ����
//    ConfigureSurface();
//    CreateDepthTexture();
//
//    // ����/�ε���/�ν��Ͻ� ���� ����
//    vertexBuffer = CreateBuffer(
//        cubeVertices,
//        sizeof(cubeVertices),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst
//    );
//
//    indexBuffer = CreateBuffer(
//        cubeIndices,
//        sizeof(cubeIndices),
//        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst
//    );
//
//    GenerateInstanceOffsets();
//    instanceBuffer = CreateBuffer(
//        instanceOffsets.data(),
//        instanceOffsets.size() * sizeof(float),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst
//    );
//
//    // BindGroupLayout: UBO ���̾ƿ� ����
//    wgpu::BindGroupLayoutEntry entry;
//    entry.binding = 0;
//    entry.visibility = wgpu::ShaderStage::Vertex;
//    entry.buffer.type = wgpu::BufferBindingType::Uniform;
//    entry.buffer.minBindingSize = sizeof(MVP);
//
//    wgpu::BindGroupLayoutDescriptor layoutDesc;
//    layoutDesc.entryCount = 1;
//    layoutDesc.entries = &entry;
//    bindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
//
//    // UniformBuffer ���� �� �ʱ�ȭ
//    MVP mvp = {};
//    uniformBuffer = CreateBuffer(
//        &mvp,
//        sizeof(MVP),
//        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst
//    );
//
//    // BindGroup: UBO�� ���̴��� ���ε�
//    wgpu::BindGroupEntry bentry;
//    bentry.binding = 0;
//    bentry.buffer = uniformBuffer;
//    bentry.offset = 0;
//    bentry.size = sizeof(MVP);
//
//    wgpu::BindGroupDescriptor bgDesc;
//    bgDesc.layout = bindGroupLayout;
//    bgDesc.entryCount = 1;
//    bgDesc.entries = &bentry;
//    bindGroup = device.CreateBindGroup(&bgDesc);
//
//    // RenderPipeline: ���̴�, ���� ���̾ƿ�, depth ����
//    wgpu::ShaderSourceWGSL wgsl;
//    wgsl.code = kShader;
//    wgpu::ShaderModuleDescriptor smDesc;
//    smDesc.nextInChain = &wgsl;
//    wgpu::ShaderModule shader = device.CreateShaderModule(&smDesc);
//
//    wgpu::VertexAttribute vattrs[3];
//    vattrs[0].format = wgpu::VertexFormat::Float32x3;
//    vattrs[0].offset = 0;
//    vattrs[0].shaderLocation = 0; // ��ġ
//
//    vattrs[1].format = wgpu::VertexFormat::Float32x3;
//    vattrs[1].offset = sizeof(float) * 3;
//    vattrs[1].shaderLocation = 1; // ����
//
//    wgpu::VertexBufferLayout vbl[2];
//    vbl[0].arrayStride = sizeof(float) * 6;
//    vbl[0].attributeCount = 2;
//    vbl[0].attributes = vattrs;
//    vbl[0].stepMode = wgpu::VertexStepMode::Vertex;
//
//    vattrs[2].format = wgpu::VertexFormat::Float32x3;
//    vattrs[2].offset = 0;
//    vattrs[2].shaderLocation = 2; // �ν��Ͻ� ��ġ
//
//    vbl[1].arrayStride = sizeof(float) * 3;
//    vbl[1].attributeCount = 1;
//    vbl[1].attributes = &vattrs[2];
//    vbl[1].stepMode = wgpu::VertexStepMode::Instance;
//
//    wgpu::ColorTargetState colorTarget;
//    colorTarget.format = format;
//
//    wgpu::FragmentState fs;
//    fs.module = shader;
//    fs.entryPoint = "fragmentMain";
//    fs.targetCount = 1;
//    fs.targets = &colorTarget;
//
//    wgpu::DepthStencilState ds;
//    ds.format = wgpu::TextureFormat::Depth24Plus;
//    ds.depthWriteEnabled = true; // ���̰� ���
//    ds.depthCompare = wgpu::CompareFunction::Less; // ���� �׽�Ʈ: �� ����� ���� �׸���
//    ds.stencilReadMask = 0;
//    ds.stencilWriteMask = 0;
//
//    wgpu::PipelineLayoutDescriptor plDesc;
//    plDesc.bindGroupLayoutCount = 1;
//    plDesc.bindGroupLayouts = &bindGroupLayout;
//    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&plDesc);
//
//    wgpu::RenderPipelineDescriptor rpDesc;
//    rpDesc.layout = pipelineLayout;
//    rpDesc.vertex.module = shader;
//    rpDesc.vertex.entryPoint = "vertexMain";
//    rpDesc.vertex.bufferCount = 2;
//    rpDesc.vertex.buffers = vbl;
//    rpDesc.fragment = &fs;
//    rpDesc.depthStencil = &ds;
//    pipeline = device.CreateRenderPipeline(&rpDesc);
//}
//
//// ===================== �Է� �ݹ� =====================
//// Ű/���콺 �Է� ���� ���� �� ó��
//Camera camera;
//float deltaTime = 0.0f, lastFrame = 0.0f;
//bool keys[1024] = {};
//
//void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
//    if (key >= 0 && key < 1024) {
//        if (action == GLFW_PRESS) keys[key] = true;
//        else if (action == GLFW_RELEASE) keys[key] = false;
//    }
//}
//
//void MouseCallback(GLFWwindow* window, double xpos, double ypos) {
//    camera.ProcessMouse((float)xpos, (float)ypos);
//}
//
//void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
//    camera.ProcessScroll((float)yoffset);
//}
//
//void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
//    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
//        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); // ���콺 Ŀ�� ���̱�
//    }
//    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
//        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // ���콺 Ŀ�� �����
//    }
//}
//
//// ===================== ������ ���� =====================
//// �� �����Ӹ��� ī�޶�/��� ����, �ν��Ͻ� ��ο� ȣ��
//void Render() {
//    float currentFrame = (float)glfwGetTime();
//    deltaTime = currentFrame - lastFrame;
//    lastFrame = currentFrame;
//
//    for (int i = 0; i < 1024; ++i)
//        if (keys[i]) camera.ProcessKeyboard(i, deltaTime); // �̵�Ű ó��
//
//    MVP mvp;
//    Mat4Identity(mvp.model); // �� ���(ȸ��/�̵� ����)
//    float center[3] = {
//        camera.pos[0]+camera.front[0],
//        camera.pos[1]+camera.front[1],
//        camera.pos[2]+camera.front[2]
//    };
//    Mat4LookAt(mvp.view, camera.pos, center, camera.up); // ī�޶� �� ���
//    Mat4Perspective(mvp.proj, camera.fov, (float)kWidth/(float)kHeight, 0.1f, 1000.0f); // ���� ����
//    device.GetQueue().WriteBuffer(uniformBuffer, 0, &mvp, sizeof(MVP)); // UBO ������Ʈ
//
//    wgpu::SurfaceTexture st;
//    surface.GetCurrentTexture(&st);
//    wgpu::TextureView backbuffer = st.texture.CreateView();
//
//    wgpu::RenderPassColorAttachment colorAttachment;
//    colorAttachment.view = backbuffer;
//    colorAttachment.loadOp = wgpu::LoadOp::Clear;
//    colorAttachment.storeOp = wgpu::StoreOp::Store;
//    colorAttachment.clearValue = {0.1,0.1,0.15,1.0}; // ����
//
//    wgpu::RenderPassDepthStencilAttachment depthAttachment;
//    depthAttachment.view = depthView;
//    depthAttachment.depthLoadOp = wgpu::LoadOp::Clear;
//    depthAttachment.depthStoreOp = wgpu::StoreOp::Store;
//    depthAttachment.depthClearValue = 1.0f;
//
//    wgpu::RenderPassDescriptor rpDesc;
//    rpDesc.colorAttachmentCount = 1;
//    rpDesc.colorAttachments = &colorAttachment;
//    rpDesc.depthStencilAttachment = &depthAttachment;
//
//    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
//    {
//        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&rpDesc);
//        pass.SetPipeline(pipeline); // ������ ���������� ����
//        pass.SetBindGroup(0, bindGroup); // UBO ���ε�
//        pass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize); // ���� ����
//        pass.SetVertexBuffer(1, instanceBuffer, 0, wgpu::kWholeSize); // �ν��Ͻ� ����
//        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize); // �ε��� ����
//        pass.DrawIndexed(36, INSTANCE_COUNT, 0, 0, 0); // 125,000�� ť�� �ν��Ͻ� ��ο�
//        pass.End();
//    }
//    wgpu::CommandBuffer cmd = encoder.Finish();
//    device.GetQueue().Submit(1, &cmd); // GPU�� ��� ����
//}
//
//// ===================== ���� =====================
//// GLFW ������ ����, �Է� �ݹ� ���, ������ ���� ����
//int main() {
//    Init(); // WebGPU �ʱ�ȭ
//    if (!glfwInit()) return -1;
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(
//        kWidth,
//        kHeight,
//        "WebGPU 3D Cube Instanced",
//        nullptr,
//        nullptr
//    );
//    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
//
//    InitGraphics(); // GPU ���ҽ� �غ�
//    glfwSetKeyCallback(window, KeyCallback); // Ű �Է� �ݹ�
//    glfwSetCursorPosCallback(window, MouseCallback); // ���콺 �̵� �ݹ�
//    glfwSetScrollCallback(window, ScrollCallback); // ���콺 �� �ݹ�
//    glfwSetMouseButtonCallback(window, MouseButtonCallback); // ���콺 ��ư �ݹ�
//    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // �ʱ�: ī�޶� ��Ʈ�� ���(Ŀ�� ����)
//
//    #if defined(__EMSCRIPTEN__)
//        emscripten_set_main_loop(Render, 0, false);
//    #else
//        while (!glfwWindowShouldClose(window)) {
//        glfwPollEvents(); //�Է� �̺�Ʈ ó��
//        Render(); //������
//        surface.Present(); //ȭ�鿡 ��� ǥ��
//        instance.ProcessEvents(); //WebGPU �̺�Ʈ ó��
//        }
//        glfwDestroyWindow(window);
//        glfwTerminate();
//    #endif
//
//    return 0;
//}
