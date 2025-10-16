//// ===================== WebGPU + GLFW로 3D 큐브 인스턴스 렌더링 =====================
//// 50x50x50(총 125,000개) 큐브를 인스턴싱으로 배치
//// 카메라 이동/회전, MVP 행렬, UBO, Depth Test, 마우스/키보드 입력, 인스턴스 버퍼
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
//// ===================== 카메라 클래스 =====================
//// 카메라 위치, 방향, 시야각(FOV) 및 입력 처리 기능 담당
//struct Camera {
//    float pos[3] = {0.0f, 0.0f, 150.0f}; // 카메라 초기 위치 (멀리서 전체 큐브 관찰)
//    float front[3] = {0.0f, 0.0f, -1.0f}; // 카메라가 바라보는 방향
//    float up[3] = {0.0f, 1.0f, 0.0f}; // 카메라의 위쪽 벡터
//    float yaw = -90.0f, pitch = 0.0f; // 마우스 회전 각도
//    float fov = 45.0f; // 시야각
//    float lastX = 500.0f, lastY = 500.0f; // 마우스 위치 저장
//    bool firstMouse = true; // 첫 마우스 입력 여부
//
//    // 키보드 입력에 따라 카메라 이동 (WASD)
//    void ProcessKeyboard(int key, float deltaTime) {
//        float speed = 30.0f * deltaTime; // 기존 10.0f -> 30.0f로 더 빠르게
//        float right[3];
//        Cross(front, up, right); // 오른쪽 벡터 계산
//        Normalize(right);
//        if (key == GLFW_KEY_W) Move(front, speed);      // 앞으로 이동
//        if (key == GLFW_KEY_S) Move(front, -speed);     // 뒤로 이동
//        if (key == GLFW_KEY_A) Move(right, -speed);     // 왼쪽 이동
//        if (key == GLFW_KEY_D) Move(right, speed);      // 오른쪽 이동
//    }
//
//    // 마우스 이동에 따라 카메라 회전
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
//        UpdateFront(); // 방향 벡터 갱신
//    }
//
//    // 마우스 휠로 FOV(줌) 조절
//    void ProcessScroll(float yoffset) {
//        fov -= yoffset;
//        if (fov < 1.0f) fov = 1.0f;
//        if (fov > 45.0f) fov = 45.0f;
//    }
//
//    // yaw/pitch 각도로 front 벡터 갱신
//    void UpdateFront() {
//        float radYaw = yaw * 3.14159265f / 180.0f;
//        float radPitch = pitch * 3.14159265f / 180.0f;
//        front[0] = cos(radYaw) * cos(radPitch);
//        front[1] = sin(radPitch);
//        front[2] = sin(radYaw) * cos(radPitch);
//        Normalize(front);
//    }
//
//    // 방향 벡터(dir)로 카메라 위치 이동
//    void Move(const float* dir, float amt) {
//        for (int i = 0; i < 3; ++i)
//            pos[i] += dir[i] * amt;
//    }
//
//    // 벡터 외적
//    static void Cross(const float* a, const float* b, float* out) {
//        out[0] = a[1]*b[2] - a[2]*b[1];
//        out[1] = a[2]*b[0] - a[0]*b[2];
//        out[2] = a[0]*b[1] - a[1]*b[0];
//    }
//
//    // 벡터 정규화
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
//// ===================== 행렬 유틸 =====================
//// 4x4 행렬 관련 함수: 단위행렬, 원근투영, LookAt, 행렬곱
//void Mat4Identity(float* m) {
//    memset(m, 0, sizeof(float)*16);
//    m[0]=m[5]=m[10]=m[15]=1.0f;
//}
//
//void Mat4Perspective(float* m, float fov, float aspect, float near, float far) {
//    float tanHalfFov = tan(fov/2.0f*3.14159265f/180.0f);
//    memset(m, 0, sizeof(float)*16);
//    m[0] = 1.0f/(aspect*tanHalfFov); // x축 스케일
//    m[5] = 1.0f/tanHalfFov;          // y축 스케일
//    m[10] = -(far+near)/(far-near);  // z축 깊이 변환
//    m[11] = -1.0f;
//    m[14] = -(2.0f*far*near)/(far-near); // z축 깊이 변환
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
//// ===================== UBO 구조체 =====================
//// 셰이더에 전달할 모델/뷰/프로젝션 행렬
//struct MVP {
//    float model[16]; // 모델 행렬 (월드 변환)
//    float view[16];  // 뷰 행렬 (카메라 변환)
//    float proj[16];  // 프로젝션 행렬 (원근 투영)
//};
//
//// ===================== 큐브 정점/인덱스 데이터 =====================
//// 8개 정점(x,y,z,r,g,b), 12개 면(36개 인덱스)
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
//// ===================== 인스턴스 데이터 생성 =====================
//// 50x50x50 격자 형태로 각 큐브의 위치 오프셋(x, y, z) 생성
//constexpr int GRID = 50;
//constexpr int INSTANCE_COUNT = GRID * GRID * GRID;
//std::vector<float> instanceOffsets;
//
//void GenerateInstanceOffsets() {
//    instanceOffsets.resize(INSTANCE_COUNT * 3);
//    int idx = 0;
//    float spacing = 2.2f; // 큐브 간격
//    float offset = (GRID-1)*spacing/2.0f; // 중앙 정렬
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
//// ===================== WGSL 셰이더 =====================
//// vertex: 모델/뷰/프로젝션 행렬로 변환, 인스턴스 위치 적용, 색상 전달
//// fragment: 색상 출력
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
//  let worldPos = inPos + instanceOffset; // 인스턴스 위치 적용
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
//// ===================== WebGPU 핵심 객체 =====================
//// GPU와 연결되는 주요 객체들 (인스턴스, 어댑터, 디바이스, 파이프라인 등)
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
//const uint32_t kWidth = 1200, kHeight = 900; // 윈도우 크기
//
//// ===================== 버퍼 생성 헬퍼 =====================
//// CPU 데이터를 GPU 버퍼로 복사
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
//// ===================== Depth Texture 생성 =====================
//// 깊이 테스트용 텍스처 생성 (스텐실 없음)
//void CreateDepthTexture() {
//    wgpu::TextureDescriptor td;
//    td.size = {kWidth, kHeight, 1};
//    td.format = wgpu::TextureFormat::Depth24Plus;
//    td.usage = wgpu::TextureUsage::RenderAttachment;
//    depthTexture = device.CreateTexture(&td);
//    depthView = depthTexture.CreateView();
//}
//
//// ===================== Surface 설정 =====================
//// 화면에 그릴 Surface(캔버스) 설정
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
//// ===================== WebGPU 초기화 =====================
//// 인스턴스, 어댑터, 디바이스 생성
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
//// ===================== 그래픽 리소스 초기화 =====================
//// Surface, Depth, 버퍼, 파이프라인 등 GPU 리소스 생성
//void InitGraphics() {
//    // Surface 및 Depth 텍스처 생성
//    ConfigureSurface();
//    CreateDepthTexture();
//
//    // 정점/인덱스/인스턴스 버퍼 생성
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
//    // BindGroupLayout: UBO 레이아웃 정의
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
//    // UniformBuffer 생성 및 초기화
//    MVP mvp = {};
//    uniformBuffer = CreateBuffer(
//        &mvp,
//        sizeof(MVP),
//        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst
//    );
//
//    // BindGroup: UBO를 셰이더에 바인딩
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
//    // RenderPipeline: 셰이더, 버퍼 레이아웃, depth 설정
//    wgpu::ShaderSourceWGSL wgsl;
//    wgsl.code = kShader;
//    wgpu::ShaderModuleDescriptor smDesc;
//    smDesc.nextInChain = &wgsl;
//    wgpu::ShaderModule shader = device.CreateShaderModule(&smDesc);
//
//    wgpu::VertexAttribute vattrs[3];
//    vattrs[0].format = wgpu::VertexFormat::Float32x3;
//    vattrs[0].offset = 0;
//    vattrs[0].shaderLocation = 0; // 위치
//
//    vattrs[1].format = wgpu::VertexFormat::Float32x3;
//    vattrs[1].offset = sizeof(float) * 3;
//    vattrs[1].shaderLocation = 1; // 색상
//
//    wgpu::VertexBufferLayout vbl[2];
//    vbl[0].arrayStride = sizeof(float) * 6;
//    vbl[0].attributeCount = 2;
//    vbl[0].attributes = vattrs;
//    vbl[0].stepMode = wgpu::VertexStepMode::Vertex;
//
//    vattrs[2].format = wgpu::VertexFormat::Float32x3;
//    vattrs[2].offset = 0;
//    vattrs[2].shaderLocation = 2; // 인스턴스 위치
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
//    ds.depthWriteEnabled = true; // 깊이값 기록
//    ds.depthCompare = wgpu::CompareFunction::Less; // 깊이 테스트: 더 가까운 값만 그리기
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
//// ===================== 입력 콜백 =====================
//// 키/마우스 입력 상태 저장 및 처리
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
//        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); // 마우스 커서 보이기
//    }
//    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
//        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // 마우스 커서 숨기기
//    }
//}
//
//// ===================== 렌더링 루프 =====================
//// 매 프레임마다 카메라/행렬 갱신, 인스턴스 드로우 호출
//void Render() {
//    float currentFrame = (float)glfwGetTime();
//    deltaTime = currentFrame - lastFrame;
//    lastFrame = currentFrame;
//
//    for (int i = 0; i < 1024; ++i)
//        if (keys[i]) camera.ProcessKeyboard(i, deltaTime); // 이동키 처리
//
//    MVP mvp;
//    Mat4Identity(mvp.model); // 모델 행렬(회전/이동 없음)
//    float center[3] = {
//        camera.pos[0]+camera.front[0],
//        camera.pos[1]+camera.front[1],
//        camera.pos[2]+camera.front[2]
//    };
//    Mat4LookAt(mvp.view, camera.pos, center, camera.up); // 카메라 뷰 행렬
//    Mat4Perspective(mvp.proj, camera.fov, (float)kWidth/(float)kHeight, 0.1f, 1000.0f); // 원근 투영
//    device.GetQueue().WriteBuffer(uniformBuffer, 0, &mvp, sizeof(MVP)); // UBO 업데이트
//
//    wgpu::SurfaceTexture st;
//    surface.GetCurrentTexture(&st);
//    wgpu::TextureView backbuffer = st.texture.CreateView();
//
//    wgpu::RenderPassColorAttachment colorAttachment;
//    colorAttachment.view = backbuffer;
//    colorAttachment.loadOp = wgpu::LoadOp::Clear;
//    colorAttachment.storeOp = wgpu::StoreOp::Store;
//    colorAttachment.clearValue = {0.1,0.1,0.15,1.0}; // 배경색
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
//        pass.SetPipeline(pipeline); // 렌더링 파이프라인 설정
//        pass.SetBindGroup(0, bindGroup); // UBO 바인딩
//        pass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize); // 정점 버퍼
//        pass.SetVertexBuffer(1, instanceBuffer, 0, wgpu::kWholeSize); // 인스턴스 버퍼
//        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize); // 인덱스 버퍼
//        pass.DrawIndexed(36, INSTANCE_COUNT, 0, 0, 0); // 125,000개 큐브 인스턴스 드로우
//        pass.End();
//    }
//    wgpu::CommandBuffer cmd = encoder.Finish();
//    device.GetQueue().Submit(1, &cmd); // GPU에 명령 제출
//}
//
//// ===================== 메인 =====================
//// GLFW 윈도우 생성, 입력 콜백 등록, 렌더링 루프 실행
//int main() {
//    Init(); // WebGPU 초기화
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
//    InitGraphics(); // GPU 리소스 준비
//    glfwSetKeyCallback(window, KeyCallback); // 키 입력 콜백
//    glfwSetCursorPosCallback(window, MouseCallback); // 마우스 이동 콜백
//    glfwSetScrollCallback(window, ScrollCallback); // 마우스 휠 콜백
//    glfwSetMouseButtonCallback(window, MouseButtonCallback); // 마우스 버튼 콜백
//    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // 초기: 카메라 컨트롤 모드(커서 숨김)
//
//    #if defined(__EMSCRIPTEN__)
//        emscripten_set_main_loop(Render, 0, false);
//    #else
//        while (!glfwWindowShouldClose(window)) {
//        glfwPollEvents(); //입력 이벤트 처리
//        Render(); //렌더링
//        surface.Present(); //화면에 결과 표시
//        instance.ProcessEvents(); //WebGPU 이벤트 처리
//        }
//        glfwDestroyWindow(window);
//        glfwTerminate();
//    #endif
//
//    return 0;
//}
