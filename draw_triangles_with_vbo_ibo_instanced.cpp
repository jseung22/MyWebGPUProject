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
//// WebGPU 핵심 객체들 - GPU와 연결되는 주요 인터페이스들
//// ======================================================================================
//wgpu::Instance instance;        // WebGPU 인스턴스 - 모든 GPU 작업의 시작점
//wgpu::Adapter  adapter;         // GPU 어댑터 - 물리적 GPU 하드웨어를 나타냄
//wgpu::Device   device;          // GPU 디바이스 - 실제 GPU 리소스와 명령 실행
//wgpu::RenderPipeline pipeline;  // 렌더링 파이프라인 - GPU 렌더링 상태를 정의
//
//// Surface는 화면에 그릴 수 있는 캔버스를 나타냄 (윈도우 시스템과 연결)
//wgpu::Surface surface;
//wgpu::TextureFormat format;
//
//// ======================================================================================
//// 버퍼 객체들 - GPU 메모리에 저장되는 데이터
//// ======================================================================================
//wgpu::Buffer vertexBuffer;      // 정점 데이터 (위치 + 색상) - 삼각형 모양 정의
//wgpu::Buffer indexBuffer;       // 인덱스 데이터 - 정점 연결 순서 정의
//wgpu::Buffer instanceBuffer;    // 인스턴스 데이터 - 각 삼각형의 위치 오프셋
//
//// ======================================================================================
//// Uniform 시스템 - 셰이더에 동적 데이터 전달
//// ======================================================================================
//wgpu::Buffer uniformBuffer;           // uniform 데이터 저장 버퍼
//wgpu::BindGroup bindGroup;           // uniform buffer를 셰이더와 연결하는 그룹
//wgpu::BindGroupLayout bindGroupLayout; // bind group의 레이아웃 정의
//
//// ======================================================================================
//// 렌더링 설정 상수들
//// ======================================================================================
//const uint32_t kWidth = 1000;
//const uint32_t kHeight = 1000;
//
//static constexpr uint32_t GRID = 100;                    // 100x100 격자
//static constexpr uint32_t INSTANCE_COUNT = GRID * GRID;  // 총 10,000개 삼각형 인스턴스
//
//// ======================================================================================
//// Uniform 데이터 구조체 - CPU에서 GPU로 전달할 데이터
//// ======================================================================================
//struct UniformData {
//    float grid;         // 격자 크기 값
//    float padding[3];   // GPU 메모리 정렬을 위한 패딩 (16바이트 단위 정렬)
//};
//
//// ======================================================================================
//// Surface 설정 - 화면에 그리기 위한 캔버스 구성
//// ======================================================================================
//void ConfigureSurface() {
//    // GPU가 지원하는 surface 기능 조회
//    wgpu::SurfaceCapabilities capabilities;
//    surface.GetCapabilities(adapter, &capabilities);
//
//    // 첫 번째로 지원되는 포맷 사용 (보통 BGRA8Unorm 또는 RGBA8Unorm)
//    format = capabilities.formats[0];
//
//    // Surface 설정 구조체 생성
//    wgpu::SurfaceConfiguration config;
//    config.device = device;           // 사용할 GPU 디바이스
//    config.format = format;           // 픽셀 포맷
//    config.width = kWidth;            // 화면 너비
//    config.height = kHeight;          // 화면 높이
//    config.presentMode = wgpu::PresentMode::Fifo;  // V-Sync 방식 (60fps 제한)
//    surface.Configure(&config);
//}
//
//// ======================================================================================
//// WebGPU 초기화 - Instance, Adapter, Device 순서로 설정
//// ======================================================================================
//void Init() {
//    // TimedWaitAny 기능 요청 - 비동기 작업 대기를 위한 기능
//    const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
//
//    // WebGPU Instance 생성 - 모든 WebGPU 작업의 시작점
//    wgpu::InstanceDescriptor instanceDesc;
//    instanceDesc.requiredFeatureCount = 1;
//    instanceDesc.requiredFeatures = &kTimedWaitAny;
//    instance = wgpu::CreateInstance(&instanceDesc);
//
//    // ======================================================================================
//    // GPU Adapter 요청 - 시스템의 GPU 하드웨어 찾기
//    // ======================================================================================
//    wgpu::Future f1 = instance.RequestAdapter(
//        nullptr,                           // 기본 옵션 사용
//        wgpu::CallbackMode::WaitAnyOnly,   // 동기식 대기
//        [](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message) {
//            if (status != wgpu::RequestAdapterStatus::Success) {
//                std::cerr << "RequestAdapter failed: " << message << "\n";
//                std::exit(1);
//            }
//            adapter = std::move(a);  // 전역 변수에 adapter 저장
//        });
//    instance.WaitAny(f1, UINT64_MAX);  // 완료될 때까지 대기
//
//    // ======================================================================================
//    // GPU Device 요청 - 실제 GPU 리소스와 명령 실행을 위한 디바이스
//    // ======================================================================================
//    wgpu::DeviceDescriptor desc;
//    // GPU 에러 발생 시 콜백 함수 설정
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
//            device = std::move(d);  // 전역 변수에 device 저장
//        });
//    instance.WaitAny(f2, UINT64_MAX);  // 완료될 때까지 대기
//}
//
//// ======================================================================================
//// WGSL 셰이더 - GPU에서 실행되는 프로그램
//// ======================================================================================
//static const char kShader[] = R"(
//  // ======================================================================================
//  // Uniform 구조체 정의 - CPU에서 전달받을 데이터
//  // ======================================================================================
//  struct Uniforms {
//    grid : f32,    // 격자 크기 (100.0)
//  };
//
//  // @group(0) @binding(0): 첫 번째 bind group의 첫 번째 바인딩
//  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
//
//  // ======================================================================================
//  // 정점 셰이더 출력 구조체
//  // ======================================================================================
//  struct VSOut {
//    @builtin(position) pos : vec4f,   // 클립 공간 좌표 (필수)
//    @location(0) color : vec3f        // 색상 데이터 (fragment shader로 전달)
//  };
//
//  // ======================================================================================
//  // 정점 셰이더 - 각 정점마다 실행됨
//  // ======================================================================================
//  @vertex
//  fn vertexMain(
//      @location(0) inPos : vec2f,        // 정점 위치 (vertex buffer slot 0)
//      @location(1) inCol : vec3f,        // 정점 색상 (vertex buffer slot 0)
//      @location(2) instOffset : vec2f    // 인스턴스 오프셋 (vertex buffer slot 1)
//  ) -> VSOut {
//    // NDC(-1~1) 공간에서 셀 크기 계산
//    let step : f32 = 2.0 / uniforms.grid;    // 각 셀의 크기 (0.02)
//    let scale : f32 = 0.45 * step;           // 삼각형 크기 (셀의 45% 크기)
//
//    var out : VSOut;
//    // 최종 위치 = 인스턴스 오프셋 + (기본 삼각형 * 스케일)
//    out.pos = vec4f(instOffset + inPos * scale, 0.0, 1.0);
//    out.color = inCol;    // 색상은 그대로 전달
//    return out;
//  }
//
//  // ======================================================================================
//  // 프래그먼트 셰이더 - 각 픽셀마다 실행됨
//  // ======================================================================================
//  @fragment
//  fn fragmentMain(@location(0) color : vec3f) -> @location(0) vec4f {
//    return vec4f(color, 1.0);    // RGB + Alpha(불투명)
//  }
//)";
//
//// ======================================================================================
//// 버퍼 생성 헬퍼 함수 - 데이터를 GPU 메모리에 복사
//// ======================================================================================
//wgpu::Buffer CreateBufferFromData(const void* data, size_t size, wgpu::BufferUsage usage) {
//    wgpu::BufferDescriptor bd;
//    bd.size = size;                    // 버퍼 크기
//    bd.usage = usage;                  // 버퍼 사용 용도
//    bd.mappedAtCreation = true;        // 생성 시점에 CPU에서 접근 가능하게 설정
//    wgpu::Buffer buf = device.CreateBuffer(&bd);
//
//    // CPU 메모리의 데이터를 GPU 버퍼로 복사
//    std::memcpy(buf.GetMappedRange(), data, size);
//    buf.Unmap();    // CPU 접근 해제 (이후 GPU만 접근 가능)
//    return buf;
//}
//
//// ======================================================================================
//// 삼각형 정점 및 인덱스 버퍼 생성
//// ======================================================================================
//void CreateTriangleBuffers() {
//    // ======================================================================================
//    // 정점 데이터 정의 - 인터리브드 포맷 (위치 + 색상)
//    // ======================================================================================
//    const float vertices[] = {
//        //   x,     y,      r,    g,    b
//         0.0f,  1.0f,    1.0f, 0.3f, 0.3f,  // v0: 상단 (빨강)
//        -1.0f, -1.0f,    0.3f, 1.0f, 0.3f,  // v1: 좌하단 (초록)
//         1.0f, -1.0f,    0.3f, 0.3f, 1.0f   // v2: 우하단 (파랑)
//    };
//    // 인덱스 데이터 - 정점 연결 순서 (시계 반대 방향)
//    const uint32_t indices[] = { 0, 1, 2 };
//
//    // GPU 버퍼 생성
//    vertexBuffer = CreateBufferFromData(vertices, sizeof(vertices),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//    indexBuffer = CreateBufferFromData(indices, sizeof(indices),
//        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst);
//}
//
//// ======================================================================================
//// 인스턴스 데이터 버퍼 생성 - 10,000개 삼각형의 위치 정보
//// ======================================================================================
//void CreateInstanceBuffer() {
//    // 각 인스턴스의 위치 오프셋 저장용 벡터
//    std::vector<float> offsets;
//    offsets.reserve(INSTANCE_COUNT * 2);  // x, y 좌표이므로 *2
//
//    const float grid = static_cast<float>(GRID);     // 100.0
//    const float step = 2.0f / grid;                  // NDC 공간에서 셀 크기 (0.02)
//    const float start = -1.0f + step * 0.5f;        // 첫 번째 셀의 중심 (-0.99)
//
//    // ======================================================================================
//    // 100x100 격자의 각 셀 중심 좌표 계산
//    // ======================================================================================
//    for (uint32_t y = 0; y < GRID; ++y) {
//        for (uint32_t x = 0; x < GRID; ++x) {
//            // 각 셀의 중심 좌표 계산
//            float cx = start + step * static_cast<float>(x);  // X 좌표
//            float cy = start + step * static_cast<float>(y);  // Y 좌표
//            offsets.push_back(cx);
//            offsets.push_back(cy);
//        }
//    }
//
//    // 인스턴스 버퍼 생성 (per-instance 데이터용)
//    instanceBuffer = CreateBufferFromData(
//        offsets.data(), offsets.size() * sizeof(float),
//        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst);
//}
//
//// ======================================================================================
//// Uniform 버퍼 생성 - 셰이더에 전달할 uniform 데이터
//// ======================================================================================
//void CreateUniformBuffer() {
//    UniformData uniformData;
//    uniformData.grid = static_cast<float>(GRID);  // 격자 크기 설정 (100.0)
//
//    // Uniform 버퍼 생성
//    uniformBuffer = CreateBufferFromData(&uniformData, sizeof(UniformData),
//        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//}
//
//// ======================================================================================
//// Bind Group Layout 생성 - uniform buffer의 레이아웃 정의
//// ======================================================================================
//void CreateBindGroupLayout() {
//    wgpu::BindGroupLayoutEntry entry;
//    entry.binding = 0;                                     // @binding(0)에 해당
//    entry.visibility = wgpu::ShaderStage::Vertex;          // 정점 셰이더에서만 접근
//    entry.buffer.type = wgpu::BufferBindingType::Uniform;  // uniform buffer 타입
//    entry.buffer.minBindingSize = sizeof(UniformData);     // 최소 버퍼 크기
//
//    wgpu::BindGroupLayoutDescriptor layoutDesc;
//    layoutDesc.entryCount = 1;
//    layoutDesc.entries = &entry;
//    bindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
//}
//
//// ======================================================================================
//// Bind Group 생성 - 실제 uniform buffer를 바인딩
//// ======================================================================================
//void CreateBindGroup() {
//    wgpu::BindGroupEntry entry;
//    entry.binding = 0;                    // @binding(0)에 해당
//    entry.buffer = uniformBuffer;         // 실제 uniform buffer
//    entry.offset = 0;                     // 버퍼 시작 오프셋
//    entry.size = sizeof(UniformData);     // 바인딩할 데이터 크기
//
//    wgpu::BindGroupDescriptor bindGroupDesc;
//    bindGroupDesc.layout = bindGroupLayout;  // 위에서 만든 layout 사용
//    bindGroupDesc.entryCount = 1;
//    bindGroupDesc.entries = &entry;
//    bindGroup = device.CreateBindGroup(&bindGroupDesc);
//}
//
//// ======================================================================================
//// 렌더링 파이프라인 생성 - GPU의 렌더링 상태 정의
//// ======================================================================================
//void CreateRenderPipeline() {
//    // ======================================================================================
//    // 셰이더 모듈 생성
//    // ======================================================================================
//    wgpu::ShaderSourceWGSL wgsl;
//    wgsl.code = kShader;  // WGSL 코드
//    wgpu::ShaderModuleDescriptor smDesc;
//    smDesc.nextInChain = &wgsl;
//    wgpu::ShaderModule shader = device.CreateShaderModule(&smDesc);
//
//    // ======================================================================================
//    // 정점 버퍼 레이아웃 설정 (Slot 0: per-vertex 데이터)
//    // ======================================================================================
//    wgpu::VertexAttribute vattrs0[2];
//    // 위치 속성 (@location(0))
//    vattrs0[0].format = wgpu::VertexFormat::Float32x2;  // vec2f
//    vattrs0[0].offset = 0;                              // 버퍼 시작부터
//    vattrs0[0].shaderLocation = 0;                      // @location(0)
//
//    // 색상 속성 (@location(1))
//    vattrs0[1].format = wgpu::VertexFormat::Float32x3;  // vec3f
//    vattrs0[1].offset = sizeof(float) * 2;              // 위치 데이터 뒤부터
//    vattrs0[1].shaderLocation = 1;                      // @location(1)
//
//    wgpu::VertexBufferLayout vbl0;
//    vbl0.arrayStride = sizeof(float) * (2 + 3);         // 정점당 데이터 크기 (위치2 + 색상3)
//    vbl0.attributeCount = 2;                            // 속성 개수
//    vbl0.attributes = vattrs0;
//    vbl0.stepMode = wgpu::VertexStepMode::Vertex;       // 정점마다 데이터 진행
//
//    // ======================================================================================
//    // 인스턴스 버퍼 레이아웃 설정 (Slot 1: per-instance 데이터)
//    // ======================================================================================
//    wgpu::VertexAttribute vattrs1[1];
//    // 인스턴스 오프셋 속성 (@location(2))
//    vattrs1[0].format = wgpu::VertexFormat::Float32x2;  // vec2f
//    vattrs1[0].offset = 0;                              // 버퍼 시작부터
//    vattrs1[0].shaderLocation = 2;                      // @location(2)
//
//    wgpu::VertexBufferLayout vbl1;
//    vbl1.arrayStride = sizeof(float) * 2;               // 인스턴스당 데이터 크기 (x, y)
//    vbl1.attributeCount = 1;                            // 속성 개수
//    vbl1.attributes = vattrs1;
//    vbl1.stepMode = wgpu::VertexStepMode::Instance;     // 인스턴스마다 데이터 진행
//
//    // ======================================================================================
//    // 색상 출력 설정
//    // ======================================================================================
//    wgpu::ColorTargetState colorTarget;
//    colorTarget.format = format;  // Surface와 같은 포맷 사용
//
//    // ======================================================================================
//    // 프래그먼트 셰이더 설정
//    // ======================================================================================
//    wgpu::FragmentState fs;
//    fs.module = shader;              // 셰이더 모듈
//    fs.entryPoint = "fragmentMain";  // 프래그먼트 셰이더 진입점
//    fs.targetCount = 1;              // 색상 출력 개수
//    fs.targets = &colorTarget;
//
//    // ======================================================================================
//    // 파이프라인 레이아웃 생성 (uniform buffer 포함)
//    // ======================================================================================
//    wgpu::PipelineLayoutDescriptor layoutDesc;
//    layoutDesc.bindGroupLayoutCount = 1;
//    layoutDesc.bindGroupLayouts = &bindGroupLayout;
//    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&layoutDesc);
//
//    // ======================================================================================
//    // 렌더링 파이프라인 생성
//    // ======================================================================================
//    wgpu::RenderPipelineDescriptor rpDesc;
//    rpDesc.layout = pipelineLayout;                      // 파이프라인 레이아웃
//    rpDesc.vertex.module = shader;                       // 정점 셰이더 모듈
//    rpDesc.vertex.entryPoint = "vertexMain";             // 정점 셰이더 진입점
//
//    // 두 개의 정점 버퍼 설정 (per-vertex, per-instance)
//    wgpu::VertexBufferLayout vbuffers[2] = { vbl0, vbl1 };
//    rpDesc.vertex.bufferCount = 2;
//    rpDesc.vertex.buffers = vbuffers;
//
//    rpDesc.fragment = &fs;  // 프래그먼트 셰이더 설정
//
//    pipeline = device.CreateRenderPipeline(&rpDesc);
//}
//
//// ======================================================================================
//// 렌더링 루프 - 매 프레임마다 실행
//// ======================================================================================
//void Render() {
//    // ======================================================================================
//    // 성능 측정용 정적 변수들
//    // ======================================================================================
//    static auto lastTime = std::chrono::high_resolution_clock::now();
//    static int frameCount = 0;
//    static double totalMs = 0.0;
//    static constexpr int kReportEvery = 60;  // 60프레임마다 리포트
//
//    auto t0 = std::chrono::high_resolution_clock::now();
//
//    // ======================================================================================
//    // 렌더 타겟 준비 - Surface에서 현재 프레임의 텍스처 가져오기
//    // ======================================================================================
//    wgpu::SurfaceTexture st;
//    surface.GetCurrentTexture(&st);
//    wgpu::TextureView backbuffer = st.texture.CreateView();
//
//    // ======================================================================================
//    // 렌더 패스 설정 - 화면 클리어 및 렌더링 준비
//    // ======================================================================================
//    wgpu::RenderPassColorAttachment colorAttachment;
//    colorAttachment.view = backbuffer;                            // 렌더 타겟
//    colorAttachment.loadOp = wgpu::LoadOp::Clear;                // 화면 클리어
//    colorAttachment.storeOp = wgpu::StoreOp::Store;              // 결과 저장
//    colorAttachment.clearValue = { 0.05, 0.05, 0.06, 1.0 };     // 클리어 색상 (어두운 회색)
//
//    wgpu::RenderPassDescriptor rpDesc;
//    rpDesc.colorAttachmentCount = 1;
//    rpDesc.colorAttachments = &colorAttachment;
//
//    // ======================================================================================
//    // 명령 인코더 생성 및 렌더링 명령 기록
//    // ======================================================================================
//    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
//    {
//        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&rpDesc);
//        
//        // 렌더링 상태 설정
//        pass.SetPipeline(pipeline);                                    // 렌더링 파이프라인
//        pass.SetBindGroup(0, bindGroup);                              // uniform buffer 바인딩
//        pass.SetVertexBuffer(0, vertexBuffer, 0, wgpu::kWholeSize);    // 정점 버퍼 (slot 0)
//        pass.SetVertexBuffer(1, instanceBuffer, 0, wgpu::kWholeSize);  // 인스턴스 버퍼 (slot 1)
//        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32, 0, wgpu::kWholeSize);
//        
//        // 실제 드로우 콜 - 10,000개 삼각형 인스턴스 렌더링
//        pass.DrawIndexed(3,              // 인덱스 개수 (삼각형 = 3개 인덱스)
//                        INSTANCE_COUNT,  // 인스턴스 개수 (10,000개)
//                        0,               // 인덱스 시작 오프셋
//                        0,               // 정점 시작 오프셋
//                        0);              // 인스턴스 시작 오프셋
//        pass.End();
//    }
//    
//    // ======================================================================================
//    // 명령 제출 및 화면 표시
//    // ======================================================================================
//    wgpu::CommandBuffer cmd = encoder.Finish();  // 명령 버퍼 완성
//    device.GetQueue().Submit(1, &cmd);           // GPU에 명령 제출
//    surface.Present();                           // 화면에 결과 표시
//
//    // ======================================================================================
//    // 성능 측정 및 리포트
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
//// 그래픽 초기화 - 모든 GPU 리소스 준비
//// ======================================================================================
//void InitGraphics() {
//    ConfigureSurface();        // Surface 설정
//    CreateTriangleBuffers();   // 정점/인덱스 버퍼 생성
//    CreateInstanceBuffer();    // 인스턴스 버퍼 생성
//    CreateBindGroupLayout();   // Bind group layout 생성
//    CreateUniformBuffer();     // Uniform buffer 생성
//    CreateBindGroup();         // Bind group 생성
//    CreateRenderPipeline();    // 렌더링 파이프라인 생성
//}
//
//// ======================================================================================
//// 윈도우 생성 및 메인 루프 시작
//// ======================================================================================
//void Start() {
//    if (!glfwInit()) return;
//
//    // OpenGL API 사용 안함 (WebGPU 사용)
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "WebGPU instanced triangles (VBO+IBO)", nullptr, nullptr);
//
//    // GLFW 윈도우와 WebGPU Surface 연결
//    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
//    InitGraphics();
//
//#if defined(__EMSCRIPTEN__)
//    // 웹 환경: 브라우저의 애니메이션 루프 사용
//    emscripten_set_main_loop([]() { Render(); }, 0, false);
//#else
//    // 네이티브 환경: 직접 루프 실행
//    while (!glfwWindowShouldClose(window)) {
//        glfwPollEvents();           // 윈도우 이벤트 처리
//        Render();                   // 렌더링 실행
//        instance.ProcessEvents();   // WebGPU 이벤트 처리
//    }
//    glfwDestroyWindow(window);
//    glfwTerminate();
//#endif
//}
//
//// ======================================================================================
//// 프로그램 진입점
//// ======================================================================================
//int main() {
//    Init();    // WebGPU 초기화 (Instance, Adapter, Device)
//    Start();   // 윈도우 생성 및 렌더링 루프 시작
//    return 0;
//}
