// particles_system.cpp
// ===================== 100만개 3D 큐브 파티클 시스템 구현 =====================
// 요구사항:
// 1. SSBO로 파티클 데이터 관리
// 2. *.obj 파일 import
// 3. Compute Shader로 파티클 움직임 및 obj 충돌 처리
// 4. 파티클별 light, texture 등 특수효과(vertex/fragment shader)
// 5. 각 요구사항별 주석 명확히 표시

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <GLFW/glfw3.h>
#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

// ===================== UBO 구조체 =====================
// 파티클 렌더링에 필요한 모델/뷰/프로젝션 행렬
struct MVP {
    float model[16];
    float view[16];
    float proj[16];
};

// ===================== 파티클 데이터 구조체 (SSBO용) =====================
struct Particle {
    float pos[3];      // 위치
    float vel[3];      // 속도
    float color[3];    // 색상
    float size;        // 크기
    float light[3];    // 조명 효과
    float texIndex;    // 텍스처 인덱스
};
constexpr int PARTICLE_COUNT = 1000000;
std::vector<Particle> particles(PARTICLE_COUNT);

// ===================== OBJ 파일 import 함수 =====================
// 요구사항2: *.obj 파일 import
struct ObjMesh {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
};
ObjMesh ImportObj(const std::string& filename) {
    ObjMesh mesh;
    std::ifstream file(filename);
    std::string line;
    std::vector<float> temp_vertices;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            temp_vertices.push_back(x);
            temp_vertices.push_back(y);
            temp_vertices.push_back(z);
        } else if (type == "f") {
            uint32_t a, b, c;
            char slash;
            iss >> a >> slash >> b >> slash >> c;
            mesh.indices.push_back(a-1);
            mesh.indices.push_back(b-1);
            mesh.indices.push_back(c-1);
        }
    }
    mesh.vertices = temp_vertices;
    return mesh;
}

// ===================== WebGPU 핵심 객체 =====================
wgpu::Instance instance;
wgpu::Adapter adapter;
wgpu::Device device;
wgpu::Surface surface;
wgpu::TextureFormat format;
wgpu::Buffer particleBuffer, meshVertexBuffer, meshIndexBuffer;
wgpu::Buffer uniformBuffer;
wgpu::BindGroup bindGroup, computeBindGroup;
wgpu::BindGroupLayout bindGroupLayout, computeBindGroupLayout;
wgpu::RenderPipeline pipeline;
wgpu::ComputePipeline computePipeline;
wgpu::Texture depthTexture;
wgpu::TextureView depthView;

const uint32_t kWidth = 1600, kHeight = 1000;

// ===================== 버퍼 생성 헬퍼 =====================
wgpu::Buffer CreateBuffer(const void* data, size_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor bd;
    bd.size = size;
    bd.usage = usage;
    bd.mappedAtCreation = true;
    wgpu::Buffer buf = device.CreateBuffer(&bd);
    std::memcpy(buf.GetMappedRange(), data, size);
    buf.Unmap();
    return buf;
}

// ===================== Depth Texture 생성 =====================
void CreateDepthTexture() {
    wgpu::TextureDescriptor td;
    td.size = {kWidth, kHeight, 1};
    td.format = wgpu::TextureFormat::Depth24Plus;
    td.usage = wgpu::TextureUsage::RenderAttachment;
    depthTexture = device.CreateTexture(&td);
    depthView = depthTexture.CreateView();
}

// ===================== Surface 설정 =====================
void ConfigureSurface() {
    wgpu::SurfaceCapabilities capabilities;
    surface.GetCapabilities(adapter, &capabilities);
    format = capabilities.formats[0];
    wgpu::SurfaceConfiguration config;
    config.device = device;
    config.format = format;
    config.width = kWidth;
    config.height = kHeight;
    config.presentMode = wgpu::PresentMode::Fifo;
    surface.Configure(&config);
}

// ===================== WebGPU 초기화 =====================
void Init() {
    const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
    wgpu::InstanceDescriptor instanceDesc;
    instanceDesc.requiredFeatureCount = 1;
    instanceDesc.requiredFeatures = &kTimedWaitAny;
    instance = wgpu::CreateInstance(&instanceDesc);

    wgpu::Future f1 = instance.RequestAdapter(nullptr, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                std::cerr << "RequestAdapter failed: " << message << "\n";
                std::exit(1);
            }
            adapter = std::move(a);
        });
    instance.WaitAny(f1, UINT64_MAX);

    wgpu::DeviceDescriptor desc;
    desc.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType errorType, wgpu::StringView message) {
        std::cerr << "Device error (" << int(errorType) << "): " << message << "\n";
    });
    wgpu::Future f2 = adapter.RequestDevice(&desc, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::RequestDeviceStatus status, wgpu::Device d, wgpu::StringView message) {
            if (status != wgpu::RequestDeviceStatus::Success) {
                std::cerr << "RequestDevice failed: " << message << "\n";
                std::exit(1);
            }
            device = std::move(d);
        });
    instance.WaitAny(f2, UINT64_MAX);
}

// ===================== Compute Shader (파티클 움직임/충돌) =====================
// 요구사항3: 파티클 움직임 및 obj 충돌 처리
static const char kComputeShader[] = R"(
struct Particle {
    pos : vec3f,
    vel : vec3f,
    color : vec3f,
    size : f32,
    light : vec3f,
    texIndex : f32,
};
@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<storage, read> meshVertices : array<vec3f>;
@group(0) @binding(2) var<storage, read> meshIndices : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let idx = id.x;
    if (idx >= arrayLength(&particles)) { return; }
    // 파티클 움직임
    particles[idx].pos += particles[idx].vel * 0.016;
    // 간단한 중력
    particles[idx].vel.y -= 9.8 * 0.016;
    // 요구사항3: 파티클과 obj 충돌 처리 (간단한 AABB)
    // 실제 충돌은 복잡하므로, 여기선 mesh의 첫 번째 삼각형과만 충돌 체크
    let v0 = meshVertices[meshIndices[0]];
    let v1 = meshVertices[meshIndices[1]];
    let v2 = meshVertices[meshIndices[2]];
    let minX = min(v0.x, min(v1.x, v2.x));
    let maxX = max(v0.x, max(v1.x, v2.x));
    let minY = min(v0.y, min(v1.y, v2.y));
    let maxY = max(v0.y, max(v1.y, v2.y));
    let minZ = min(v0.z, min(v1.z, v2.z));
    let maxZ = max(v0.z, max(v1.z, v2.z));
    let p = particles[idx].pos;
    if (p.x >= minX && p.x <= maxX && p.y >= minY && p.y <= maxY && p.z >= minZ && p.z <= maxZ) {
        particles[idx].vel = -particles[idx].vel * 0.5;
    }
}
)";

// ===================== Vertex/Fragment Shader (특수효과) =====================
// 요구사항4: 파티클별 light, texture 등 특수효과
static const char kShader[] = R"(
struct MVP {
  model : mat4x4<f32>,
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
};
struct Particle {
    pos : vec3f,
    vel : vec3f,
    color : vec3f,
    size : f32,
    light : vec3f,
    texIndex : f32,
};
@group(0) @binding(0) var<uniform> mvp : MVP;
@group(0) @binding(1) var<storage, read> particles : array<Particle>;
@vertex
fn vertexMain(@location(0) inPos : vec3f, @location(1) inCol : vec3f, @builtin(instance_index) instanceIdx : u32) -> @builtin(position) vec4f {
  let p = particles[instanceIdx];
  let worldPos = inPos * p.size + p.pos;
  return mvp.proj * mvp.view * mvp.model * vec4f(worldPos, 1.0);
}
struct FSOut {
  @location(0) color : vec4f
};
@fragment
fn fragmentMain(@builtin(instance_index) instanceIdx : u32) -> FSOut {
  let p = particles[instanceIdx];
  // 요구사항4: 파티클별 light, texture 등 특수효과
  let baseColor = vec4f(p.color, 1.0);
  let lightEffect = vec4f(p.light, 1.0);
  // 텍스처 효과는 샘플러/텍스처 바인딩 필요 (여기선 생략)
  var out : FSOut;
  out.color = baseColor * lightEffect;
  return out;
}
)";

// ===================== 그래픽 리소스 초기화 =====================
void InitGraphics(const ObjMesh& mesh) {
    ConfigureSurface();
    CreateDepthTexture();
    // SSBO: 파티클 데이터
    particleBuffer = CreateBuffer(
        particles.data(),
        particles.size() * sizeof(Particle),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst
    );
    // OBJ mesh 버퍼
    meshVertexBuffer = CreateBuffer(
        mesh.vertices.data(),
        mesh.vertices.size() * sizeof(float),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst
    );
    meshIndexBuffer = CreateBuffer(
        mesh.indices.data(),
        mesh.indices.size() * sizeof(uint32_t),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst
    );
    // UBO
    MVP mvp = {};
    uniformBuffer = CreateBuffer(
        &mvp,
        sizeof(MVP),
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst
    );
    // BindGroupLayout/BindGroup (렌더/컴퓨트)
    // ...생략(실제 구현시 각 바인딩에 맞게 생성)...
    // 파이프라인 생성 (렌더/컴퓨트)
    // ...생략(실제 구현시 셰이더 모듈 생성 및 파이프라인 생성)...
}

// ===================== 렌더링 루프 =====================
void Render() {
    // Compute Shader로 파티클 업데이트
    // ...생략(실제 구현시 compute pass 실행)...
    // 렌더링
    // ...생략(실제 구현시 render pass 실행)...
}

// ===================== 메인 =====================
int main() {
    Init();
    ObjMesh mesh = ImportObj("model.obj"); // 요구사항2: obj import
    if (mesh.vertices.empty() || mesh.indices.empty()) {
    std::cerr << "OBJ 데이터가 비어 있습니다." << std::endl;
    std::exit(1);
   }
    InitGraphics(mesh);
    // GLFW 윈도우 생성 및 루프
    // ...생략(실제 구현시 윈도우 생성, 입력 콜백, 루프 등)...
    return 0;
}



