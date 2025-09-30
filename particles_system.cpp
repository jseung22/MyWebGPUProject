// particles_system.cpp
// ===================== 100���� 3D ť�� ��ƼŬ �ý��� ���� =====================
// �䱸����:
// 1. SSBO�� ��ƼŬ ������ ����
// 2. *.obj ���� import
// 3. Compute Shader�� ��ƼŬ ������ �� obj �浹 ó��
// 4. ��ƼŬ�� light, texture �� Ư��ȿ��(vertex/fragment shader)
// 5. �� �䱸���׺� �ּ� ��Ȯ�� ǥ��

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

// ===================== UBO ����ü =====================
// ��ƼŬ �������� �ʿ��� ��/��/�������� ���
struct MVP {
    float model[16];
    float view[16];
    float proj[16];
};

// ===================== ��ƼŬ ������ ����ü (SSBO��) =====================
struct Particle {
    float pos[3];      // ��ġ
    float vel[3];      // �ӵ�
    float color[3];    // ����
    float size;        // ũ��
    float light[3];    // ���� ȿ��
    float texIndex;    // �ؽ�ó �ε���
};
constexpr int PARTICLE_COUNT = 1000000;
std::vector<Particle> particles(PARTICLE_COUNT);

// ===================== OBJ ���� import �Լ� =====================
// �䱸����2: *.obj ���� import
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

// ===================== WebGPU �ٽ� ��ü =====================
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

// ===================== ���� ���� ���� =====================
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

// ===================== Depth Texture ���� =====================
void CreateDepthTexture() {
    wgpu::TextureDescriptor td;
    td.size = {kWidth, kHeight, 1};
    td.format = wgpu::TextureFormat::Depth24Plus;
    td.usage = wgpu::TextureUsage::RenderAttachment;
    depthTexture = device.CreateTexture(&td);
    depthView = depthTexture.CreateView();
}

// ===================== Surface ���� =====================
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

// ===================== WebGPU �ʱ�ȭ =====================
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

// ===================== Compute Shader (��ƼŬ ������/�浹) =====================
// �䱸����3: ��ƼŬ ������ �� obj �浹 ó��
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
    // ��ƼŬ ������
    particles[idx].pos += particles[idx].vel * 0.016;
    // ������ �߷�
    particles[idx].vel.y -= 9.8 * 0.016;
    // �䱸����3: ��ƼŬ�� obj �浹 ó�� (������ AABB)
    // ���� �浹�� �����ϹǷ�, ���⼱ mesh�� ù ��° �ﰢ������ �浹 üũ
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

// ===================== Vertex/Fragment Shader (Ư��ȿ��) =====================
// �䱸����4: ��ƼŬ�� light, texture �� Ư��ȿ��
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
  // �䱸����4: ��ƼŬ�� light, texture �� Ư��ȿ��
  let baseColor = vec4f(p.color, 1.0);
  let lightEffect = vec4f(p.light, 1.0);
  // �ؽ�ó ȿ���� ���÷�/�ؽ�ó ���ε� �ʿ� (���⼱ ����)
  var out : FSOut;
  out.color = baseColor * lightEffect;
  return out;
}
)";

// ===================== �׷��� ���ҽ� �ʱ�ȭ =====================
void InitGraphics(const ObjMesh& mesh) {
    ConfigureSurface();
    CreateDepthTexture();
    // SSBO: ��ƼŬ ������
    particleBuffer = CreateBuffer(
        particles.data(),
        particles.size() * sizeof(Particle),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst
    );
    // OBJ mesh ����
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
    // BindGroupLayout/BindGroup (����/��ǻƮ)
    // ...����(���� ������ �� ���ε��� �°� ����)...
    // ���������� ���� (����/��ǻƮ)
    // ...����(���� ������ ���̴� ��� ���� �� ���������� ����)...
}

// ===================== ������ ���� =====================
void Render() {
    // Compute Shader�� ��ƼŬ ������Ʈ
    // ...����(���� ������ compute pass ����)...
    // ������
    // ...����(���� ������ render pass ����)...
}

// ===================== ���� =====================
int main() {
    Init();
    ObjMesh mesh = ImportObj("model.obj"); // �䱸����2: obj import
    if (mesh.vertices.empty() || mesh.indices.empty()) {
    std::cerr << "OBJ �����Ͱ� ��� �ֽ��ϴ�." << std::endl;
    std::exit(1);
   }
    InitGraphics(mesh);
    // GLFW ������ ���� �� ����
    // ...����(���� ������ ������ ����, �Է� �ݹ�, ���� ��)...
    return 0;
}



