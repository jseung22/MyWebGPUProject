//// ===================== WebGPU + GLFW: 1M Particles + Textured OBJ Model =====================
//// - Compute shader: Linear motion with boundary bouncing
//// - Render: 1M instanced cube particles + PBR textured OBJ model at center
//// - Camera: WASD move, mouse look, wheel zoom
//// - Textures: Albedo, Normal, Metallic, Roughness, AO
//
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <chrono>
//#include <cstring>
//#include <random>
//#include <fstream>
//#include <sstream>
//#include <GLFW/glfw3.h>
//#include <dawn/webgpu_cpp_print.h>
//#include <webgpu/webgpu_cpp.h>
//#include <webgpu/webgpu_glfw.h>
//
//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"
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
//// Particle system
//wgpu::Buffer particleCubeVertexBuffer, particleCubeIndexBuffer;
//wgpu::Buffer particleBuffer;
//wgpu::Buffer simUniform;
//wgpu::Buffer particleMvpUniform;
//wgpu::BindGroup simBG, particleRenderBG;
//wgpu::BindGroupLayout simBGL, particleRenderBGL;
//wgpu::RenderPipeline particleRenderPipeline;
//wgpu::ComputePipeline computePipeline;
//
//// OBJ model with textures
//wgpu::Buffer objVertexBuffer;
//wgpu::Buffer objIndexBuffer;
//wgpu::Buffer objMvpUniform;
//wgpu::Buffer objLightUniform;
//wgpu::Texture albedoTex, normalTex, metallicTex, roughnessTex, aoTex;
//wgpu::TextureView albedoView, normalView, metallicView, roughnessView, aoView;
//wgpu::Sampler textureSampler;
//wgpu::BindGroup objRenderBG;
//wgpu::BindGroupLayout objRenderBGL;
//wgpu::RenderPipeline objRenderPipeline;
//uint32_t objIndexCount = 0;
//
//static constexpr uint32_t WORKGROUP_SIZE = 256u;
//static constexpr uint32_t NUM_PARTICLES = 1'000'000u;
//
//// ============ Cube mesh for particles ============
//const float cubeVerts[] = {
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
//// ===================== Texture Loader =====================
//struct TextureData {
//    unsigned char* data;
//    int width, height, channels;
//};
//
//bool LoadTexture(const char* filename, TextureData& tex) {
//    tex.data = stbi_load(filename, &tex.width, &tex.height, &tex.channels, 4);
//    if (!tex.data) {
//        std::cerr << "Failed to load texture: " << filename << "\n";
//        return false;
//    }
//    std::cout << "Loaded texture: " << filename << " (" << tex.width << "x" << tex.height << ")\n";
//    return true;
//}
//
//wgpu::Texture CreateTextureFromData(const TextureData& texData) {
//    wgpu::TextureDescriptor texDesc{};
//    texDesc.size = { (uint32_t)texData.width, (uint32_t)texData.height, 1 };
//    texDesc.format = wgpu::TextureFormat::RGBA8Unorm;
//    texDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
//    texDesc.mipLevelCount = 1;
//    auto texture = device.CreateTexture(&texDesc);
//
//    // Use WriteTexture similar to how WriteBuffer is used in this codebase
//    // Based on the Frame() function pattern: device.GetQueue().WriteBuffer(buffer, offset, data, size)
//    // For textures, we need to provide texture destination info
//    
//    wgpu::TexelCopyTextureInfo destination{};
//    destination.texture = texture;
//    destination.mipLevel = 0;
//    destination.origin = { 0, 0, 0 };
//    destination.aspect = wgpu::TextureAspect::All;
//
//    // Create texture data layout
//    wgpu::TexelCopyBufferInfo source{};
//    source.layout.bytesPerRow = texData.width * 4;
//    source.layout.rowsPerImage = texData.height;
//
//    wgpu::Extent3D copySize = { (uint32_t)texData.width, (uint32_t)texData.height, 1 };
//
//    // Write texture data
//    device.GetQueue().WriteTexture(&destination, texData.data, 
//                                   texData.width * texData.height * 4, 
//                                   &source.layout, &copySize);
//
//    std::cout << "Successfully loaded texture: " << texData.width << "x" << texData.height << "\n";
//    return texture;
//}
//
//// ===================== OBJ Loader =====================
//struct OBJMesh {
//    std::vector<float> vertices;  // interleaved: x,y,z, nx,ny,nz, u,v
//    std::vector<uint32_t> indices;
//};
//
//bool LoadOBJ(const char* filename, OBJMesh& mesh) {
//    std::ifstream file(filename);
//    if (!file.is_open()) {
//        std::cerr << "Failed to open OBJ file: " << filename << "\n";
//        return false;
//    }
//
//    std::vector<float> positions;
//    std::vector<float> normals;
//    std::vector<float> texcoords;
//    std::vector<uint32_t> posIndices, normIndices, uvIndices;
//
//    std::string line;
//    while (std::getline(file, line)) {
//        std::istringstream iss(line);
//        std::string prefix;
//        iss >> prefix;
//
//        if (prefix == "v") {
//            float x, y, z;
//            iss >> x >> y >> z;
//            positions.push_back(x);
//            positions.push_back(y);
//            positions.push_back(z);
//        }
//        else if (prefix == "vn") {
//            float nx, ny, nz;
//            iss >> nx >> ny >> nz;
//            normals.push_back(nx);
//            normals.push_back(ny);
//            normals.push_back(nz);
//        }
//        else if (prefix == "vt") {
//            float u, v;
//            iss >> u >> v;
//            texcoords.push_back(u);
//            texcoords.push_back(v);
//        }
//        else if (prefix == "f") {
//            std::string vertex;
//            std::vector<uint32_t> facePos, faceNorm, faceUV;
//            
//            while (iss >> vertex) {
//                std::istringstream viss(vertex);
//                std::string indexStr;
//                int idx = 0;
//                uint32_t posIdx = 0, uvIdx = 0, normIdx = 0;
//                
//                while (std::getline(viss, indexStr, '/')) {
//                    if (!indexStr.empty()) {
//                        if (idx == 0) posIdx = std::stoi(indexStr) - 1;
//                        else if (idx == 1) uvIdx = std::stoi(indexStr) - 1;
//                        else if (idx == 2) normIdx = std::stoi(indexStr) - 1;
//                    }
//                    idx++;
//                }
//                facePos.push_back(posIdx);
//                faceUV.push_back(uvIdx);
//                if (normIdx < normals.size() / 3) faceNorm.push_back(normIdx);
//            }
//
//            for (size_t i = 1; i + 1 < facePos.size(); ++i) {
//                posIndices.push_back(facePos[0]);
//                posIndices.push_back(facePos[i]);
//                posIndices.push_back(facePos[i + 1]);
//                
//                uvIndices.push_back(faceUV[0]);
//                uvIndices.push_back(faceUV[i]);
//                uvIndices.push_back(faceUV[i + 1]);
//                
//                if (!faceNorm.empty() && faceNorm.size() == facePos.size()) {
//                    normIndices.push_back(faceNorm[0]);
//                    normIndices.push_back(faceNorm[i]);
//                    normIndices.push_back(faceNorm[i + 1]);
//                }
//            }
//        }
//    }
//
//    if (normIndices.empty()) normIndices = posIndices;
//    if (uvIndices.empty()) {
//        for (size_t i = 0; i < posIndices.size(); ++i) uvIndices.push_back(0);
//    }
//    
//    for (size_t i = 0; i < posIndices.size(); ++i) {
//        uint32_t pi = posIndices[i];
//        uint32_t ni = (i < normIndices.size()) ? normIndices[i] : pi;
//        uint32_t ui = (i < uvIndices.size()) ? uvIndices[i] : 0;
//        
//        mesh.vertices.push_back(positions[pi * 3 + 0]);
//        mesh.vertices.push_back(positions[pi * 3 + 1]);
//        mesh.vertices.push_back(positions[pi * 3 + 2]);
//        
//        if (ni * 3 + 2 < normals.size()) {
//            mesh.vertices.push_back(normals[ni * 3 + 0]);
//            mesh.vertices.push_back(normals[ni * 3 + 1]);
//            mesh.vertices.push_back(normals[ni * 3 + 2]);
//        } else {
//            mesh.vertices.push_back(0.0f);
//            mesh.vertices.push_back(1.0f);
//            mesh.vertices.push_back(0.0f);
//        }
//        
//        if (ui * 2 + 1 < texcoords.size()) {
//            mesh.vertices.push_back(texcoords[ui * 2 + 0]);
//            mesh.vertices.push_back(texcoords[ui * 2 + 1]);
//        } else {
//            mesh.vertices.push_back(0.0f);
//            mesh.vertices.push_back(0.0f);
//        }
//        
//        mesh.indices.push_back(static_cast<uint32_t>(i));
//    }
//
//    std::cout << "Loaded OBJ: " << mesh.indices.size() / 3 << " triangles\n";
//    return true;
//}
//
//// ===================== WGSL Shaders =====================
//// Particle compute shader (unchanged)
//static const char kComputeWGSL[] = R"(
//struct Particle {
//  pos   : vec4<f32>,
//  vel   : vec4<f32>,
//  color : vec4<f32>,
//  misc  : vec4<f32>,
//};
//
//struct Sim {
//  dt_time  : vec2<f32>,
//  _pad0    : vec2<f32>,
//  bounds   : vec4<f32>,
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
//  let dt = sim.dt_time.x;
//  let B = sim.bounds.xyz;
//
//  var pos = p.pos.xyz + p.vel.xyz * dt;
//  var vel = p.vel.xyz;
//
//  if (pos.x > B.x) { pos.x = B.x; vel.x = -vel.x; }
//  if (pos.x < -B.x) { pos.x = -B.x; vel.x = -vel.x; }
//  if (pos.y > B.y) { pos.y = B.y; vel.y = -vel.y; }
//  if (pos.y < -B.y) { pos.y = -B.y; vel.y = -vel.y; }
//  if (pos.z > B.z) { pos.z = B.z; vel.z = -vel.z; }
//  if (pos.z < -B.z) { pos.z = -B.z; vel.z = -vel.z; }
//
//  p.misc.x += dt;
//  p.pos = vec4<f32>(pos, p.pos.w);
//  p.vel = vec4<f32>(vel, 0.0);
//  particles[i] = p;
//}
//)";
//
//// Particle render shader (unchanged)
//static const char kParticleRenderWGSL[] = R"(
//struct Particle {
//  pos   : vec4<f32>,
//  vel   : vec4<f32>,
//  color : vec4<f32>,
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
//// PBR OBJ render shader with textures and UV debugging
//static const char kObjRenderWGSL[] = R"(
//struct MVP {
//  model : mat4x4<f32>,
//  view  : mat4x4<f32>,
//  proj  : mat4x4<f32>,
//};
//
//struct Light {
//  viewPos : vec3<f32>,
//  _pad0   : f32,
//  lightPos : vec3<f32>,
//  _pad1   : f32,
//  lightColor : vec3<f32>,
//  _pad2   : f32,
//};
//
//@group(0) @binding(0) var<uniform> mvp : MVP;
//@group(0) @binding(1) var<uniform> light : Light;
//@group(0) @binding(2) var albedoTex : texture_2d<f32>;
//@group(0) @binding(3) var normalTex : texture_2d<f32>;
//@group(0) @binding(4) var metallicTex : texture_2d<f32>;
//@group(0) @binding(5) var roughnessTex : texture_2d<f32>;
//@group(0) @binding(6) var aoTex : texture_2d<f32>;
//@group(0) @binding(7) var texSampler : sampler;
//
//struct VSOut {
//  @builtin(position) pos : vec4<f32>,
//  @location(0) worldPos : vec3<f32>,
//  @location(1) normal : vec3<f32>,
//  @location(2) uv : vec2<f32>
//};
//
//@vertex
//fn vsMain(@location(0) inPos : vec3<f32>, @location(1) inNormal : vec3<f32>, @location(2) inUV : vec2<f32>) -> VSOut {
//  var out : VSOut;
//  let worldPos = mvp.model * vec4<f32>(inPos, 1.0);
//  out.worldPos = worldPos.xyz;
//  out.pos = mvp.proj * mvp.view * worldPos;
//  out.normal = normalize((mvp.model * vec4<f32>(inNormal, 0.0)).xyz);
//  out.uv = inUV;
//  return out;
//}
//
//const PI = 3.14159265359;
//
//fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
//  let a = roughness * roughness;
//  let a2 = a * a;
//  let NdotH = max(dot(N, H), 0.0);
//  let NdotH2 = NdotH * NdotH;
//  let nom = a2;
//  var denom = (NdotH2 * (a2 - 1.0) + 1.0);
//  denom = PI * denom * denom;
//  return nom / denom;
//}
//
//fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
//  let r = (roughness + 1.0);
//  let k = (r * r) / 8.0;
//  let nom = NdotV;
//  let denom = NdotV * (1.0 - k) + k;
//  return nom / denom;
//}
//
//fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
//  let NdotV = max(dot(N, V), 0.0);
//  let NdotL = max(dot(N, L), 0.0);
//  let ggx2 = GeometrySchlickGGX(NdotV, roughness);
//  let ggx1 = GeometrySchlickGGX(NdotL, roughness);
//  return ggx1 * ggx2;
//}
//
//fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
//  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
//}
//
//@fragment
//fn fsMain(@location(0) worldPos : vec3<f32>, @location(1) normal : vec3<f32>, @location(2) uv : vec2<f32>) -> @location(0) vec4<f32> {
//  // Uncomment this line to debug UV coordinates (shows UV as RGB colors)
//  // return vec4<f32>(uv, 0.0, 1.0);
//  
//  let albedo = textureSample(albedoTex, texSampler, uv).rgb;
//  let normalMap = textureSample(normalTex, texSampler, uv).rgb;
//  let metallic = textureSample(metallicTex, texSampler, uv).r;
//  let roughness = textureSample(roughnessTex, texSampler, uv).r;
//  let ao = textureSample(aoTex, texSampler, uv).r;
//  
//  // Properly apply normal mapping
//  let tangentNormal = normalMap * 2.0 - 1.0; // Convert from [0,1] to [-1,1]
//  
//  // For now, use the vertex normal (we'd need tangent space for proper normal mapping)
//  let N = normalize(normal);
//  let V = normalize(light.viewPos - worldPos);
//  
//  var F0 = vec3<f32>(0.04);
//  F0 = mix(F0, albedo, metallic);
//  
//  var Lo = vec3<f32>(0.0);
//  
//  let L = normalize(light.lightPos - worldPos);
//  let H = normalize(V + L);
//  let distance = length(light.lightPos - worldPos);
//  let attenuation = 1.0 / (distance * distance);
//  let radiance = light.lightColor * attenuation;
//  
//  let NDF = DistributionGGX(N, H, roughness);
//  let G = GeometrySmith(N, V, L, roughness);
//  let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
//  
//  let numerator = NDF * G * F;
//  let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
//  let specular = numerator / denominator;
//  
//  let kS = F;
//  var kD = vec3<f32>(1.0) - kS;
//  kD *= 1.0 - metallic;
//  
//  let NdotL = max(dot(N, L), 0.0);
//  Lo += (kD * albedo / PI + specular) * radiance * NdotL;
//  
//  let ambient = vec3<f32>(0.03) * albedo * ao;
//  var color = ambient + Lo;
//  
//  color = color / (color + vec3<f32>(1.0));
//  color = pow(color, vec3<f32>(1.0 / 2.2));
//  
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
//
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
//// ===================== Particle System =====================
//struct CPU_Particle {
//    float pos[4];
//    float vel[4];
//    float color[4];
//    float misc[4];
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
//        p.pos[0] = U(rng) * halfX * 0.8f;
//        p.pos[1] = U(rng) * halfY * 0.8f;
//        p.pos[2] = U(rng) * halfZ * 0.8f;
//        p.pos[3] = 0.05f;
//
//        float theta = U01(rng) * 6.28318f;
//        float phi = U01(rng) * 3.14159f;
//        float speed = 5.0f + U01(rng) * 15.0f;
//        
//        p.vel[0] = speed * std::sin(phi) * std::cos(theta);
//        p.vel[1] = speed * std::sin(phi) * std::sin(theta);
//        p.vel[2] = speed * std::cos(phi);
//        p.vel[3] = 0.f;
//
//        p.color[0] = 0.2f + U01(rng) * 0.8f;
//        p.color[1] = 0.2f + U01(rng) * 0.8f;
//        p.color[2] = 0.2f + U01(rng) * 0.8f;
//        p.color[3] = 1.0f;
//
//        p.misc[0] = 0.0f;
//        p.misc[1] = p.misc[2] = p.misc[3] = 0.f;
//    }
//}
//
//struct MVP { float model[16], view[16], proj[16]; };
//struct SimData {
//    float dt, time, _p0, _p1;
//    float bounds[4];
//    float _pad1[4];
//};
//struct LightData {
//    float viewPos[3], _p0;
//    float lightPos[3], _p1;
//    float lightColor[3], _p2;
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
//    dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType t, wgpu:: StringView m) {
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
//void CreatePipelinesAndResources(const char* objFilePath, const char* textureFolder) {
//    ConfigureSurface();
//    CreateDepth();
//
//    // ===== Particle System =====
//    particleCubeVertexBuffer = CreateBuffer(cubeVerts, sizeof(cubeVerts), wgpu::BufferUsage::Vertex);
//    particleCubeIndexBuffer = CreateBuffer(cubeIdx, sizeof(cubeIdx), wgpu::BufferUsage::Index);
//
//    std::vector<CPU_Particle> init;
//    const float halfX = 120.f, halfY = 80.f, halfZ = 120.f;
//    InitParticles(init, halfX, halfY, halfZ);
//    particleBuffer = CreateBuffer(init.data(), init.size() * sizeof(CPU_Particle),
//        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
//
//    simUniform = CreateZeroedBuffer(sizeof(SimData), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//    particleMvpUniform = CreateZeroedBuffer(sizeof(MVP), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//
//    // Compute pipeline
//    wgpu::BindGroupLayoutEntry cEntries[2]{};
//    cEntries[0].binding = 0; cEntries[0].visibility = wgpu::ShaderStage::Compute; cEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
//    cEntries[1].binding = 1; cEntries[1].visibility = wgpu::ShaderStage::Compute; cEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
//    wgpu::BindGroupLayoutDescriptor cBGLd{}; cBGLd.entryCount = 2; cBGLd.entries = cEntries;
//    simBGL = device.CreateBindGroupLayout(&cBGLd);
//    
//    wgpu::BindGroupEntry cBGEs[2]{};
//    cBGEs[0].binding = 0; cBGEs[0].buffer = particleBuffer; cBGEs[0].size = wgpu::kWholeSize;
//    cBGEs[1].binding = 1; cBGEs[1].buffer = simUniform; cBGEs[1].size = sizeof(SimData);
//    wgpu::BindGroupDescriptor cBGd{}; cBGd.layout = simBGL; cBGd.entryCount = 2; cBGd.entries = cBGEs;
//    simBG = device.CreateBindGroup(&cBGd);
//
//    wgpu::ShaderSourceWGSL cWGSL; cWGSL.code = kComputeWGSL;
//    wgpu::ShaderModuleDescriptor cSMD{}; cSMD.nextInChain = &cWGSL;
//    auto cSM = device.CreateShaderModule(&cSMD);
//
//    wgpu::PipelineLayoutDescriptor cPLd{}; cPLd.bindGroupLayoutCount = 1; cPLd.bindGroupLayouts = &simBGL;
//    auto cPL = device.CreatePipelineLayout(&cPLd);
//
//    wgpu::ComputePipelineDescriptor cPD{}; cPD.layout = cPL; cPD.compute.module = cSM; cPD.compute.entryPoint = "csMain";
//    computePipeline = device.CreateComputePipeline(&cPD);
//
//    // Particle render pipeline
//    wgpu::BindGroupLayoutEntry pEntries[2]{};
//    pEntries[0].binding = 0; pEntries[0].visibility = wgpu::ShaderStage::Vertex; pEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
//    pEntries[1].binding = 1; pEntries[1].visibility = wgpu::ShaderStage::Vertex; pEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
//    wgpu::BindGroupLayoutDescriptor pBGLd{}; pBGLd.entryCount = 2; pBGLd.entries = pEntries;
//    particleRenderBGL = device.CreateBindGroupLayout(&pBGLd);
//    
//    wgpu::BindGroupEntry pBGEs[2]{};
//    pBGEs[0].binding = 0; pBGEs[0].buffer = particleBuffer; pBGEs[0].size = wgpu::kWholeSize;
//    pBGEs[1].binding = 1; pBGEs[1].buffer = particleMvpUniform; pBGEs[1].size = sizeof(MVP);
//    wgpu::BindGroupDescriptor pBGDescriptor{}; pBGDescriptor.layout = particleRenderBGL; pBGDescriptor.entryCount = 2; pBGDescriptor.entries = pBGEs;
//    particleRenderBG = device.CreateBindGroup(&pBGDescriptor);
//
//    wgpu::ShaderSourceWGSL pWGSL; pWGSL.code = kParticleRenderWGSL;
//    wgpu::ShaderModuleDescriptor pSMD{}; pSMD.nextInChain = &pWGSL;
//    auto pSM = device.CreateShaderModule(&pSMD);
//
//    wgpu::VertexAttribute pAttr{}; pAttr.format = wgpu::VertexFormat::Float32x3; pAttr.shaderLocation = 0;
//    wgpu::VertexBufferLayout pVBL{}; pVBL.arrayStride = 12; pVBL.attributeCount = 1; pVBL.attributes = &pAttr;
//
//    wgpu::ColorTargetState pCTS{}; pCTS.format = colorFormat;
//    wgpu::FragmentState pFS{}; pFS.module = pSM; pFS.entryPoint = "fsMain"; pFS.targetCount = 1; pFS.targets = &pCTS;
//    wgpu::DepthStencilState pDS{}; pDS.format = wgpu::TextureFormat::Depth24Plus; pDS.depthWriteEnabled = true; pDS.depthCompare = wgpu::CompareFunction::Less;
//
//    wgpu::PipelineLayoutDescriptor pPLd{}; pPLd.bindGroupLayoutCount = 1; pPLd.bindGroupLayouts = &particleRenderBGL;
//    auto pPL = device.CreatePipelineLayout(&pPLd);
//
//    wgpu::RenderPipelineDescriptor pRPD{};
//    pRPD.layout = pPL;
//    pRPD.vertex.module = pSM; pRPD.vertex.entryPoint = "vsMain"; pRPD.vertex.bufferCount = 1; pRPD.vertex.buffers = &pVBL;
//    pRPD.fragment = &pFS;
//    pRPD.depthStencil = &pDS;
//    particleRenderPipeline = device.CreateRenderPipeline(&pRPD);
//
//    // ===== Load Textures =====
//    std::string basePath = std::string(textureFolder) + "/";
//    TextureData albedoData, normalData, metallicData, roughnessData, aoData;
//    
//    if (!LoadTexture((basePath + "floatplane_albedo.png").c_str(), albedoData)) std::exit(1);
//    if (!LoadTexture((basePath + "floatplane_normal.png").c_str(), normalData)) std::exit(1);
//    if (!LoadTexture((basePath + "floatplane_metallic.png").c_str(), metallicData)) std::exit(1);
//    if (!LoadTexture((basePath + "floatplane_roughness.png").c_str(), roughnessData)) std::exit(1);
//    if (!LoadTexture((basePath + "floatplane_ao.png").c_str(), aoData)) std::exit(1);
//    
//    albedoTex = CreateTextureFromData(albedoData);
//    normalTex = CreateTextureFromData(normalData);
//    metallicTex = CreateTextureFromData(metallicData);
//    roughnessTex = CreateTextureFromData(roughnessData);
//    aoTex = CreateTextureFromData(aoData);
//    
//    albedoView = albedoTex.CreateView();
//    normalView = normalTex.CreateView();
//    metallicView = metallicTex.CreateView();
//    roughnessView = roughnessTex.CreateView();
//    aoView = aoTex.CreateView();
//    
//    stbi_image_free(albedoData.data);
//    stbi_image_free(normalData.data);
//    stbi_image_free(metallicData.data);
//    stbi_image_free(roughnessData.data);
//    stbi_image_free(aoData.data);
//    
//    // Create sampler with better settings for texture mapping
//    wgpu::SamplerDescriptor samplerDesc{};
//    samplerDesc.addressModeU = wgpu::AddressMode::Repeat;
//    samplerDesc.addressModeV = wgpu::AddressMode::Repeat;
//    samplerDesc.addressModeW = wgpu::AddressMode::Repeat;
//    samplerDesc.magFilter = wgpu::FilterMode::Linear;
//    samplerDesc.minFilter = wgpu::FilterMode::Linear;
//    samplerDesc.mipmapFilter = wgpu::MipmapFilterMode::Linear;
//    samplerDesc.maxAnisotropy = 16; // Add anisotropic filtering for better quality
//    textureSampler = device.CreateSampler(&samplerDesc);
//
//    // ===== OBJ Model =====
//    OBJMesh objMesh;
//    if (!LoadOBJ(objFilePath, objMesh)) {
//        std::cerr << "Failed to load OBJ\n";
//        std::exit(1);
//    }
//
//    objIndexCount = static_cast<uint32_t>(objMesh.indices.size());
//    objVertexBuffer = CreateBuffer(objMesh.vertices.data(), objMesh.vertices.size() * sizeof(float), wgpu::BufferUsage::Vertex);
//    objIndexBuffer = CreateBuffer(objMesh.indices.data(), objMesh.indices.size() * sizeof(uint32_t), wgpu::BufferUsage::Index);
//    objMvpUniform = CreateZeroedBuffer(sizeof(MVP), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//    objLightUniform = CreateZeroedBuffer(sizeof(LightData), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
//
//    // OBJ render pipeline with textures
//    wgpu::BindGroupLayoutEntry oEntries[8]{};
//    oEntries[0].binding = 0; oEntries[0].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment; oEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
//    oEntries[1].binding = 1; oEntries[1].visibility = wgpu::ShaderStage::Fragment; oEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
//    oEntries[2].binding = 2; oEntries[2].visibility = wgpu::ShaderStage::Fragment; oEntries[2].texture.sampleType = wgpu::TextureSampleType::Float;
//    oEntries[3].binding = 3; oEntries[3].visibility = wgpu::ShaderStage::Fragment; oEntries[3].texture.sampleType = wgpu::TextureSampleType::Float;
//    oEntries[4].binding = 4; oEntries[4].visibility = wgpu::ShaderStage::Fragment; oEntries[4].texture.sampleType = wgpu::TextureSampleType::Float;
//    oEntries[5].binding = 5; oEntries[5].visibility = wgpu::ShaderStage::Fragment; oEntries[5].texture.sampleType = wgpu::TextureSampleType::Float;
//    oEntries[6].binding = 6; oEntries[6].visibility = wgpu::ShaderStage::Fragment; oEntries[6].texture.sampleType = wgpu::TextureSampleType::Float;
//    oEntries[7].binding = 7; oEntries[7].visibility = wgpu::ShaderStage::Fragment; oEntries[7].sampler.type = wgpu::SamplerBindingType::Filtering;
//    
//    wgpu::BindGroupLayoutDescriptor oBGLd{}; oBGLd.entryCount = 8; oBGLd.entries = oEntries;
//    objRenderBGL = device.CreateBindGroupLayout(&oBGLd);
//    
//    wgpu::BindGroupEntry oBGEs[8]{};
//    oBGEs[0].binding = 0; oBGEs[0].buffer = objMvpUniform; oBGEs[0].size = sizeof(MVP);
//    oBGEs[1].binding = 1; oBGEs[1].buffer = objLightUniform; oBGEs[1].size = sizeof(LightData);
//    oBGEs[2].binding = 2; oBGEs[2].textureView = albedoView;
//    oBGEs[3].binding = 3; oBGEs[3].textureView = normalView;
//    oBGEs[4].binding = 4; oBGEs[4].textureView = metallicView;
//    oBGEs[5].binding = 5; oBGEs[5].textureView = roughnessView;
//    oBGEs[6].binding = 6; oBGEs[6].textureView = aoView;
//    oBGEs[7].binding = 7; oBGEs[7].sampler = textureSampler;
//    
//    wgpu::BindGroupDescriptor oBGd{}; oBGd.layout = objRenderBGL; oBGd.entryCount = 8; oBGd.entries = oBGEs;
//    objRenderBG = device.CreateBindGroup(&oBGd);
//
//    wgpu::ShaderSourceWGSL oWGSL; oWGSL.code = kObjRenderWGSL;
//    wgpu::ShaderModuleDescriptor oSMD{}; oSMD.nextInChain = &oWGSL;
//    auto oSM = device.CreateShaderModule(&oSMD);
//
//    wgpu::VertexAttribute oAttrs[3]{};
//    oAttrs[0].format = wgpu::VertexFormat::Float32x3; oAttrs[0].offset = 0; oAttrs[0].shaderLocation = 0;
//    oAttrs[1].format = wgpu::VertexFormat::Float32x3; oAttrs[1].offset = 12; oAttrs[1].shaderLocation = 1;
//    oAttrs[2].format = wgpu::VertexFormat::Float32x2; oAttrs[2].offset = 24; oAttrs[2].shaderLocation = 2;
//    wgpu::VertexBufferLayout oVBL{}; oVBL.arrayStride = 32; oVBL.attributeCount = 3; oVBL.attributes = oAttrs;
//
//    wgpu::ColorTargetState oCTS{}; oCTS.format = colorFormat;
//    wgpu::FragmentState oFS{}; oFS.module = oSM; oFS.entryPoint = "fsMain"; oFS.targetCount = 1; oFS.targets = &oCTS;
//    wgpu::DepthStencilState oDS{}; oDS.format = wgpu::TextureFormat::Depth24Plus; oDS.depthWriteEnabled = true; oDS.depthCompare = wgpu::CompareFunction::Less;
//
//    wgpu::PipelineLayoutDescriptor oPLd{}; oPLd.bindGroupLayoutCount = 1; oPLd.bindGroupLayouts = &objRenderBGL;
//    auto oPL = device.CreatePipelineLayout(&oPLd);
//
//    wgpu::RenderPipelineDescriptor oRPD{};
//    oRPD.layout = oPL;
//    oRPD.vertex.module = oSM; oRPD.vertex.entryPoint = "vsMain"; oRPD.vertex.bufferCount = 1; oRPD.vertex.buffers = &oVBL;
//    oRPD.fragment = &oFS;
//    oRPD.depthStencil = &oDS;
//    objRenderPipeline = device.CreateRenderPipeline(&oRPD);
//
//    std::cout << "All pipelines and textures loaded successfully\n";
//}
//
//// ===================== Frame Update =====================
//void Frame() {
//    float now = (float)glfwGetTime();
//    deltaTime = now - lastFrame; lastFrame = now; globalTime += deltaTime;
//
//    for (int i = 0; i < 1024; ++i) if (keys[i]) cam.ProcessKeyboard(i, deltaTime);
//
//    // Update uniforms
//    SimData sd{};
//    sd.dt = std::min(deltaTime, 1.f / 30.f);
//    sd.time = globalTime;
//    sd.bounds[0] = 120.f; sd.bounds[1] = 80.f; sd.bounds[2] = 120.f;
//    device.GetQueue().WriteBuffer(simUniform, 0, &sd, sizeof(sd));
//
//    MVP mvp{};
//    Mat4Identity(mvp.model);
//    float center[3] = { cam.pos[0] + cam.front[0], cam.pos[1] + cam.front[1], cam.pos[2] + cam.front[2] };
//    Mat4LookAt(mvp.view, cam.pos, center, cam.up);
//    Mat4Perspective(mvp.proj, cam.fov, (float)kWidth / (float)kHeight, 0.1f, 2000.f);
//    device.GetQueue().WriteBuffer(particleMvpUniform, 0, &mvp, sizeof(mvp));
//    device.GetQueue().WriteBuffer(objMvpUniform, 0, &mvp, sizeof(mvp));
//    
//    LightData light{};
//    light.viewPos[0] = cam.pos[0]; light.viewPos[1] = cam.pos[1]; light.viewPos[2] = cam.pos[2];
//    light.lightPos[0] = 100.f; light.lightPos[1] = 100.f; light.lightPos[2] = 100.f;
//    light.lightColor[0] = 1.0f; light.lightColor[1] = 1.0f; light.lightColor[2] = 1.0f;
//    device.GetQueue().WriteBuffer(objLightUniform, 0, &light, sizeof(light));
//
//    // Render
//    wgpu::SurfaceTexture st{}; surface.GetCurrentTexture(&st);
//    auto backView = st.texture.CreateView();
//    auto encoder = device.CreateCommandEncoder();
//
//    // Compute pass
//    {
//        auto cpass = encoder.BeginComputePass();
//        cpass.SetPipeline(computePipeline);
//        cpass.SetBindGroup(0, simBG);
//        uint32_t groups = (NUM_PARTICLES + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
//        cpass.DispatchWorkgroups(groups);
//        cpass.End();
//    }
//
//    // Render pass
//    wgpu::RenderPassColorAttachment ca{};
//    ca.view = backView; ca.loadOp = wgpu::LoadOp::Clear; ca.storeOp = wgpu::StoreOp::Store;
//    ca.clearValue = { 0.06,0.07,0.10,1.0 };
//
//    wgpu::RenderPassDepthStencilAttachment da{};
//    da.view = depthView; da.depthLoadOp = wgpu::LoadOp::Clear; da.depthClearValue = 1.0f; da.depthStoreOp = wgpu::StoreOp::Store;
//
//    wgpu::RenderPassDescriptor rpd{}; rpd.colorAttachmentCount = 1; rpd.colorAttachments = &ca; rpd.depthStencilAttachment = &da;
//
//    {
//        auto rpass = encoder.BeginRenderPass(&rpd);
//        
//        // Draw particles
//        rpass.SetPipeline(particleRenderPipeline);
//        rpass.SetBindGroup(0, particleRenderBG);
//        rpass.SetVertexBuffer(0, particleCubeVertexBuffer);
//        rpass.SetIndexBuffer(particleCubeIndexBuffer, wgpu::IndexFormat::Uint32);
//        rpass.DrawIndexed(kCubeIndexCount, NUM_PARTICLES, 0, 0, 0);
//        
//        // Draw textured OBJ model
//        rpass.SetPipeline(objRenderPipeline);
//        rpass.SetBindGroup(0, objRenderBG);
//        rpass.SetVertexBuffer(0, objVertexBuffer);
//        rpass.SetIndexBuffer(objIndexBuffer, wgpu::IndexFormat::Uint32);
//        rpass.DrawIndexed(objIndexCount, 1, 0, 0, 0);
//        
//        rpass.End();
//    }
//
//    auto cmd = encoder.Finish();
//    device.GetQueue().Submit(1, &cmd);
//    surface.Present();
//}
//
//// ===================== Main =====================
//int main(int argc, char** argv) {
//    InitWebGPU();
//
//    const char* objPath = (argc > 1) ? argv[1] : "floatplane/floatplane.obj";
//    const char* texturePath = (argc > 2) ? argv[2] : "floatplane/textures";
//
//    if (!glfwInit()) return -1;
//    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "1M Particles + Textured OBJ (PBR)", nullptr, nullptr);
//
//    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
//    CreatePipelinesAndResources(objPath, texturePath);
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
