// 3D 黑洞渲染器主逻辑

let scene, camera, renderer, controls;
let blackHoleMesh;
let uniforms;
let backgroundTexture;

const vertexShader = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = `
uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_background;
uniform vec3 u_cameraPos;
uniform mat4 u_cameraWorldMatrix;
uniform mat4 u_cameraProjectionMatrixInverse;
uniform float u_mass;
uniform float u_diskDensity;
uniform float u_doppler;

varying vec2 vUv;

// 常量
const float PI = 3.14159265359;
const int MAX_STEPS = 150;
const float STEP_SIZE = 0.1;

// 获取背景颜色
vec3 getBackground(vec3 dir) {
    vec2 bgUv = vec2(atan(dir.z, dir.x) / (2.0 * PI) + 0.5, asin(dir.y) / PI + 0.5);
    return texture2D(u_background, bgUv).rgb;
}

// 简单的伪随机噪声
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// 2D 噪声
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// 吸积盘密度函数
float getDiskDensity(vec3 p, float rs) {
    float r = length(p.xz);
    float inner = rs * 2.5;
    float outer = rs * 8.0;
    
    // 检查是否在圆盘范围内
    if (r < inner || r > outer) return 0.0;
    
    // 垂直厚度
    float thickness = 0.1 * (r - inner) / (outer - inner) + 0.05;
    if (abs(p.y) > thickness) return 0.0;
    
    // 密度分布
    float density = smoothstep(inner, inner + 1.0, r) * smoothstep(outer, outer - 2.0, r);
    
    // 添加一些纹理细节
    float angle = atan(p.z, p.x);
    vec2 uv = vec2(r * 2.0, angle * 5.0 - u_time * 2.0 * (inner / r));
    density *= noise(uv) * 0.5 + 0.5;
    
    // 垂直衰减
    density *= 1.0 - abs(p.y) / thickness;
    
    return density * u_diskDensity;
}

void main() {
    // 归一化设备坐标 (NDC)
    vec2 ndc = (vUv - 0.5) * 2.0;
    
    // 计算射线方向
    vec4 target = u_cameraProjectionMatrixInverse * vec4(ndc.x, ndc.y, 1.0, 1.0);
    vec3 rayDir = normalize((u_cameraWorldMatrix * vec4(normalize(target.xyz / target.w), 0.0)).xyz);
    vec3 rayPos = u_cameraPos;

    // 黑洞参数
    float rs = u_mass; // 史瓦西半径
    float rs2 = rs * rs;
    
    vec3 col = vec3(0.0);
    vec3 diskCol = vec3(0.0);
    float transmittance = 1.0;
    bool hitBlackHole = false;
    
    // 光线追踪循环
    for (int i = 0; i < MAX_STEPS; i++) {
        float r = length(rayPos);
        
        // 如果进入事件视界，光线被吸收
        if (r < rs) {
            hitBlackHole = true;
            break;
        }
        
        // 如果光线逃逸到足够远，采样背景
        if (r > 20.0) {
            break;
        }
        
        // 采样吸积盘
        float density = getDiskDensity(rayPos, rs);
        if (density > 0.0) {
            // 基础颜色：内侧偏蓝白，外侧偏红橙
            float rNorm = (length(rayPos.xz) - rs * 2.5) / (rs * 5.5);
            vec3 baseColor = mix(vec3(1.0, 0.9, 0.8), vec3(1.0, 0.4, 0.1), rNorm);
            
            // 计算多普勒效应
            // 假设吸积盘逆时针旋转
            vec3 velocityDir = normalize(vec3(-rayPos.z, 0.0, rayPos.x));
            // 速度大小与 1/sqrt(r) 成正比
            float velocityMag = sqrt(rs / length(rayPos.xz)) * 0.5; 
            vec3 velocity = velocityDir * velocityMag;
            
            // 观察者视线方向与速度的点积
            float dotProduct = dot(rayDir, velocity);
            
            // 多普勒因子 (简化版)
            float dopplerFactor = 1.0 + dotProduct * u_doppler * 2.0;
            
            // 亮度变化 (beaming)
            float beaming = pow(max(dopplerFactor, 0.1), 3.0);
            
            // 颜色偏移 (蓝移/红移)
            vec3 shiftColor = mix(vec3(1.0, 0.2, 0.1), vec3(0.5, 0.8, 1.0), smoothstep(0.5, 1.5, dopplerFactor));
            baseColor = mix(baseColor, baseColor * shiftColor, u_doppler);
            
            // 应用多普勒效应到亮度和颜色
            vec3 finalColor = baseColor * beaming;
            
            // 累加颜色和不透明度
            float alpha = density * STEP_SIZE * 5.0;
            diskCol += finalColor * density * transmittance * STEP_SIZE * 10.0;
            transmittance *= (1.0 - alpha);
            
            if (transmittance < 0.01) break;
        }
        
        // 计算引力导致的光线弯曲
        vec3 gravityDir = -normalize(rayPos);
        float gravityStrength = 1.5 * rs * rs2 / (r * r * r * r);
        
        // 更新射线方向和位置
        rayDir = normalize(rayDir + gravityDir * gravityStrength * STEP_SIZE);
        rayPos += rayDir * STEP_SIZE;
    }

    if (hitBlackHole) {
        col = diskCol; // 只有吸积盘的颜色
    } else {
        vec3 bgCol = getBackground(rayDir);
        col = diskCol + bgCol * transmittance;
    }

    // 简单的色调映射
    col = col / (1.0 + col);
    // Gamma 校正
    col = pow(col, vec3(1.0 / 2.2));

    gl_FragColor = vec4(col, 1.0);
}
`;

// 初始化场景
function init() {
    const container = document.getElementById('canvas-container');

    // 创建场景
    scene = new THREE.Scene();

    // 创建摄像机
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 2, 10);

    // 创建渲染器
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // 添加控制器
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2;
    controls.maxDistance = 50;

    // 创建星空背景
    createStarfield();

    // 设置着色器 uniforms
    uniforms = {
        u_time: { value: 0.0 },
        u_resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        u_background: { value: backgroundTexture },
        u_cameraPos: { value: camera.position },
        u_cameraWorldMatrix: { value: camera.matrixWorld },
        u_cameraProjectionMatrixInverse: { value: camera.projectionMatrixInverse },
        u_mass: { value: 1.0 },
        u_diskDensity: { value: 1.0 },
        u_doppler: { value: 1.0 }
    };

    // 创建全屏四边形
    const geometry = new THREE.PlaneGeometry(2, 2);
    const material = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        uniforms: uniforms,
        depthWrite: false,
        depthTest: false
    });
    blackHoleMesh = new THREE.Mesh(geometry, material);
    
    // 使用一个正交相机来渲染全屏四边形
    const orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const orthoScene = new THREE.Scene();
    orthoScene.add(blackHoleMesh);

    // 覆盖默认的 render 逻辑
    renderer.autoClear = false;
    const renderScene = function() {
        // 更新 uniforms
        uniforms.u_cameraPos.value.copy(camera.position);
        uniforms.u_cameraWorldMatrix.value.copy(camera.matrixWorld);
        uniforms.u_cameraProjectionMatrixInverse.value.copy(camera.projectionMatrixInverse);
        
        // 获取 UI 控制值
        uniforms.u_mass.value = parseFloat(document.getElementById('mass').value);
        uniforms.u_diskDensity.value = parseFloat(document.getElementById('diskDensity').value);
        uniforms.u_doppler.value = parseFloat(document.getElementById('doppler').value);

        renderer.clear();
        renderer.render(orthoScene, orthoCamera);
    };

    // 隐藏加载提示
    document.getElementById('loading').style.opacity = '0';

    // 监听窗口大小变化
    window.addEventListener('resize', onWindowResize, false);

    // 开始动画循环
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        
        if (uniforms) {
            uniforms.u_time.value += 0.01;
        }
        
        renderScene();
    }
    animate();
}

function createStarfield() {
    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');

    // 黑色背景
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 绘制星星
    for (let i = 0; i < 5000; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        const radius = Math.random() * 1.5;
        const intensity = Math.random();
        
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${intensity})`;
        ctx.fill();
    }

    // 绘制一些星云
    for (let i = 0; i < 20; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        const radius = Math.random() * 200 + 50;
        
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        const r = Math.floor(Math.random() * 100);
        const g = Math.floor(Math.random() * 100 + 50);
        const b = Math.floor(Math.random() * 150 + 100);
        
        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.1)`);
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.mapping = THREE.EquirectangularReflectionMapping;
    
    backgroundTexture = texture;
    // scene.background = texture;
    // scene.environment = texture;
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    if (uniforms) {
        uniforms.u_resolution.value.set(window.innerWidth, window.innerHeight);
    }
}

// 页面加载完成后初始化
window.onload = init;