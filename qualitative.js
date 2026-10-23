import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21];
const GT_BODY = "#f3ddcc";
const PRED_BODY = "#d5e3f3";
const GT_TRAIL = "#8d98a3";
const PRED_TRAIL = "#7f97b4";
const TRAIL_WIDTH = 0.048;
const homeOffset = new THREE.Vector3();

const root = document.getElementById("qualitative");
const canvas = document.getElementById("qual-canvas");
const statusEl = document.getElementById("qual-status");
const strip = document.getElementById("qual-strip");
const playButton = document.getElementById("qual-play");
const seek = document.getElementById("qual-seek");

const rotations = Array.from({ length: 24 }, () => new Float32Array(9));
const matA = new Float32Array(9);
const matB = new Float32Array(9);
const matT = new Float32Array(9);
const posedJoints = new Float32Array(72);
const center = new THREE.Vector3();

let smpl = null;
let renderer;
let scene;
let camera;
let controls;
let keyLight;
let fillLight;
let floor;
let gtActor;
let predActor;
let trails = [];
let clips = [];
let clip = null;
let frame = 0;
let playing = false;
let onScreen = true;
let accumulator = 0;
let lastTime = 0;

const PLAY_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l11-6.5z" fill="currentColor"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.2v14H7zm6.8 0H17v14h-3.2z" fill="currentColor"/></svg>';

/**
 * 读取 viewer 使用的紧凑 SMPL 网格。
 * @param {ArrayBuffer} buffer
 */
function parseSmpl(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== "SMPL") throw new Error("SMPL mesh is invalid");
  const vertexCount = view.getUint32(4, true);
  const faceCount = view.getUint32(8, true);
  let offset = 12;
  const restVertices = new Float32Array(buffer, offset, vertexCount * 3);
  offset += vertexCount * 12;
  const faces = new Uint16Array(buffer, offset, faceCount * 3);
  offset += faceCount * 6;
  const influences = new Uint8Array(buffer, offset, vertexCount * 4);
  offset += vertexCount * 4;
  const influenceWeights = new Float32Array(buffer, offset, vertexCount * 4);
  offset += vertexCount * 16;
  const restJoints = new Float32Array(buffer, offset, 72);
  const children = Array.from({ length: 24 }, () => []);
  SMPL_PARENTS.forEach((parent, joint) => {
    if (parent >= 0) children[parent].push(joint);
  });
  smpl = { vertexCount, faces, restVertices, influences, influenceWeights, restJoints, children };
}

/**
 * 读取一条 GT / Prediction 窗口。
 * @param {ArrayBuffer} buffer
 * @param {{id: string, title: string, category: string, fps?: number}} meta
 */
function decodeClip(buffer, meta) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== "QLIP") throw new Error(`Clip ${meta.id} is invalid`);
  const frames = view.getUint32(4, true);
  const referenceY = view.getFloat32(8, true);
  let offset = 12;
  const take = (count) => {
    const values = new Float32Array(buffer, offset, count);
    offset += count * 4;
    return values;
  };
  return {
    ...meta,
    frames,
    referenceY,
    gtRoot: take(frames * 3),
    predRoot: take(frames * 3),
    gtJoints: take(frames * 72),
    predJoints: take(frames * 72),
  };
}

/** @param {Float32Array} matrix */
function setIdentity(matrix) {
  matrix.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
}

/**
 * 把行主序 3×3 矩阵作用到向量上。
 * @param {Float32Array} matrix
 * @param {number} x
 * @param {number} y
 * @param {number} z
 * @param {Float32Array} out
 */
function matVec(matrix, x, y, z, out) {
  out[0] = matrix[0] * x + matrix[1] * y + matrix[2] * z;
  out[1] = matrix[3] * x + matrix[4] * y + matrix[5] * z;
  out[2] = matrix[6] * x + matrix[7] * y + matrix[8] * z;
}

/**
 * @param {Float32Array} left
 * @param {Float32Array} right
 * @param {Float32Array} out
 */
function matMul(left, right, out) {
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      matT[row * 3 + col] = left[row * 3] * right[col] + left[row * 3 + 1] * right[3 + col] + left[row * 3 + 2] * right[6 + col];
    }
  }
  out.set(matT);
}

/**
 * 用单位四元数写入旋转矩阵。
 * @param {number} x
 * @param {number} y
 * @param {number} z
 * @param {number} w
 * @param {Float32Array} out
 */
function quatToMat(x, y, z, w, out) {
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  out[0] = 1 - 2 * (yy + zz);
  out[1] = 2 * (xy - wz);
  out[2] = 2 * (xz + wy);
  out[3] = 2 * (xy + wz);
  out[4] = 1 - 2 * (xx + zz);
  out[5] = 2 * (yz - wx);
  out[6] = 2 * (xz - wy);
  out[7] = 2 * (yz + wx);
  out[8] = 1 - 2 * (xx + yy);
}

/**
 * 求把单位向量 a 转到 b 的旋转。
 * @param {number} ax
 * @param {number} ay
 * @param {number} az
 * @param {number} bx
 * @param {number} by
 * @param {number} bz
 * @param {Float32Array} out
 */
function rotationBetween(ax, ay, az, bx, by, bz, out) {
  const dot = ax * bx + ay * by + az * bz;
  if (dot > 0.999999) {
    setIdentity(out);
    return;
  }
  let x;
  let y;
  let z;
  let w;
  if (dot < -0.999999) {
    x = Math.abs(ax) < 0.9 ? 0 : -az;
    y = Math.abs(ax) < 0.9 ? az : 0;
    z = Math.abs(ax) < 0.9 ? -ay : ax;
    const length = Math.hypot(x, y, z) || 1;
    x /= length;
    y /= length;
    z /= length;
    w = 0;
  } else {
    x = ay * bz - az * by;
    y = az * bx - ax * bz;
    z = ax * by - ay * bx;
    w = 1 + dot;
    const length = Math.hypot(x, y, z, w);
    x /= length;
    y /= length;
    z /= length;
    w /= length;
  }
  quatToMat(x, y, z, w, out);
}

/**
 * @param {Float32Array} points
 * @param {number} joint
 * @param {number} left
 * @param {number} right
 * @param {number} upward
 * @param {Float32Array} out
 */
function writeBasis(points, joint, left, right, upward, out) {
  let x0 = points[left * 3] - points[right * 3];
  let x1 = points[left * 3 + 1] - points[right * 3 + 1];
  let x2 = points[left * 3 + 2] - points[right * 3 + 2];
  let y0 = points[upward * 3] - points[joint * 3];
  let y1 = points[upward * 3 + 1] - points[joint * 3 + 1];
  let y2 = points[upward * 3 + 2] - points[joint * 3 + 2];
  const xl = Math.hypot(x0, x1, x2) || 1;
  const yl = Math.hypot(y0, y1, y2) || 1;
  x0 /= xl;
  x1 /= xl;
  x2 /= xl;
  y0 /= yl;
  y1 /= yl;
  y2 /= yl;
  let z0 = x1 * y2 - x2 * y1;
  let z1 = x2 * y0 - x0 * y2;
  let z2 = x0 * y1 - x1 * y0;
  const zl = Math.hypot(z0, z1, z2) || 1;
  z0 /= zl;
  z1 /= zl;
  z2 /= zl;
  x0 = y1 * z2 - y2 * z1;
  x1 = y2 * z0 - y0 * z2;
  x2 = y0 * z1 - y1 * z0;
  out.set([x0, y0, z0, x1, y1, z1, x2, y2, z2]);
}

/**
 * 用髋或肩的三点建立关节朝向。
 * @param {number} joint
 * @param {number} left
 * @param {number} right
 * @param {number} upward
 * @param {Float32Array} points
 */
function frameJoint(joint, left, right, upward, points) {
  writeBasis(smpl.restJoints, joint, left, right, upward, matA);
  writeBasis(points, joint, left, right, upward, matB);
  matA.set([matA[0], matA[3], matA[6], matA[1], matA[4], matA[7], matA[2], matA[5], matA[8]]);
  matMul(matB, matA, rotations[joint]);
}

/**
 * 单子关节只补摆动，扭转沿用父关节。
 * @param {number} joint
 * @param {number} child
 * @param {Float32Array} points
 */
function swingJoint(joint, child, points) {
  const parent = rotations[SMPL_PARENTS[joint]];
  let rx = smpl.restJoints[child * 3] - smpl.restJoints[joint * 3];
  let ry = smpl.restJoints[child * 3 + 1] - smpl.restJoints[joint * 3 + 1];
  let rz = smpl.restJoints[child * 3 + 2] - smpl.restJoints[joint * 3 + 2];
  const cx = points[child * 3] - points[joint * 3];
  const cy = points[child * 3 + 1] - points[joint * 3 + 1];
  const cz = points[child * 3 + 2] - points[joint * 3 + 2];
  const restLength = Math.hypot(rx, ry, rz);
  const currentLength = Math.hypot(cx, cy, cz);
  if (restLength < 1e-5 || currentLength < 1e-5) {
    rotations[joint].set(parent);
    return;
  }
  rx /= restLength;
  ry /= restLength;
  rz /= restLength;
  const lx = parent[0] * cx + parent[3] * cy + parent[6] * cz;
  const ly = parent[1] * cx + parent[4] * cy + parent[7] * cz;
  const lz = parent[2] * cx + parent[5] * cy + parent[8] * cz;
  const localLength = Math.hypot(lx, ly, lz) || 1;
  rotationBetween(rx, ry, rz, lx / localLength, ly / localLength, lz / localLength, matA);
  matMul(parent, matA, rotations[joint]);
}

/**
 * 由当前关节位置估计 24 个全局旋转。
 * @param {Float32Array} points
 */
function solveRotations(points) {
  frameJoint(0, 1, 2, 3, points);
  for (let joint = 1; joint < 24; joint += 1) {
    if (joint === 9) frameJoint(9, 13, 14, 12, points);
    else if (smpl.children[joint].length === 1) swingJoint(joint, smpl.children[joint][0], points);
    else rotations[joint].set(rotations[SMPL_PARENTS[joint]]);
  }
}

/**
 * 创建一个可蒙皮的 SMPL 身体。
 * @param {string} color
 * @returns {{group: THREE.Group, mesh: THREE.Mesh}}
 */
function createActor(color) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(smpl.vertexCount * 3), 3));
  geometry.setIndex(new THREE.BufferAttribute(smpl.faces.slice(), 1));
  const material = new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.46,
    metalness: 0,
    clearcoat: 0.12,
    clearcoatRoughness: 0.55,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.frustumCulled = false;
  const group = new THREE.Group();
  group.add(mesh);
  scene.add(group);
  return { group, mesh };
}

/**
 * 生成浅色棋盘地面，铺在参考高度上。
 * @returns {THREE.Mesh}
 */
function createFloor() {
  const image = document.createElement("canvas");
  image.width = 256;
  image.height = 256;
  const ctx = image.getContext("2d");
  const cells = 8;
  const cell = image.width / cells;
  for (let y = 0; y < cells; y += 1) {
    for (let x = 0; x < cells; x += 1) {
      ctx.fillStyle = (x + y) % 2 === 0 ? "#fbfcfd" : "#e7ebf0";
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
  const texture = new THREE.CanvasTexture(image);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(18, 18);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(72, 72),
    new THREE.MeshStandardMaterial({ map: texture, roughness: 1, metalness: 0 }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  scene.add(mesh);
  return mesh;
}

/** 初始化并排视图用的场景、灯光和轨道控制。 */
function initScene() {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setClearColor("#f7f8fa");
  scene = new THREE.Scene();
  scene.background = new THREE.Color("#f7f8fa");
  scene.add(new THREE.AmbientLight(0xffffff, 0.72));
  scene.add(new THREE.HemisphereLight(0xffffff, 0xf4f1ee, 0.85));
  keyLight = new THREE.DirectionalLight(0xfffaf6, 1.85);
  keyLight.castShadow = true;
  keyLight.shadow.intensity = 0.32;
  keyLight.shadow.mapSize.set(1024, 1024);
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 28;
  keyLight.shadow.camera.left = -6;
  keyLight.shadow.camera.right = 6;
  keyLight.shadow.camera.top = 6;
  keyLight.shadow.camera.bottom = -6;
  fillLight = new THREE.DirectionalLight(0xf7fbff, 0.7);
  scene.add(keyLight, keyLight.target, fillLight, fillLight.target);
  camera = new THREE.PerspectiveCamera(32, 1, 0.05, 200);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.6;
  controls.maxDistance = 18;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.target.set(0, 0.9, 0);
  floor = createFloor();
  gtActor = createActor(GT_BODY);
  predActor = createActor(PRED_BODY);
}

/**
 * 把当前帧关节平移到根位置后做线性蒙皮。
 * @param {{mesh: THREE.Mesh}} actor
 * @param {Float32Array} joints
 * @param {Float32Array} roots
 */
function updateActor(actor, joints, roots) {
  const root = frame * 3;
  for (let joint = 0; joint < 24; joint += 1) {
    const source = frame * 72 + joint * 3;
    posedJoints[joint * 3] = joints[source] + roots[root];
    posedJoints[joint * 3 + 1] = joints[source + 1] + roots[root + 1];
    posedJoints[joint * 3 + 2] = joints[source + 2] + roots[root + 2];
  }
  solveRotations(posedJoints);
  const positions = actor.mesh.geometry.attributes.position.array;
  const rest = smpl.restVertices;
  const restJoints = smpl.restJoints;
  for (let vertex = 0; vertex < smpl.vertexCount; vertex += 1) {
    let x = 0;
    let y = 0;
    let z = 0;
    const rx = rest[vertex * 3];
    const ry = rest[vertex * 3 + 1];
    const rz = rest[vertex * 3 + 2];
    for (let influence = 0; influence < 4; influence += 1) {
      const weight = smpl.influenceWeights[vertex * 4 + influence];
      if (weight < 1e-4) continue;
      const joint = smpl.influences[vertex * 4 + influence];
      matVec(
        rotations[joint],
        rx - restJoints[joint * 3],
        ry - restJoints[joint * 3 + 1],
        rz - restJoints[joint * 3 + 2],
        matT,
      );
      x += weight * (matT[0] + posedJoints[joint * 3]);
      y += weight * (matT[1] + posedJoints[joint * 3 + 1]);
      z += weight * (matT[2] + posedJoints[joint * 3 + 2]);
    }
    positions[vertex * 3] = x;
    positions[vertex * 3 + 1] = y;
    positions[vertex * 3 + 2] = z;
  }
  actor.mesh.geometry.attributes.position.needsUpdate = true;
  actor.mesh.geometry.computeVertexNormals();
}

/** 地面轨迹只画到当前帧。 */
function syncTrails() {
  const segments = Math.max(frame, 0);
  trails.forEach((trail) => trail.geometry.setDrawRange(0, segments * 6));
}

/**
 * 用一条固定宽度的贴地带画出根轨迹。普通 WebGL 线段不能加粗。
 * @param {Float32Array} positions
 * @param {number} color
 * @param {number} width
 * @returns {THREE.Mesh}
 */
function makeRibbon(positions, color, width) {
  const count = positions.length / 3;
  const segments = Math.max(count - 1, 0);
  const half = width / 2;
  const ribbon = new Float32Array(segments * 12);
  const indices = new Uint32Array(segments * 6);
  for (let i = 0; i < segments; i += 1) {
    const ax = positions[i * 3];
    const ay = positions[i * 3 + 1];
    const az = positions[i * 3 + 2];
    const bx = positions[(i + 1) * 3];
    const by = positions[(i + 1) * 3 + 1];
    const bz = positions[(i + 1) * 3 + 2];
    let dx = bx - ax;
    let dz = bz - az;
    const length = Math.hypot(dx, dz) || 1;
    dx /= length;
    dz /= length;
    const px = -dz * half;
    const pz = dx * half;
    const vertex = i * 12;
    ribbon[vertex] = ax + px;
    ribbon[vertex + 1] = ay;
    ribbon[vertex + 2] = az + pz;
    ribbon[vertex + 3] = ax - px;
    ribbon[vertex + 4] = ay;
    ribbon[vertex + 5] = az - pz;
    ribbon[vertex + 6] = bx + px;
    ribbon[vertex + 7] = by;
    ribbon[vertex + 8] = bz + pz;
    ribbon[vertex + 9] = bx - px;
    ribbon[vertex + 10] = by;
    ribbon[vertex + 11] = bz - pz;
    const base = i * 4;
    const index = i * 6;
    indices[index] = base;
    indices[index + 1] = base + 1;
    indices[index + 2] = base + 2;
    indices[index + 3] = base + 1;
    indices[index + 4] = base + 3;
    indices[index + 5] = base + 2;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(ribbon, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.setDrawRange(0, 0);
  const material = new THREE.MeshBasicMaterial({
    color,
    depthTest: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return mesh;
}

/** 按当前序列重建贴地轨迹。 */
function makeTrails() {
  trails.forEach((trail) => {
    scene.remove(trail);
    trail.geometry.dispose();
    trail.material.dispose();
  });
  trails = [];
  const y = clip.referenceY + 0.03;
  [[clip.gtRoot, GT_TRAIL], [clip.predRoot, PRED_TRAIL]].forEach(([array, color]) => {
    const positions = new Float32Array(clip.frames * 3);
    for (let i = 0; i < clip.frames; i += 1) {
      positions[i * 3] = array[i * 3];
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = array[i * 3 + 2];
    }
    const ribbon = makeRibbon(positions, color, TRAIL_WIDTH);
    scene.add(ribbon);
    trails.push(ribbon);
  });
}

/** 棋盘放在这段动作的水平中心，高度取脚的低分位。 */
function placeFloor() {
  let sx = 0;
  let sz = 0;
  let n = 0;
  [clip.gtRoot, clip.predRoot].forEach((array) => {
    for (let i = 0; i < array.length; i += 3) {
      sx += array[i];
      sz += array[i + 2];
      n += 1;
    }
  });
  floor.position.set(sx / n, clip.referenceY, sz / n);
}

/** 取两具身体当前根位置的中点，作为轨道中心。 */
function bodyCenter() {
  const i = frame * 3;
  center.set(
    (clip.gtRoot[i] + clip.predRoot[i]) / 2,
    (clip.gtRoot[i + 1] + clip.predRoot[i + 1]) / 2,
    (clip.gtRoot[i + 2] + clip.predRoot[i + 2]) / 2,
  );
  return center;
}

/** 把主光移到当前人物附近，避免走过棋盘后变暗。 */
function moveLight() {
  bodyCenter();
  keyLight.position.set(center.x + 2.6, clip.referenceY + 6.4, center.z + 3.1);
  keyLight.target.position.set(center.x, clip.referenceY + 0.95, center.z);
  keyLight.target.updateMatrixWorld();
  fillLight.position.set(center.x - 2.8, clip.referenceY + 4.2, center.z - 1.4);
  fillLight.target.position.copy(keyLight.target.position);
  fillLight.target.updateMatrixWorld();
}

/** 刷新两具身体、轨迹和灯光。 */
function updatePose() {
  if (!clip) return;
  updateActor(gtActor, clip.gtJoints, clip.gtRoot);
  updateActor(predActor, clip.predJoints, clip.predRoot);
  syncTrails();
  moveLight();
}

/**
 * 画并排视图。缩略图模式只画 GT，铺满画布。
 * @param {{thumb?: boolean}} [options]
 */
function draw({ thumb = false } = {}) {
  if (!renderer) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = renderer.getPixelRatio();
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    renderer.setSize(width, height, false);
  }
  if (thumb) {
    gtActor.group.visible = true;
    predActor.group.visible = false;
    trails.forEach((trail) => { trail.visible = false; });
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, width, height);
    renderer.render(scene, camera);
    return;
  }
  renderer.setScissorTest(true);
  for (let view = 0; view < 2; view += 1) {
    const left = Math.floor(width * view / 2);
    const viewWidth = Math.floor(width * (view + 1) / 2) - left;
    camera.aspect = viewWidth / height;
    camera.updateProjectionMatrix();
    renderer.setViewport(left, 0, viewWidth, height);
    renderer.setScissor(left, 0, viewWidth, height);
    gtActor.group.visible = view === 0;
    predActor.group.visible = view === 1;
    trails.forEach((trail, index) => { trail.visible = index === view && frame > 0; });
    renderer.render(scene, camera);
  }
  renderer.setScissorTest(false);
}

/**
 * 用当前 GT 髋部朝向放一个前侧 3/4 机位。
 * @param {THREE.Vector3} target
 * @param {number} distance
 * @param {number} [frameIndex]
 * @returns {THREE.Vector3}
 */
function viewOffset(target, distance, frameIndex = frame) {
  const joints = clip.gtJoints;
  const root = clip.gtRoot;
  const base = frameIndex * 72;
  const r = frameIndex * 3;
  const leftX = (joints[base + 3] + root[r]) - (joints[base + 6] + root[r]);
  const leftZ = (joints[base + 5] + root[r + 2]) - (joints[base + 8] + root[r + 2]);
  const length = Math.hypot(leftX, leftZ) || 1;
  const lx = leftX / length;
  const lz = leftZ / length;
  target.set(-lz * 1.25 + lx * 0.42, 0.7, lx * 1.25 + lz * 0.42);
  return target.normalize().multiplyScalar(distance);
}

/**
 * 按整段 2–12 秒的轨迹摆一次机位，之后不再跟随。
 * 半幅画面更窄，距离按路径跨度放大，避免人走出画面。
 */
function resetView() {
  if (!clip) return;
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  [clip.gtRoot, clip.predRoot].forEach((array) => {
    for (let i = 0; i < array.length; i += 3) {
      minX = Math.min(minX, array[i]);
      maxX = Math.max(maxX, array[i]);
      minZ = Math.min(minZ, array[i + 2]);
      maxZ = Math.max(maxZ, array[i + 2]);
    }
  });
  center.set((minX + maxX) / 2, clip.referenceY + 0.9, (minZ + maxZ) / 2);
  const span = Math.max(Math.hypot(maxX - minX, maxZ - minZ), 1.2) + 2.4;
  const halfFov = THREE.MathUtils.degToRad(camera.fov) / 2;
  const distance = Math.max(3.4, (span * 0.5) / Math.tan(halfFov) / 0.9);
  controls.target.copy(center);
  camera.position.copy(center).add(viewOffset(homeOffset, distance, 0));
  controls.update();
  draw();
}

/**
 * 切换播放状态。
 * @param {boolean} next
 */
function setPlaying(next) {
  playing = next && !!clip;
  playButton.disabled = !clip;
  playButton.innerHTML = playing ? PAUSE_ICON : PLAY_ICON;
  playButton.setAttribute("aria-label", playing ? "Pause" : "Play");
}

/**
 * 跳到指定帧并刷新画面。
 * @param {number} next
 */
function setFrame(next) {
  if (!clip) return;
  frame = Math.max(0, Math.min(clip.frames - 1, Math.round(next)));
  seek.value = String(frame);
  updatePose();
  draw();
}

/** 让当前缩略图停在轮播视口中间。 */
function centerActiveThumb() {
  const item = strip.querySelector(".is-active");
  if (!item) return;
  const left = item.offsetLeft - (strip.clientWidth - item.offsetWidth) / 2;
  strip.scrollTo({ left: Math.max(0, left), behavior: "smooth" });
}

/**
 * 选中一条序列，从头播放。
 * @param {number} index
 */
function selectClip(index) {
  const next = clips[(index + clips.length) % clips.length];
  clip = next;
  frame = 0;
  accumulator = 0;
  seek.max = String(next.frames - 1);
  seek.value = "0";
  strip.querySelectorAll(".qual-thumb").forEach((button) => {
    const active = button.dataset.id === next.id;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  placeFloor();
  makeTrails();
  updatePose();
  resetView();
  centerActiveThumb();
  setPlaying(!window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

/**
 * 用当前 GT 姿态生成轮播封面。
 * @param {object} item
 */
function captureThumb(item) {
  clip = item;
  frame = Math.min(item.frames - 1, Math.floor(item.frames * 0.42));
  placeFloor();
  updateActor(gtActor, item.gtJoints, item.gtRoot);
  moveLight();
  bodyCenter();
  controls.target.copy(center);
  camera.position.copy(center).add(viewOffset(homeOffset, 3.45));
  camera.lookAt(center);
  draw({ thumb: true });
  item.button.querySelector("img").src = canvas.toDataURL("image/jpeg", 0.82);
}

/** 按清单建立轮播按钮。 */
function buildCarousel() {
  clips.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "qual-thumb";
    button.dataset.id = item.id;
    button.setAttribute("aria-label", item.title);
    button.setAttribute("aria-pressed", "false");
    button.innerHTML = `<span class="qual-shot"><img alt=""><span class="qual-pill"></span></span>`;
    button.querySelector(".qual-pill").textContent = item.title;
    button.addEventListener("click", () => selectClip(index));
    strip.append(button);
    item.button = button;
  });
}

/** 播放时钟。画面不在视口内时停住。 */
function loop(now) {
  requestAnimationFrame(loop);
  if (!renderer || !clip || !onScreen) {
    lastTime = now;
    return;
  }
  const dt = Math.min(0.05, (now - lastTime) / 1000 || 0);
  lastTime = now;
  if (playing) {
    accumulator += dt * (clip.fps || 30);
    if (accumulator >= 1) {
      const step = Math.floor(accumulator);
      accumulator -= step;
      frame = (frame + step) % clip.frames;
      seek.value = String(frame);
      updatePose();
    }
  }
  const moved = controls.update();
  if (playing || moved) draw();
}

/** 绑定播放、跟随、视角和序列切换。 */
function bindControls() {
  playButton.addEventListener("click", () => setPlaying(!playing));
  seek.addEventListener("input", () => {
    accumulator = 0;
    setFrame(Number(seek.value));
  });
  document.getElementById("qual-prev").addEventListener("click", () => {
    selectClip(clips.indexOf(clip) - 1);
  });
  document.getElementById("qual-next").addEventListener("click", () => {
    selectClip(clips.indexOf(clip) + 1);
  });
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  root.addEventListener("keydown", (event) => {
    if (event.target.closest(".qual-controls")) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      selectClip(clips.indexOf(clip) - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      selectClip(clips.indexOf(clip) + 1);
    }
  });
  const observer = new ResizeObserver(() => draw());
  observer.observe(canvas.parentElement);
  const visibility = new IntersectionObserver(([entry]) => {
    onScreen = entry.isIntersecting;
    if (onScreen) draw();
  });
  visibility.observe(root);
}

/** 加载网格和十段测试动作，再开始轮播。 */
async function init() {
  if (!root || !canvas) return;
  setPlaying(false);
  const [manifest, mesh] = await Promise.all([
    fetch("qualitative/manifest.json?v=2").then((response) => {
      if (!response.ok) throw new Error("manifest");
      return response.json();
    }),
    fetch("qualitative/smpl.bin").then((response) => {
      if (!response.ok) throw new Error("smpl");
      return response.arrayBuffer();
    }),
  ]);
  parseSmpl(mesh);
  initScene();
  const buffers = await Promise.all(manifest.clips.map(async (item) => {
    const response = await fetch(`qualitative/${item.file}?v=2`);
    if (!response.ok) throw new Error(item.file);
    return decodeClip(await response.arrayBuffer(), { ...item, fps: manifest.fps || 30 });
  }));
  clips = buffers;
  buildCarousel();
  bindControls();
  clips.forEach(captureThumb);
  selectClip(0);
  statusEl.hidden = true;
  requestAnimationFrame(loop);
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => {
    console.error(error);
    if (!statusEl) return;
    statusEl.hidden = false;
    statusEl.textContent = "Could not load the comparison viewer.";
  });
});
