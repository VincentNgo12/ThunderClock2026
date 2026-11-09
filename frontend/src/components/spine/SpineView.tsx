// three.js view of the vertebral column driven by the four spine sensors.
// The scene is built once; telemetry updates only move targets that the render loop
// eases toward, so React re-renders never rebuild geometry.

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  computeSpinePose,
  fillMissingAnchors,
  neutralCenter,
  SENSOR_ANCHORS,
  VERTEBRAE,
  type InterpolationMode,
  type VertebraSpec,
} from "../../lib/posture3d/spine";
import { testFrameAt } from "../../lib/posture3d/testAnimation";
import { REGION_LABELS, type FreshnessStatus, type Region } from "../../telemetry/types";

export type ViewName = "oblique" | "side" | "back";

export interface SensorInput {
  status: FreshnessStatus;
  rotation: THREE.Quaternion | null; // already neutral-corrected; null = no data, inferred from neighbours
}

interface Props {
  sensors: Record<Region, SensorInput>;
  view: { name: ViewName; nonce: number }; // a new nonce re-applies the same preset
  showAxes: boolean;
  interpolation: InterpolationMode;
  testAnimation: boolean; // play the built-in bend sequence instead of live data
  onTestLabel?: (label: string) => void; // current movement name while testing
}

const TEST_COLOR = 0x7c3aed; // pods turn violet while the test animation drives them

const STATUS_COLORS: Record<FreshnessStatus, number> = {
  valid: 0x16a37a,
  stale: 0xe0911b,
  unavailable: 0xa3a9b0,
  invalid: 0xd14343,
};

const CAMERA_POSITIONS: Record<ViewName, THREE.Vector3> = {
  oblique: new THREE.Vector3(-0.95, 0.3, 0.95),
  side: new THREE.Vector3(-1.4, 0.05, 0),
  back: new THREE.Vector3(0, 0.05, 1.4),
};

const SMOOTHING_SECONDS = 0.015; // light easing between snapshots (up to 100 Hz; 20 Hz strip quaternions)

const CURVE_COLOR = 0x2563eb;
const CURVE_RADIUS = 0.0032; // metres
const CURVE_SEGMENTS = 160;

// Anatomical vertebrae (BodyParts3D, CC BY 4.0) built by scripts/build_spine_mesh.py. Each mesh is
// already in its vertebra's local frame. Until it loads (or if it fails) the procedural shapes show.
const VERTEBRAE_URL = `${import.meta.env.BASE_URL}models/spine/vertebrae.glb`;

// Where a sensor pod sits on a vertebra (local space): on the skin behind the spinous process.
// The sensor curve passes through this point on every vertebra.
function podOffset(spec: VertebraSpec): THREE.Vector3 {
  if (spec.pod) {
    return new THREE.Vector3(...spec.pod);
  }
  return new THREE.Vector3(0, spec.height * 0.45, spec.depth * 2.2 + 0.012);
}

const boneMaterial = new THREE.MeshStandardMaterial({ color: 0xe6dccb, roughness: 0.7, metalness: 0 });
const discMaterial = new THREE.MeshStandardMaterial({ color: 0x9fb7cc, roughness: 0.5, metalness: 0 });

// One vertebra in local space: base at y=0, top at y=spec.height, posterior is +Z.
function buildVertebra(spec: VertebraSpec, isSacrum: boolean): THREE.Group {
  const group = new THREE.Group();
  const bodyHeight = isSacrum ? spec.height * 0.92 : spec.height * 0.74;
  const discHeight = spec.height - bodyHeight;

  const body = new THREE.Mesh(
    isSacrum
      ? new THREE.CylinderGeometry(0.5, 0.2, 1, 28) // wedge that narrows toward the tip
      : new THREE.CylinderGeometry(0.5, 0.5, 1, 28),
    boneMaterial,
  );
  body.scale.set(spec.width, bodyHeight, spec.depth);
  body.position.y = bodyHeight / 2;
  body.castShadow = true;
  group.add(body);

  const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 1, 28), discMaterial);
  disc.scale.set(spec.width * 0.9, discHeight * 0.9, spec.depth * 0.9);
  disc.position.y = bodyHeight + discHeight / 2;
  group.add(disc);

  if (isSacrum) {
    return group;
  }

  const midY = bodyHeight * 0.55;

  // Neural arch joining the body to the posterior processes.
  const arch = new THREE.Mesh(new THREE.BoxGeometry(spec.width * 0.55, bodyHeight * 0.55, spec.depth * 0.6), boneMaterial);
  arch.position.set(0, midY, spec.depth * 0.68);
  arch.castShadow = true;
  group.add(arch);

  // Spinous process, sloping down and back.
  const spinousLength = spec.depth * 1.15;
  const spinous = new THREE.Mesh(new THREE.BoxGeometry(spec.width * 0.16, bodyHeight * 0.42, spinousLength), boneMaterial);
  spinous.rotation.x = 0.45;
  spinous.position.set(0, midY - Math.sin(0.45) * spinousLength * 0.5, spec.depth * 0.9 + Math.cos(0.45) * spinousLength * 0.5);
  spinous.castShadow = true;
  group.add(spinous);

  // Transverse processes, one per side.
  for (const side of [-1, 1]) {
    const transverse = new THREE.Mesh(new THREE.BoxGeometry(spec.width * 0.5, bodyHeight * 0.3, spec.depth * 0.28), boneMaterial);
    transverse.position.set(side * spec.width * 0.52, midY, spec.depth * 0.62);
    transverse.rotation.y = side * -0.25;
    transverse.castShadow = true;
    group.add(transverse);
  }
  return group;
}

function makeLabel(region: Region): { object: CSS2DObject; dot: HTMLSpanElement } {
  const element = document.createElement("div");
  element.className = "sensor-label";
  const dot = document.createElement("span");
  dot.className = "sensor-dot";
  const text = document.createElement("span");
  text.textContent = REGION_LABELS[region];
  element.append(dot, text);
  return { object: new CSS2DObject(element), dot };
}

export function SpineView({ sensors, view, showAxes, interpolation, testAnimation, onTestLabel }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sensorsRef = useRef(sensors);
  sensorsRef.current = sensors;
  const interpolationRef = useRef(interpolation);
  interpolationRef.current = interpolation;
  const testRef = useRef({ active: testAnimation, onLabel: onTestLabel });
  testRef.current.active = testAnimation;
  testRef.current.onLabel = onTestLabel;
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const axesRef = useRef<THREE.AxesHelper[]>([]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20);
    camera.position.copy(CAMERA_POSITIONS.oblique);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.domElement.className = "label-layer";
    container.appendChild(labelRenderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 0.35;
    controls.maxDistance = 3;
    controlsRef.current = controls;

    scene.add(new THREE.HemisphereLight(0xffffff, 0xd8dde3, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.8);
    key.position.set(-0.8, 1.6, 1.0);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -0.6;
    key.shadow.camera.right = 0.6;
    key.shadow.camera.top = 0.6;
    key.shadow.camera.bottom = -0.6;
    key.shadow.radius = 6;
    scene.add(key);

    // Everything spine-related lives in one group, shifted so the neutral column is centred.
    const spineGroup = new THREE.Group();
    spineGroup.position.copy(neutralCenter()).multiplyScalar(-1);
    scene.add(spineGroup);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(3, 3), new THREE.ShadowMaterial({ opacity: 0.08 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.06;
    floor.receiveShadow = true;
    spineGroup.add(floor);

    // Each vertebra is a pivot group (posed every frame) holding its bone shape and, for anchors, a pod.
    const procedural: THREE.Group[] = [];
    const vertebraGroups = VERTEBRAE.map((spec, index) => {
      const group = new THREE.Group();
      const shape = buildVertebra(
        { ...spec, width: spec.width * 0.6, depth: spec.depth * 0.45 }, // bone extents -> body proportions
        index === 0,
      );
      shape.visible = false; // shown only if the anatomical model cannot be loaded
      procedural.push(shape);
      group.add(shape);
      spineGroup.add(group);
      return group;
    });

    // Per-vertebra extent in its local frame, for framing: the procedural shape until the bones load.
    const localBoxes = VERTEBRAE.map(
      (spec) =>
        new THREE.Box3(
          new THREE.Vector3(-spec.width / 2, 0, -spec.depth / 2),
          new THREE.Vector3(spec.width / 2, spec.height, spec.depth / 2),
        ),
    );
    // Keep the whole column in view and its shadow under it, whatever the pose: the column is posed
    // from a fixed sacrum pivot, so lying down would otherwise swing it out of frame and through the floor.
    const frameBox = new THREE.Box3();
    const partBox = new THREE.Box3();
    const frameTarget = new THREE.Vector3();
    const FRAME_SECONDS = 0.35; // camera-like easing of the re-centring
    let disposed = false;
    new GLTFLoader().load(
      VERTEBRAE_URL,
      (gltf) => {
        if (disposed) {
          return;
        }
        VERTEBRAE.forEach((spec, index) => {
          const source = gltf.scene.getObjectByName(spec.name);
          let geometry: THREE.BufferGeometry | null = null;
          source?.traverse((child) => {
            if (!geometry && child instanceof THREE.Mesh) {
              geometry = child.geometry as THREE.BufferGeometry;
            }
          });
          if (!geometry) {
            procedural[index].visible = true;
            return;
          }
          const shape = geometry as THREE.BufferGeometry;
          if (!shape.getAttribute("normal")) {
            shape.computeVertexNormals();
          }
          shape.computeBoundingBox();
          localBoxes[index] = shape.boundingBox!.clone();
          const bone = new THREE.Mesh(shape, boneMaterial);
          bone.castShadow = true;
          bone.receiveShadow = true;
          vertebraGroups[index].add(bone);
        });
      },
      undefined,
      () => {
        procedural.forEach((shape) => {
          shape.visible = true;
        });
      },
    );

    // Sensor pods sit on the back surface of their anchor vertebra.
    const podGeometry = new RoundedBoxGeometry(0.032, 0.021, 0.01, 3, 0.003);
    const pods = SENSOR_ANCHORS.map(({ region, index }) => {
      const spec = VERTEBRAE[index];
      const material = new THREE.MeshStandardMaterial({ color: STATUS_COLORS.unavailable, roughness: 0.4 });
      const pod = new THREE.Mesh(podGeometry, material);
      pod.position.copy(podOffset(spec));
      pod.castShadow = true;
      vertebraGroups[index].add(pod);

      const axes = new THREE.AxesHelper(0.05);
      axes.visible = false;
      pod.add(axes);
      axesRef.current.push(axes);

      const label = makeLabel(region);
      label.object.position.set(-0.035, 0, 0.01);
      label.object.center.set(1, 0.5);
      pod.add(label.object);
      return { region, material, dot: label.dot };
    });

    // Sensor curve: a tube from the lowest to the highest sensor that passes through every
    // pod and the matching point on each vertebra in between. It is rebuilt each frame from
    // the live pose, so it bends with the sensors and the chosen interpolation.
    const firstAnchor = SENSOR_ANCHORS[0].index;
    const lastAnchor = SENSOR_ANCHORS[SENSOR_ANCHORS.length - 1].index;
    const curveOffsets = VERTEBRAE.map(podOffset);
    const curveMaterial = new THREE.MeshStandardMaterial({
      color: CURVE_COLOR,
      emissive: CURVE_COLOR,
      emissiveIntensity: 0.25,
      roughness: 0.35,
    });
    const curveMesh = new THREE.Mesh(new THREE.BufferGeometry(), curveMaterial);
    curveMesh.castShadow = true;
    spineGroup.add(curveMesh);

    const updateCurve = () => {
      const points: THREE.Vector3[] = [];
      for (let index = firstAnchor; index <= lastAnchor; index += 1) {
        const group = vertebraGroups[index];
        group.updateMatrix();
        points.push(curveOffsets[index].clone().applyMatrix4(group.matrix));
      }
      const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
      curveMesh.geometry.dispose();
      curveMesh.geometry = new THREE.TubeGeometry(curve, CURVE_SEGMENTS, CURVE_RADIUS, 10, false);
    };

    // Displayed rotations ease toward the latest measured ones.
    const displayed = SENSOR_ANCHORS.map(() => new THREE.Quaternion());

    const resize = () => {
      const { clientWidth, clientHeight } = container;
      renderer.setSize(clientWidth, clientHeight);
      labelRenderer.setSize(clientWidth, clientHeight);
      camera.aspect = clientWidth / Math.max(clientHeight, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    const clock = new THREE.Clock();
    let testSeconds = 0;
    let testLabel = "";
    renderer.setAnimationLoop(() => {
      const dt = clock.getDelta();
      const blend = 1 - Math.exp(-dt / SMOOTHING_SECONDS);
      const current = sensorsRef.current;
      const test = testRef.current;

      // The test sequence restarts from upright each time it is switched on.
      let testFrame = null;
      if (test.active) {
        testFrame = testFrameAt(testSeconds);
        testSeconds += dt;
        if (testFrame.label !== testLabel) {
          testLabel = testFrame.label;
          test.onLabel?.(testLabel);
        }
      } else {
        testSeconds = 0;
        testLabel = "";
      }

      // A sensor without data follows its neighbours (slerp between the nearest measured sensors, or a
      // copy of the outermost one) instead of staying pinned where it last was.
      const filled = fillMissingAnchors(
        SENSOR_ANCHORS.map(({ region }) => (testFrame ? testFrame.rotations[region] : current[region].rotation)),
      );
      SENSOR_ANCHORS.forEach(({ region }, k) => {
        const input = current[region];
        if (filled) {
          displayed[k].slerp(filled.rotations[k], blend);
        }
        const color = testFrame ? TEST_COLOR : STATUS_COLORS[input.status];
        pods[k].material.color.setHex(color);
        pods[k].dot.style.background = `#${color.toString(16).padStart(6, "0")}`;
      });

      computeSpinePose(displayed, interpolationRef.current).forEach((pose, index) => {
        vertebraGroups[index].position.copy(pose.position);
        vertebraGroups[index].quaternion.copy(pose.rotation);
      });
      updateCurve();

      // Re-centre the posed column (spine-group space) and rest the shadow just under its lowest bone.
      frameBox.makeEmpty();
      vertebraGroups.forEach((group, index) => {
        group.updateMatrix();
        frameBox.union(partBox.copy(localBoxes[index]).applyMatrix4(group.matrix));
      });
      if (!frameBox.isEmpty()) {
        frameBox.getCenter(frameTarget).multiplyScalar(-1);
        spineGroup.position.lerp(frameTarget, 1 - Math.exp(-dt / FRAME_SECONDS));
        floor.position.y = frameBox.min.y - 0.01;
      }

      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    });

    return () => {
      disposed = true;
      renderer.setAnimationLoop(null);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      axesRef.current = [];
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
        }
      });
      container.removeChild(renderer.domElement);
      container.removeChild(labelRenderer.domElement);
    };
  }, []);

  useEffect(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (camera && controls) {
      camera.position.copy(CAMERA_POSITIONS[view.name]);
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }, [view]);

  useEffect(() => {
    axesRef.current.forEach((axes) => {
      axes.visible = showAxes;
    });
  }, [showAxes]);

  return <div className="spine-view" ref={containerRef} />;
}
