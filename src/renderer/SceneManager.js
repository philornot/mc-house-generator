import * as THREE from 'three';

/**
 * Manages Three.js scene, camera, and controls.
 */
export class SceneManager {
  /**
   * Creates a new scene manager.
   *
   * @param {HTMLElement} container - DOM element to render to
   */
  constructor(container) {
    this.container = container;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = {
      azimuth: Math.PI / 4,
      elevation: Math.PI / 6,
      distance: 25
    };
    this.isDragging = false;
    this.previousMousePosition = { x: 0, y: 0 };

    this.init();
    this.setupEventListeners();
  }

  /**
   * Initializes the Three.js scene.
   */
  init() {
    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x87ceeb);

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      this.container.clientWidth / this.container.clientHeight,
      0.1,
      1000
    );

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(
      this.container.clientWidth,
      this.container.clientHeight
    );
    this.container.appendChild(this.renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    this.scene.add(directionalLight);

    // Grid
    const gridHelper = new THREE.GridHelper(50, 50);
    this.scene.add(gridHelper);

    this.updateCameraPosition();
  }

  /**
   * Sets up mouse and keyboard event listeners.
   */
  setupEventListeners() {
    const canvas = this.renderer.domElement;

    // Mouse down
    canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    // Mouse move
    canvas.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        const deltaX = e.clientX - this.previousMousePosition.x;
        const deltaY = e.clientY - this.previousMousePosition.y;

        this.controls.azimuth -= deltaX * 0.01;
        this.controls.elevation += deltaY * 0.01;
        this.controls.elevation = Math.max(
          0.1,
          Math.min(Math.PI / 2 - 0.1, this.controls.elevation)
        );

        this.updateCameraPosition();
        this.previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    });

    // Mouse up
    canvas.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    // Mouse wheel
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.controls.distance += e.deltaY * 0.01;
      this.controls.distance = Math.max(5, Math.min(50, this.controls.distance));
      this.updateCameraPosition();
    });

    // Window resize
    window.addEventListener('resize', () => this.handleResize());
  }

  /**
   * Updates camera position based on orbital controls.
   *
   * @param {number} centerX - X coordinate of look target
   * @param {number} centerY - Y coordinate of look target
   * @param {number} centerZ - Z coordinate of look target
   */
  updateCameraPosition(centerX = 0, centerY = 3, centerZ = 0) {
    const { azimuth, elevation, distance } = this.controls;

    this.camera.position.x =
      centerX + distance * Math.sin(elevation) * Math.cos(azimuth);
    this.camera.position.y =
      centerY + distance * Math.cos(elevation);
    this.camera.position.z =
      centerZ + distance * Math.sin(elevation) * Math.sin(azimuth);

    this.camera.lookAt(centerX, centerY, centerZ);
  }

  /**
   * Handles window resize events.
   */
  handleResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  /**
   * Renders the scene.
   */
  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Animation loop.
   */
  animate() {
    requestAnimationFrame(() => this.animate());
    this.render();
  }

  /**
   * Cleans up resources.
   */
  dispose() {
    if (this.renderer && this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}