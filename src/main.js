import { SceneManager } from './renderer/SceneManager.js';
import { BlockRenderer } from './renderer/BlockRenderer.js';
import { HouseGenerator } from './generators/HouseGenerator.js';
import { RandomHouseGenerator } from './generators/RandomHouseGenerator.js';

/**
 * Main application class.
 */
class App {
  /**
   * Initializes the application.
   */
  constructor() {
    this.sceneManager = null;
    this.currentHouse = null;
    this.mode = 'parametric'; // 'parametric' or 'random'

    // Parametric mode parameters
    this.params = {
      width: 8,
      height: 4,
      depth: 6,
      addWindows: true,
      addStairs: true,
      addColumns: true,
      columnSpacing: 3,
      roofProfile: 'gable',
      roofDirection: 'x'
    };

    // Random mode parameters
    this.seed = Date.now();

    // Roof descriptions
    this.roofDescriptions = {
      'gable': 'Classic triangular roof with two slopes meeting at a ridge.',
      'gambrel': 'Barn-style roof with a steep lower section and gentle upper section.',
      'hip': 'Roof with slopes on all four sides, creating a pyramid-like shape.',
      'mono-pitched': 'Simple single-slope roof, modern and minimalist.',
      'flat': 'Traditional flat roof with no slopes (original style).'
    };

    this.init();
  }

  /**
   * Initializes the application.
   */
  init() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.setup());
    } else {
      this.setup();
    }
  }

  /**
   * Sets up the scene and UI.
   */
  setup() {
    const container = document.getElementById('canvas-container');
    this.sceneManager = new SceneManager(container);
    this.sceneManager.animate();

    this.setupUI();
    this.generateHouse();
  }

  /**
   * Sets up UI event listeners.
   */
  setupUI() {
    // Mode toggle
    const parametricBtn = document.getElementById('parametric-mode');
    const randomBtn = document.getElementById('random-mode');

    parametricBtn.addEventListener('click', () => {
      this.mode = 'parametric';
      this.updateModeUI();
      this.generateHouse();
    });

    randomBtn.addEventListener('click', () => {
      this.mode = 'random';
      this.updateModeUI();
      this.generateHouse();
    });

    // Parametric controls
    document.getElementById('width').addEventListener('input', (e) => {
      this.params.width = parseInt(e.target.value);
      document.getElementById('width-value').textContent = e.target.value;
      this.generateHouse();
    });

    document.getElementById('height').addEventListener('input', (e) => {
      this.params.height = parseInt(e.target.value);
      document.getElementById('height-value').textContent = e.target.value;
      this.generateHouse();
    });

    document.getElementById('depth').addEventListener('input', (e) => {
      this.params.depth = parseInt(e.target.value);
      document.getElementById('depth-value').textContent = e.target.value;
      this.generateHouse();
    });

    document.getElementById('windows').addEventListener('change', (e) => {
      this.params.addWindows = e.target.checked;
      this.generateHouse();
    });

    document.getElementById('stairs').addEventListener('change', (e) => {
      this.params.addStairs = e.target.checked;
      this.generateHouse();
    });

    document.getElementById('columns').addEventListener('change', (e) => {
      this.params.addColumns = e.target.checked;
      this.generateHouse();
    });

    document.getElementById('column-spacing').addEventListener('input', (e) => {
      this.params.columnSpacing = parseInt(e.target.value);
      document.getElementById('column-spacing-value').textContent = e.target.value;
      this.generateHouse();
    });

    // Roof controls
    document.getElementById('roof-profile').addEventListener('change', (e) => {
      this.params.roofProfile = e.target.value;
      this.updateRoofDescription();
      this.generateHouse();
    });

    document.getElementById('roof-direction').addEventListener('change', (e) => {
      this.params.roofDirection = e.target.value;
      this.generateHouse();
    });

    // Random controls
    document.getElementById('seed').addEventListener('input', (e) => {
      this.seed = parseInt(e.target.value) || 0;
      if (this.mode === 'random') {
        this.generateHouse();
      }
    });

    document.getElementById('random-seed').addEventListener('click', () => {
      this.seed = Date.now();
      document.getElementById('seed').value = this.seed;
      if (this.mode === 'random') {
        this.generateHouse();
      }
    });

    document.getElementById('regenerate').addEventListener('click', () => {
      this.generateHouse();
    });

    // Initialize roof description
    this.updateRoofDescription();
  }

  /**
   * Updates UI based on current mode.
   */
  updateModeUI() {
    const parametricBtn = document.getElementById('parametric-mode');
    const randomBtn = document.getElementById('random-mode');
    const parametricControls = document.getElementById('parametric-controls');
    const randomControls = document.getElementById('random-controls');

    if (this.mode === 'parametric') {
      parametricBtn.classList.add('active');
      randomBtn.classList.remove('active');
      parametricControls.style.display = 'block';
      randomControls.style.display = 'none';
    } else {
      parametricBtn.classList.remove('active');
      randomBtn.classList.add('active');
      parametricControls.style.display = 'none';
      randomControls.style.display = 'block';
    }
  }

  /**
   * Updates roof description based on selected profile.
   */
  updateRoofDescription() {
    const descElement = document.getElementById('roof-description');
    const profile = this.params.roofProfile;
    descElement.textContent = this.roofDescriptions[profile] || '';
  }

  /**
   * Generates a house based on current mode.
   */
  generateHouse() {
    if (this.mode === 'parametric') {
      this.currentHouse = HouseGenerator.generate(
        this.params.width,
        this.params.height,
        this.params.depth,
        {
          addWindows: this.params.addWindows,
          addStairs: this.params.addStairs,
          addColumns: this.params.addColumns,
          columnSpacing: this.params.columnSpacing,
          windowSpacing: 2,
          roofProfile: this.params.roofProfile,
          roofDirection: this.params.roofDirection
        }
      );
    } else {
      this.currentHouse = RandomHouseGenerator.generate(this.seed);
    }

    this.renderHouse();
    this.updateStats();
  }

  /**
   * Renders the current house.
   */
  renderHouse() {
    if (!this.currentHouse || !this.sceneManager) return;

    BlockRenderer.renderHouse(this.currentHouse, this.sceneManager.scene);

    // Update camera to center on house
    this.sceneManager.updateCameraPosition(
      this.currentHouse.width / 2,
      this.currentHouse.height / 2,
      this.currentHouse.depth / 2
    );
  }

  /**
   * Updates statistics display.
   */
  updateStats() {
    if (!this.currentHouse) return;

    const stats = document.getElementById('stats');
    const roofBlocks = this.currentHouse.getBlocksByType('roof').length +
                       this.currentHouse.getBlocksByType('stairs').length;
    const columnBlocks = this.currentHouse.getBlocksByType('column').length;

    stats.innerHTML = `
      <div><strong>House Statistics</strong></div>
      <div>Dimensions: ${this.currentHouse.width}×${this.currentHouse.height}×${this.currentHouse.depth}</div>
      <div>Total blocks: ${this.currentHouse.getBlockCount()}</div>
      <div>Walls: ${this.currentHouse.getBlocksByType('wall').length}</div>
      <div>Windows: ${this.currentHouse.getBlocksByType('window').length}</div>
      <div>Roof blocks: ${roofBlocks}</div>
      <div>Columns: ${columnBlocks}</div>
      <div>Entrance stairs: ${this.currentHouse.getBlocksByType('stairs').filter(b => b.y === 0).length}</div>
    `;
  }
}

// Start the application
new App();