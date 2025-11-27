import * as THREE from 'three';

/**
 * Renders different types of blocks in Three.js.
 */
export class BlockRenderer {
  /**
   * Color palette for different block types.
   */
  static COLORS = {
    floor: 0x8B4513,    // Brown
    wall: 0xDEB887,     // Tan
    roof: 0xA52A2A,     // Dark red
    window: 0x87CEEB,   // Sky blue
    stairs: 0xCD853F,   // Peru/tan
    column: 0x696969,   // Dim gray (stone-like)
    default: 0x888888   // Gray
  };

  /**
   * Creates a standard cube mesh for a block.
   *
   * @param {Block} block - Block to render
   * @returns {THREE.Mesh} Block mesh
   */
  static createBlockMesh(block) {
    const color = this.COLORS[block.type] || this.COLORS.default;

    if (block.type === 'stairs') {
      return this.createStairsMesh(block);
    }

    if (block.type === 'column') {
      return this.createColumnMesh(block);
    }

    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshLambertMaterial({ color });
    const mesh = new THREE.Mesh(geometry, material);

    mesh.position.set(block.x + 0.5, block.y + 0.5, block.z + 0.5);

    // Add edges for better visibility
    this.addEdges(mesh, geometry);

    mesh.userData.block = block;

    return mesh;
  }

  /**
   * Creates a column mesh - smaller than regular blocks for detail.
   *
   * @param {Block} block - Column block
   * @returns {THREE.Mesh} Column mesh
   */
  static createColumnMesh(block) {
    const color = this.COLORS.column;

    // Columns are thinner (0.4x0.4 in XZ plane, full height in Y)
    const geometry = new THREE.BoxGeometry(0.4, 1, 0.4);
    const material = new THREE.MeshLambertMaterial({ color });
    const mesh = new THREE.Mesh(geometry, material);

    mesh.position.set(block.x + 0.5, block.y + 0.5, block.z + 0.5);

    // Add edges for better visibility
    this.addEdges(mesh, geometry);

    mesh.userData.block = block;

    return mesh;
  }

  /**
   * Creates a stairs mesh with proper geometry.
   *
   * @param {Block} block - Stairs block
   * @returns {THREE.Group} Stairs group
   */
  static createStairsMesh(block) {
    const group = new THREE.Group();
    const color = this.COLORS.stairs;
    const material = new THREE.MeshLambertMaterial({ color });

    // Stairs consist of two steps
    // Bottom step: 1x0.5x1
    const bottomGeometry = new THREE.BoxGeometry(1, 0.5, 1);
    const bottomStep = new THREE.Mesh(bottomGeometry, material);
    bottomStep.position.set(0, -0.25, 0);
    this.addEdges(bottomStep, bottomGeometry);
    group.add(bottomStep);

    // Top step: 1x0.5x0.5
    const topGeometry = new THREE.BoxGeometry(1, 0.5, 0.5);
    const topStep = new THREE.Mesh(topGeometry, material);

    // Position based on upside down status
    if (block.isUpsideDown()) {
      topStep.position.set(0, 0.25, -0.25);
    } else {
      topStep.position.set(0, 0.25, 0.25);
    }

    this.addEdges(topStep, topGeometry);
    group.add(topStep);

    // Apply rotation and position
    group.rotation.y = block.getRotation();

    // Flip if upside down
    if (block.isUpsideDown()) {
      group.rotation.z = Math.PI;
    }

    group.position.set(block.x + 0.5, block.y + 0.5, block.z + 0.5);
    group.userData.block = block;

    return group;
  }

  /**
   * Adds edge lines to a mesh for better visibility.
   *
   * @param {THREE.Mesh} mesh - Mesh to add edges to
   * @param {THREE.BufferGeometry} geometry - Geometry for edges
   */
  static addEdges(mesh, geometry) {
    const edges = new THREE.EdgesGeometry(geometry);
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x000000,
      linewidth: 1
    });
    const wireframe = new THREE.LineSegments(edges, lineMaterial);
    mesh.add(wireframe);
  }

  /**
   * Renders an entire house to the scene.
   *
   * @param {House} house - House to render
   * @param {THREE.Scene} scene - Three.js scene
   */
  static renderHouse(house, scene) {
    // Remove old house blocks
    const blocksToRemove = scene.children.filter(
      child => child.userData.isHouseBlock
    );
    blocksToRemove.forEach(block => scene.remove(block));

    // Add new blocks
    house.blocks.forEach(block => {
      const mesh = this.createBlockMesh(block);
      mesh.userData.isHouseBlock = true;
      scene.add(mesh);
    });
  }
}