/**
 * Represents a complete house structure with all blocks.
 */
export class House {
  /**
   * Creates a new house instance.
   *
   * @param {number} width - Width (X axis)
   * @param {number} height - Height (Y axis)
   * @param {number} depth - Depth (Z axis)
   */
  constructor(width, height, depth) {
    this.width = width;
    this.height = height;
    this.depth = depth;
    this.blocks = [];
  }

  /**
   * Adds a block to the house.
   *
   * @param {Block} block - Block to add
   */
  addBlock(block) {
    this.blocks.push(block);
  }

  /**
   * Gets all blocks of a specific type.
   *
   * @param {string} type - Block type to filter
   * @returns {Block[]} Filtered blocks
   */
  getBlocksByType(type) {
    return this.blocks.filter(block => block.type === type);
  }

  /**
   * Checks if a position is occupied by a block.
   *
   * @param {number} x - X coordinate
   * @param {number} y - Y coordinate
   * @param {number} z - Z coordinate
   * @returns {boolean} True if position is occupied
   */
  isPositionOccupied(x, y, z) {
    return this.blocks.some(block =>
      block.x === x && block.y === y && block.z === z
    );
  }

  /**
   * Gets block at specific position.
   *
   * @param {number} x - X coordinate
   * @param {number} y - Y coordinate
   * @param {number} z - Z coordinate
   * @returns {Block|null} Block at position or null
   */
  getBlockAt(x, y, z) {
    return this.blocks.find(block =>
      block.x === x && block.y === y && block.z === z
    ) || null;
  }

  /**
   * Removes all blocks from the house.
   */
  clear() {
    this.blocks = [];
  }

  /**
   * Gets total number of blocks.
   *
   * @returns {number} Block count
   */
  getBlockCount() {
    return this.blocks.length;
  }
}