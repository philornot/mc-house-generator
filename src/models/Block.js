/**
 * Represents a single block in 3D space.
 * Supports different block types including stairs with rotation.
 */
export class Block {
  /**
   * Creates a new block instance.
   *
   * @param {number} x - X coordinate
   * @param {number} y - Y coordinate
   * @param {number} z - Z coordinate
   * @param {string} type - Block type (floor, wall, roof, window, stairs)
   * @param {Object} metadata - Additional block properties
   */
  constructor(x, y, z, type = 'default', metadata = {}) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.type = type;
    this.metadata = metadata;
  }

  /**
   * Creates a stairs block with specific rotation.
   *
   * @param {number} x - X coordinate
   * @param {number} y - Y coordinate
   * @param {number} z - Z coordinate
   * @param {string} facing - Direction: 'north', 'south', 'east', 'west'
   * @param {boolean} upsideDown - Whether stairs are inverted
   * @returns {Block} Stairs block instance
   */
  static createStairs(x, y, z, facing = 'north', upsideDown = false) {
    return new Block(x, y, z, 'stairs', {
      facing: facing,
      upsideDown: upsideDown
    });
  }

  /**
   * Gets the rotation angle in radians based on facing direction.
   *
   * @returns {number} Rotation angle
   */
  getRotation() {
    if (this.type !== 'stairs') return 0;

    const rotations = {
      'north': 0,
      'east': Math.PI / 2,
      'south': Math.PI,
      'west': -Math.PI / 2
    };

    return rotations[this.metadata.facing] || 0;
  }

  /**
   * Checks if stairs are upside down.
   *
   * @returns {boolean} True if upside down
   */
  isUpsideDown() {
    return this.type === 'stairs' && this.metadata.upsideDown === true;
  }

  /**
   * Creates a deep copy of this block.
   *
   * @returns {Block} Cloned block
   */
  clone() {
    return new Block(
      this.x,
      this.y,
      this.z,
      this.type,
      { ...this.metadata }
    );
  }
}