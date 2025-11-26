import { Block } from '../models/Block.js';

/**
 * Generates various roof profiles using stairs blocks.
 */
export class RoofGenerator {
  /**
   * Generates a roof based on the specified profile.
   *
   * @param {House} house - House instance to add roof to
   * @param {number} width - Building width (X axis)
   * @param {number} height - Wall height (Y axis)
   * @param {number} depth - Building depth (Z axis)
   * @param {string} profile - Roof profile type
   * @param {Object} options - Additional options
   */
  static generate(house, width, height, depth, profile = 'gable', options = {}) {
    const roofStartY = height + 1;

    switch (profile) {
      case 'gable':
        this.generateGableRoof(house, width, roofStartY, depth, options);
        break;
      case 'gambrel':
        this.generateGambrelRoof(house, width, roofStartY, depth, options);
        break;
      case 'mono-pitched':
        this.generateMonoPitchedRoof(house, width, roofStartY, depth, options);
        break;
      case 'hip':
        this.generateHipRoof(house, width, roofStartY, depth, options);
        break;
      case 'flat':
        this.generateFlatRoof(house, width, roofStartY, depth, options);
        break;
      default:
        this.generateGableRoof(house, width, roofStartY, depth, options);
    }
  }

  /**
   * Generates a gable (A-frame) roof profile.
   * Classic triangular roof with two slopes meeting at a ridge.
   *
   * @param {House} house - House instance
   * @param {number} width - Building width
   * @param {number} startY - Starting Y coordinate
   * @param {number} depth - Building depth
   * @param {Object} options - Options
   */
  static generateGableRoof(house, width, startY, depth, options = {}) {
    const { direction = 'x' } = options;

    if (direction === 'x') {
      // Roof runs along X axis (ridge parallel to X)
      const midZ = Math.floor(depth / 2);
      const roofHeight = Math.ceil(depth / 2);

      for (let layer = 0; layer < roofHeight; layer++) {
        const zOffset = layer;
        const currentY = startY + layer;

        // South-facing slope (positive Z)
        for (let x = 0; x < width; x++) {
          const z = midZ + zOffset;
          if (z < depth) {
            house.addBlock(Block.createStairs(x, currentY, z, 'south', true));
          }
        }

        // North-facing slope (negative Z)
        for (let x = 0; x < width; x++) {
          const z = midZ - zOffset - 1;
          if (z >= 0) {
            house.addBlock(Block.createStairs(x, currentY, z, 'north', true));
          }
        }
      }

      // Ridge blocks (full blocks at the peak)
      if (depth % 2 === 0) {
        const ridgeY = startY + roofHeight - 1;
        for (let x = 0; x < width; x++) {
          house.addBlock(new Block(x, ridgeY, midZ, 'roof'));
          house.addBlock(new Block(x, ridgeY, midZ - 1, 'roof'));
        }
      }
    } else {
      // Roof runs along Z axis (ridge parallel to Z)
      const midX = Math.floor(width / 2);
      const roofHeight = Math.ceil(width / 2);

      for (let layer = 0; layer < roofHeight; layer++) {
        const xOffset = layer;
        const currentY = startY + layer;

        // East-facing slope (positive X)
        for (let z = 0; z < depth; z++) {
          const x = midX + xOffset;
          if (x < width) {
            house.addBlock(Block.createStairs(x, currentY, z, 'east', true));
          }
        }

        // West-facing slope (negative X)
        for (let z = 0; z < depth; z++) {
          const x = midX - xOffset - 1;
          if (x >= 0) {
            house.addBlock(Block.createStairs(x, currentY, z, 'west', true));
          }
        }
      }

      // Ridge blocks (full blocks at the peak)
      if (width % 2 === 0) {
        const ridgeY = startY + roofHeight - 1;
        for (let z = 0; z < depth; z++) {
          house.addBlock(new Block(midX, ridgeY, z, 'roof'));
          house.addBlock(new Block(midX - 1, ridgeY, z, 'roof'));
        }
      }
    }
  }

  /**
   * Generates a gambrel (barn-style) roof profile.
   * Two-slope roof with a steeper lower section and gentler upper section.
   *
   * @param {House} house - House instance
   * @param {number} width - Building width
   * @param {number} startY - Starting Y coordinate
   * @param {number} depth - Building depth
   * @param {Object} options - Options
   */
  static generateGambrelRoof(house, width, startY, depth, options = {}) {
    const { direction = 'x' } = options;

    if (direction === 'x') {
      const midZ = Math.floor(depth / 2);
      const totalHeight = Math.ceil(depth / 2);
      const transitionPoint = Math.floor(totalHeight * 0.6);

      // Lower steep section
      for (let layer = 0; layer < transitionPoint; layer++) {
        const zOffset = layer;
        const currentY = startY + layer;

        // South slope
        for (let x = 0; x < width; x++) {
          const z = midZ + zOffset;
          if (z < depth) {
            house.addBlock(Block.createStairs(x, currentY, z, 'south', true));
          }
        }

        // North slope
        for (let x = 0; x < width; x++) {
          const z = midZ - zOffset - 1;
          if (z >= 0) {
            house.addBlock(Block.createStairs(x, currentY, z, 'north', true));
          }
        }
      }

      // Upper gentle section (every other layer)
      const upperStartZ = transitionPoint;
      for (let layer = 0; layer < (totalHeight - transitionPoint) * 2; layer += 2) {
        const zOffset = upperStartZ + Math.floor(layer / 2);
        const currentY = startY + transitionPoint + Math.floor(layer / 2);

        // South slope
        for (let x = 0; x < width; x++) {
          const z = midZ + zOffset;
          if (z < depth) {
            house.addBlock(Block.createStairs(x, currentY, z, 'south', true));
          }
        }

        // North slope
        for (let x = 0; x < width; x++) {
          const z = midZ - zOffset - 1;
          if (z >= 0) {
            house.addBlock(Block.createStairs(x, currentY, z, 'north', true));
          }
        }
      }
    } else {
      const midX = Math.floor(width / 2);
      const totalHeight = Math.ceil(width / 2);
      const transitionPoint = Math.floor(totalHeight * 0.6);

      // Lower steep section
      for (let layer = 0; layer < transitionPoint; layer++) {
        const xOffset = layer;
        const currentY = startY + layer;

        // East slope
        for (let z = 0; z < depth; z++) {
          const x = midX + xOffset;
          if (x < width) {
            house.addBlock(Block.createStairs(x, currentY, z, 'east', true));
          }
        }

        // West slope
        for (let z = 0; z < depth; z++) {
          const x = midX - xOffset - 1;
          if (x >= 0) {
            house.addBlock(Block.createStairs(x, currentY, z, 'west', true));
          }
        }
      }

      // Upper gentle section
      const upperStartX = transitionPoint;
      for (let layer = 0; layer < (totalHeight - transitionPoint) * 2; layer += 2) {
        const xOffset = upperStartX + Math.floor(layer / 2);
        const currentY = startY + transitionPoint + Math.floor(layer / 2);

        // East slope
        for (let z = 0; z < depth; z++) {
          const x = midX + xOffset;
          if (x < width) {
            house.addBlock(Block.createStairs(x, currentY, z, 'east', true));
          }
        }

        // West slope
        for (let z = 0; z < depth; z++) {
          const x = midX - xOffset - 1;
          if (x >= 0) {
            house.addBlock(Block.createStairs(x, currentY, z, 'west', true));
          }
        }
      }
    }
  }

  /**
   * Generates a mono-pitched (single slope) roof.
   *
   * @param {House} house - House instance
   * @param {number} width - Building width
   * @param {number} startY - Starting Y coordinate
   * @param {number} depth - Building depth
   * @param {Object} options - Options
   */
  static generateMonoPitchedRoof(house, width, startY, depth, options = {}) {
    const { direction = 'south' } = options;

    const roofHeight = Math.max(Math.floor(depth / 2), 2);

    for (let layer = 0; layer < roofHeight; layer++) {
      const currentY = startY + layer;

      switch (direction) {
        case 'south':
          for (let x = 0; x < width; x++) {
            const z = layer;
            if (z < depth) {
              house.addBlock(Block.createStairs(x, currentY, z, 'south', true));
            }
          }
          break;

        case 'north':
          for (let x = 0; x < width; x++) {
            const z = depth - 1 - layer;
            if (z >= 0) {
              house.addBlock(Block.createStairs(x, currentY, z, 'north', true));
            }
          }
          break;

        case 'east':
          for (let z = 0; z < depth; z++) {
            const x = layer;
            if (x < width) {
              house.addBlock(Block.createStairs(x, currentY, z, 'east', true));
            }
          }
          break;

        case 'west':
          for (let z = 0; z < depth; z++) {
            const x = width - 1 - layer;
            if (x >= 0) {
              house.addBlock(Block.createStairs(x, currentY, z, 'west', true));
            }
          }
          break;
      }
    }
  }

  /**
   * Generates a hip roof (slopes on all four sides).
   *
   * @param {House} house - House instance
   * @param {number} width - Building width
   * @param {number} startY - Starting Y coordinate
   * @param {number} depth - Building depth
   * @param {Object} options - Options
   */
  static generateHipRoof(house, width, startY, depth, options = {}) {
    const maxLayers = Math.min(Math.floor(width / 2), Math.floor(depth / 2));

    for (let layer = 0; layer < maxLayers; layer++) {
      const currentY = startY + layer;

      // Calculate bounds for this layer
      const minX = layer;
      const maxX = width - 1 - layer;
      const minZ = layer;
      const maxZ = depth - 1 - layer;

      if (minX > maxX || minZ > maxZ) break;

      // North edge (z = minZ)
      for (let x = minX; x <= maxX; x++) {
        house.addBlock(Block.createStairs(x, currentY, minZ, 'north', true));
      }

      // South edge (z = maxZ)
      for (let x = minX; x <= maxX; x++) {
        house.addBlock(Block.createStairs(x, currentY, maxZ, 'south', true));
      }

      // West edge (x = minX)
      for (let z = minZ + 1; z < maxZ; z++) {
        house.addBlock(Block.createStairs(minX, currentY, z, 'west', true));
      }

      // East edge (x = maxX)
      for (let z = minZ + 1; z < maxZ; z++) {
        house.addBlock(Block.createStairs(maxX, currentY, z, 'east', true));
      }
    }

    // Fill remaining center area with full blocks
    const finalLayer = maxLayers;
    const centerMinX = finalLayer;
    const centerMaxX = width - 1 - finalLayer;
    const centerMinZ = finalLayer;
    const centerMaxZ = depth - 1 - finalLayer;

    if (centerMinX <= centerMaxX && centerMinZ <= centerMaxZ) {
      for (let x = centerMinX; x <= centerMaxX; x++) {
        for (let z = centerMinZ; z <= centerMaxZ; z++) {
          house.addBlock(new Block(x, startY + finalLayer, z, 'roof'));
        }
      }
    }
  }

  /**
   * Generates a flat roof (original style, kept for compatibility).
   *
   * @param {House} house - House instance
   * @param {number} width - Building width
   * @param {number} startY - Starting Y coordinate
   * @param {number} depth - Building depth
   * @param {Object} options - Options
   */
  static generateFlatRoof(house, width, startY, depth, options = {}) {
    for (let x = 0; x < width; x++) {
      for (let z = 0; z < depth; z++) {
        house.addBlock(new Block(x, startY, z, 'roof'));
      }
    }
  }
}