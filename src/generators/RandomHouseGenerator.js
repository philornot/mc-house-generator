import { Block } from '../models/Block.js';
import { House } from '../models/House.js';
import { SeededRandom } from '../utils/SeededRandom.js';
import { RoofGenerator } from './RoofGenerator.js';

/**
 * Generates randomized houses using seeded random generation.
 */
export class RandomHouseGenerator {
  /**
   * Generates a random house based on seed.
   *
   * @param {number} seed - Seed for reproducibility
   * @param {Object} constraints - Generation constraints
   * @returns {House} Generated house
   */
  static generate(seed, constraints = {}) {
    const rng = new SeededRandom(seed);

    const {
      minWidth = 5,
      maxWidth = 12,
      minHeight = 3,
      maxHeight = 7,
      minDepth = 5,
      maxDepth = 12
    } = constraints;

    // Generate random dimensions
    const width = rng.nextInt(minWidth, maxWidth);
    const height = rng.nextInt(minHeight, maxHeight);
    const depth = rng.nextInt(minDepth, maxDepth);

    const house = new House(width, height, depth);

    // Generate floor
    this.generateFloor(house, width, depth);

    // Randomly decide on columns
    const addColumns = rng.chance(0.7); // 70% chance
    const columnSpacing = rng.nextInt(2, 4); // 2-4 blocks apart

    if (addColumns) {
      this.generateColumns(house, width, height, depth, columnSpacing);
    }

    // Generate walls with random features
    this.generateRandomWalls(house, width, height, depth, rng);

    // Choose random roof profile
    const roofProfiles = ['gable', 'gambrel', 'mono-pitched', 'hip', 'flat'];
    const roofProfile = rng.choice(roofProfiles);
    const roofDirection = rng.choice(['x', 'z']);

    // Generate roof using RoofGenerator
    RoofGenerator.generate(house, width, height, depth, roofProfile, {
      direction: roofDirection
    });

    // Randomly add stairs
    if (rng.chance(0.7)) {
      this.generateRandomStairs(house, width, height, depth, rng);
    }

    return house;
  }

  /**
   * Generates floor blocks.
   *
   * @param {House} house - House instance
   * @param {number} width - Width
   * @param {number} depth - Depth
   */
  static generateFloor(house, width, depth) {
    for (let x = 0; x < width; x++) {
      for (let z = 0; z < depth; z++) {
        house.addBlock(new Block(x, 0, z, 'floor'));
      }
    }
  }

  /**
   * Generates columns around the building perimeter.
   *
   * @param {House} house - House instance
   * @param {number} width - Width
   * @param {number} height - Height
   * @param {number} depth - Depth
   * @param {number} spacing - Distance between columns along walls
   */
  static generateColumns(house, width, height, depth, spacing) {
    // Place columns from ground (y=1) to top of walls (y=height)
    for (let y = 1; y <= height; y++) {
      // Corner columns
      house.addBlock(Block.createColumn(-1, y, -1)); // Front-left
      house.addBlock(Block.createColumn(width, y, -1)); // Front-right
      house.addBlock(Block.createColumn(-1, y, depth)); // Back-left
      house.addBlock(Block.createColumn(width, y, depth)); // Back-right

      // Front wall columns (z = -1, between x = 0 and x = width-1)
      for (let x = spacing; x < width; x += spacing) {
        house.addBlock(Block.createColumn(x, y, -1));
      }

      // Back wall columns (z = depth, between x = 0 and x = width-1)
      for (let x = spacing; x < width; x += spacing) {
        house.addBlock(Block.createColumn(x, y, depth));
      }

      // Left wall columns (x = -1, between z = 0 and z = depth-1)
      for (let z = spacing; z < depth; z += spacing) {
        house.addBlock(Block.createColumn(-1, y, z));
      }

      // Right wall columns (x = width, between z = 0 and z = depth-1)
      for (let z = spacing; z < depth; z += spacing) {
        house.addBlock(Block.createColumn(width, y, z));
      }
    }
  }

  /**
   * Generates walls with random windows and door placement.
   *
   * @param {House} house - House instance
   * @param {number} width - Width
   * @param {number} height - Height
   * @param {number} depth - Depth
   * @param {SeededRandom} rng - Random number generator
   */
  static generateRandomWalls(house, width, height, depth, rng) {
    // Choose random door position
    const doorWall = rng.choice(['front', 'right', 'back', 'left']);
    const windowProbability = rng.nextFloat(0.2, 0.4);

    for (let y = 1; y <= height; y++) {
      const windowHeight = Math.floor(height / 2);

      // Front wall (z = 0)
      for (let x = 0; x < width; x++) {
        const isDoor = doorWall === 'front' &&
                      x === Math.floor(width / 2) &&
                      y <= 2;

        const isWindow = !isDoor &&
                        y === windowHeight &&
                        x > 0 && x < width - 1 &&
                        rng.chance(windowProbability);

        if (!isDoor) {
          house.addBlock(new Block(x, y, 0, isWindow ? 'window' : 'wall'));
        }
      }

      // Back wall (z = depth - 1)
      for (let x = 0; x < width; x++) {
        const isDoor = doorWall === 'back' &&
                      x === Math.floor(width / 2) &&
                      y <= 2;

        const isWindow = !isDoor &&
                        y === windowHeight &&
                        x > 0 && x < width - 1 &&
                        rng.chance(windowProbability);

        if (!isDoor) {
          house.addBlock(new Block(x, y, depth - 1, isWindow ? 'window' : 'wall'));
        }
      }

      // Left wall (x = 0)
      for (let z = 1; z < depth - 1; z++) {
        const isDoor = doorWall === 'left' &&
                      z === Math.floor(depth / 2) &&
                      y <= 2;

        const isWindow = !isDoor &&
                        y === windowHeight &&
                        rng.chance(windowProbability);

        if (!isDoor) {
          house.addBlock(new Block(0, y, z, isWindow ? 'window' : 'wall'));
        }
      }

      // Right wall (x = width - 1)
      for (let z = 1; z < depth - 1; z++) {
        const isDoor = doorWall === 'right' &&
                      z === Math.floor(depth / 2) &&
                      y <= 2;

        const isWindow = !isDoor &&
                        y === windowHeight &&
                        rng.chance(windowProbability);

        if (!isDoor) {
          house.addBlock(new Block(width - 1, y, z, isWindow ? 'window' : 'wall'));
        }
      }
    }
  }

  /**
   * Generates random stairs configuration.
   *
   * @param {House} house - House instance
   * @param {number} width - Width
   * @param {number} height - Height
   * @param {number} depth - Depth
   * @param {SeededRandom} rng - Random number generator
   */
  static generateRandomStairs(house, width, height, depth, rng) {
    // Random stairs position and orientation
    const side = rng.choice(['front', 'right', 'back', 'left']);
    const upsideDown = rng.chance(0.1); // 10% chance of upside down stairs

    switch (side) {
      case 'front':
        const frontX = Math.floor(width / 2);
        house.addBlock(Block.createStairs(frontX, 0, -1, 'south', upsideDown));
        house.addBlock(Block.createStairs(frontX - 1, 0, -1, 'south', upsideDown));
        break;

      case 'right':
        const rightZ = Math.floor(depth / 2);
        house.addBlock(Block.createStairs(width, 0, rightZ, 'west', upsideDown));
        house.addBlock(Block.createStairs(width, 0, rightZ - 1, 'west', upsideDown));
        break;

      case 'back':
        const backX = Math.floor(width / 2);
        house.addBlock(Block.createStairs(backX, 0, depth, 'north', upsideDown));
        house.addBlock(Block.createStairs(backX - 1, 0, depth, 'north', upsideDown));
        break;

      case 'left':
        const leftZ = Math.floor(depth / 2);
        house.addBlock(Block.createStairs(-1, 0, leftZ, 'east', upsideDown));
        house.addBlock(Block.createStairs(-1, 0, leftZ - 1, 'east', upsideDown));
        break;
    }
  }
}