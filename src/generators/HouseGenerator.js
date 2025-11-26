import {Block} from '../models/Block.js';
import {House} from '../models/House.js';
import {RoofGenerator} from './RoofGenerator.js';

/**
 * Generates basic house structures with parametric design.
 */
export class HouseGenerator {
    /**
     * Generates a simple house with given dimensions.
     *
     * @param {number} width - Width (X axis)
     * @param {number} height - Height (Y axis)
     * @param {number} depth - Depth (Z axis)
     * @param {Object} options - Generation options
     * @returns {House} Generated house
     */
    static generate(width, height, depth, options = {}) {
        const house = new House(width, height, depth);
        const {
            addWindows = true,
            windowSpacing = 2,
            doorPosition = 'front',
            addStairs = false,
            roofProfile = 'gable',
            roofDirection = 'x'
        } = options;

        // Generate floor
        this.generateFloor(house, width, depth);

        // Generate walls with door and windows
        this.generateWalls(house, width, height, depth, {
            addWindows, windowSpacing, doorPosition
        });

        // Generate roof using RoofGenerator
        RoofGenerator.generate(house, width, height, depth, roofProfile, {
            direction: roofDirection
        });

        // Add stairs if requested
        if (addStairs) {
            this.generateStairs(house, width, height, depth);
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
     * Generates walls with optional windows and door.
     *
     * @param {House} house - House instance
     * @param {number} width - Width
     * @param {number} height - Height
     * @param {number} depth - Depth
     * @param {Object} options - Wall options
     */
    static generateWalls(house, width, height, depth, options) {
        const {addWindows, windowSpacing, doorPosition} = options;

        for (let y = 1; y <= height; y++) {
            // Front wall (z = 0)
            for (let x = 0; x < width; x++) {
                const isDoorway = doorPosition === 'front' && x === Math.floor(width / 2) && y <= 2;

                const isWindow = addWindows && !isDoorway && y === Math.floor(height / 2) && x % windowSpacing === 1 && x > 0 && x < width - 1;

                if (!isDoorway) {
                    house.addBlock(new Block(x, y, 0, isWindow ? 'window' : 'wall'));
                }
            }

            // Back wall (z = depth - 1)
            for (let x = 0; x < width; x++) {
                const isWindow = addWindows && y === Math.floor(height / 2) && x % windowSpacing === 1 && x > 0 && x < width - 1;

                house.addBlock(new Block(x, y, depth - 1, isWindow ? 'window' : 'wall'));
            }

            // Left wall (x = 0)
            for (let z = 1; z < depth - 1; z++) {
                const isWindow = addWindows && y === Math.floor(height / 2) && z % windowSpacing === 1;

                house.addBlock(new Block(0, y, z, isWindow ? 'window' : 'wall'));
            }

            // Right wall (x = width - 1)
            for (let z = 1; z < depth - 1; z++) {
                const isDoorway = doorPosition === 'right' && z === Math.floor(depth / 2) && y <= 2;

                const isWindow = addWindows && !isDoorway && y === Math.floor(height / 2) && z % windowSpacing === 1;

                if (!isDoorway) {
                    house.addBlock(new Block(width - 1, y, z, isWindow ? 'window' : 'wall'));
                }
            }
        }
    }

    /**
     * Generates stairs outside the house.
     *
     * @param {House} house - House instance
     * @param {number} width - Width
     * @param {number} height - Height
     * @param {number} depth - Depth
     */
    static generateStairs(house, width, height, depth) {
        // Add stairs in front of the door rotated 180 degrees
        const doorX = Math.floor(width / 2);
        house.addBlock(Block.createStairs(doorX, 0, -1, 'north', false));
        house.addBlock(Block.createStairs(doorX - 1, 0, -1, 'north', false));
    }
}