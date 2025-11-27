import { describe, it, expect } from 'vitest';
import { RoofGenerator } from '../src/generators/RoofGenerator.js';
import { House } from '../src/models/House.js';

const countBlocksByType = (house, type) => house.blocks.filter(block => block.type === type).length;

describe('RoofGenerator.generateGableRoof', () => {
  it('fills interior with roof blocks and uses non-upside-down stairs on edges', () => {
    const width = 6;
    const height = 4;
    const depth = 6;
    const house = new House(width, height, depth);

    RoofGenerator.generateGableRoof(house, width, height + 1, depth, { direction: 'x' });

    const roofBlocks = countBlocksByType(house, 'roof');
    const stairBlocks = countBlocksByType(house, 'stairs');

    expect(roofBlocks).toBeGreaterThan(0);
    expect(stairBlocks).toBeGreaterThan(0);

    const stairs = house.blocks.filter(block => block.type === 'stairs');
    expect(stairs.every(block => block.metadata.upsideDown === false)).toBe(true);
  });
});

