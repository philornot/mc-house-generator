/**
 * Seeded random number generator for reproducible results.
 * Uses mulberry32 algorithm.
 */
export class SeededRandom {
  /**
   * Creates a new seeded random generator.
   *
   * @param {number} seed - Seed value for reproducibility
   */
  constructor(seed) {
    this.seed = seed;
  }

  /**
   * Generates next random number between 0 and 1.
   *
   * @returns {number} Random number [0, 1)
   */
  next() {
    this.seed = (this.seed + 0x6D2B79F5) | 0;
    let t = Math.imul(this.seed ^ (this.seed >>> 15), 1 | this.seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * Generates random integer in range [min, max].
   *
   * @param {number} min - Minimum value (inclusive)
   * @param {number} max - Maximum value (inclusive)
   * @returns {number} Random integer
   */
  nextInt(min, max) {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  /**
   * Generates random float in range [min, max).
   *
   * @param {number} min - Minimum value (inclusive)
   * @param {number} max - Maximum value (exclusive)
   * @returns {number} Random float
   */
  nextFloat(min, max) {
    return this.next() * (max - min) + min;
  }

  /**
   * Returns true with given probability.
   *
   * @param {number} probability - Probability [0, 1]
   * @returns {boolean} Random boolean
   */
  chance(probability) {
    return this.next() < probability;
  }

  /**
   * Picks random element from array.
   *
   * @param {Array} array - Array to pick from
   * @returns {*} Random element
   */
  choice(array) {
    return array[this.nextInt(0, array.length - 1)];
  }
}