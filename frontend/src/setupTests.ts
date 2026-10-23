import '@testing-library/jest-dom/vitest';

// Mock HTMLCanvasElement.getContext for JSDOM
HTMLCanvasElement.prototype.getContext = () => null;

// Mock ResizeObserver for JSDOM
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
