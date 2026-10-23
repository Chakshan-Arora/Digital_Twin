import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { createMriSliceTexture, type SliceOrientation } from './mriSliceTexture';

describe('createMriSliceTexture', () => {
  const orientations: SliceOrientation[] = ['axial', 'coronal', 'sagittal'];

  for (const ori of orientations) {
    it(`generates a valid CanvasTexture for ${ori} orientation`, () => {
      const texture = createMriSliceTexture(ori, 256);
      expect(texture).toBeInstanceOf(THREE.CanvasTexture);
      expect(texture.image).toBeDefined();
      expect(texture.image.width).toBe(256);
      expect(texture.image.height).toBe(256);
      expect(texture.colorSpace).toBe(THREE.SRGBColorSpace);
    });
  }
});

