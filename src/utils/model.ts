import ndarray from 'ndarray';
import { InferenceSession, Tensor } from 'onnxjs';

import { BLOCK_SIZE } from './constants';

export const convertImgDataToTensor = (imgData: Uint8ClampedArray) => {
  // Convert the 1D array to a 2D array
  const pixels = new Float32Array(BLOCK_SIZE * BLOCK_SIZE);
  for (let i = 0; i < imgData.length; i += 4) {
    pixels[i / 4] = imgData[i];
  }

  const img = ndarray(pixels, [BLOCK_SIZE, BLOCK_SIZE]);

  // add a new dimension to the 2D array to get a 4D array
  const img4d = ndarray(new Float32Array(1 * 1 * BLOCK_SIZE * BLOCK_SIZE), [
    1,
    1,
    BLOCK_SIZE,
    BLOCK_SIZE,
  ]);

  // Copy the 2D array to the first element of the 4D array
  img4d.data.set(img.data);

  // Create a Tensor with the 4D array
  return new Tensor(img4d.data, 'float32', [1, 1, 28, 28]); // [1, 1, 28, 28]
};

export const predict = async (session: InferenceSession, imgTensor: Tensor) => {
  const outputMap = await session.run([imgTensor]);
  const outputTensor = outputMap.values().next().value;

  return outputTensor.data as Float32Array;
};
