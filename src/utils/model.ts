import ndarray from 'ndarray';
import { InferenceSession, Tensor } from 'onnxruntime-web';

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
  // in the model we are using float16 for improved performance, but js doesn't have a Float16Array; maybe refer https://stackoverflow.com/questions/20925527/convert-float32array-to-16-bit-float-array-buffer-javascript
  return new Tensor('float32', img4d.data, [1, 1, BLOCK_SIZE, BLOCK_SIZE]); // [1, 1, 28, 28];
};

export const predict = async (
  session: InferenceSession,
  imgTensor: Tensor
): Promise<Float32Array> => {
  try {
    const output = await session.run({
      'input.1': imgTensor,
    });

    return output['48'].data as Float32Array; // idk why it is 48 in the new onnxruntime-web library 🤷‍♂️
  } catch (err) {
    console.error('Error while running session');
    throw err;
  }
};
