import { LINE_WIDTH } from './constants';

export const getContext = (canvas: HTMLCanvasElement) => {
  return canvas.getContext('2d') as NonNullable<CanvasRenderingContext2D>;
};

export const clearCanvas = (ctx: CanvasRenderingContext2D) => {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillStyle = 'black';
  ctx.fill();
};

export const init = (canvas: HTMLCanvasElement, side: number) => {
  canvas.width = canvas.height = side;
  clearCanvas(getContext(canvas));
};

export const beginPath = (ctx: CanvasRenderingContext2D) => {
  ctx.beginPath();
};

export const moveCursor = (
  ctx: CanvasRenderingContext2D,
  { offsetX, offsetY }: { offsetX: number; offsetY: number }
) => {
  ctx.moveTo(offsetX, offsetY);
};

export const drawLine = (
  ctx: CanvasRenderingContext2D,
  { offsetX, offsetY }: { offsetX: number; offsetY: number }
) => {
  ctx.lineTo(offsetX, offsetY);
  ctx.strokeStyle = 'white';
  ctx.lineWidth = LINE_WIDTH;
  ctx.lineCap = 'round';
  ctx.stroke();
};

export const closePath = (ctx: CanvasRenderingContext2D) => {
  ctx.beginPath();
};

export const pixelate = (
  sourceCtx: CanvasRenderingContext2D,
  targetCtx: CanvasRenderingContext2D,
  block_size: number
) => {
  // targetCtx.imageSmoothingEnabled = false;

  const pixelArr = sourceCtx.getImageData(
    0,
    0,
    sourceCtx.canvas.width,
    sourceCtx.canvas.height
  ).data;

  const sample_size = Math.ceil(sourceCtx.canvas.width / block_size);

  if (!sample_size) {
    throw Error('Block size cannot be greater that canvas side length.');
  }

  for (let y = 0; y < sourceCtx.canvas.height; y += sample_size) {
    for (let x = 0; x < sourceCtx.canvas.width; x += sample_size) {
      const p = (x + y * sourceCtx.canvas.width) * 4;
      targetCtx.fillStyle =
        'rgba(' +
        pixelArr[p] +
        ',' +
        pixelArr[p + 1] +
        ',' +
        pixelArr[p + 2] +
        ',' +
        pixelArr[p + 3] +
        ')';
      targetCtx.fillRect(x, y, sample_size, sample_size);
    }
  }
};

export const getImageData = (ctx: CanvasRenderingContext2D) => {
  return ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data;
};
