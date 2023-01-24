import { InferenceSession } from 'onnxjs';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  beginPath,
  clearCanvas,
  closePath,
  drawLine,
  getContext,
  getImageData,
  init,
  moveCursor,
} from './utils/canvas';
import { BLOCK_SIZE, MODEL_URL, NOTEBOOK_HTML_URL } from './utils/constants';
import { convertImgDataToTensor, predict } from './utils/model';

const App = () => {
  const drawCanvasRef = useRef<HTMLCanvasElement>(null);
  const transformedCanvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<HTMLCanvasElement>(null);

  const [drawCtx, setDrawCtx] = useState<CanvasRenderingContext2D>();
  const [transformCtx, setTransformCtx] = useState<CanvasRenderingContext2D>();
  const [viewrCtx, setViewerCtx] = useState<CanvasRenderingContext2D>();

  const session = useMemo(() => new InferenceSession(), []);
  const side = useMemo(() => BLOCK_SIZE * 12, []);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(-1);

  useEffect(() => {
    (async () => {
      try {
        await session.loadModel(MODEL_URL);
      } catch (err) {
        console.log("Couldn't load model");
        console.error(err);
      }
    })();
  }, []);

  // useLayoutEffect(() => {
  //   const drawCanvas = drawCanvasRef.current;
  //   const transformedCanvas = transformedCanvasRef.current;

  //   if (!drawCanvas || !transformedCanvas) return;

  //   setDrawCtx(getContext(drawCanvas));
  //   setTransformCtx(getContext(transformedCanvas));

  //   init(drawCanvas, side);
  //   init(transformedCanvas, side);
  // }, []);

  const makePrediction = useCallback(async () => {
    const data = getImageData(transformCtx!);
    const imgTensor = convertImgDataToTensor(data);

    const probabilities = await predict(session, imgTensor);

    let maxx = -1e9,
      maxIdx = -1;
    for (let i = 0; i < probabilities.length; ++i) {
      if (probabilities[i] > maxx) {
        maxx = probabilities[i];
        maxIdx = i;
      }
    }

    setPrediction(maxIdx);
  }, [isDrawing, transformCtx]);

  const startDraw = useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
      beginPath(drawCtx!);
      moveCursor(drawCtx!, ev.nativeEvent);
      setIsDrawing(true);
    },
    [isDrawing, drawCtx]
  );

  const stopDraw = useCallback(() => {
    closePath(drawCtx!);
    setIsDrawing(false);
    makePrediction();
  }, [isDrawing, drawCtx]);

  const draw = useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
      if (!isDrawing) return;

      drawLine(drawCtx!, ev.nativeEvent);

      //
      transformCtx!.imageSmoothingEnabled = false;
      transformCtx!.save();
      transformCtx!.clearRect(0, 0, side, side);
      transformCtx!.scale(BLOCK_SIZE / side, BLOCK_SIZE / side);
      transformCtx!.drawImage(drawCtx!.canvas, 0, 0);
      transformCtx!.restore();

      viewrCtx!.imageSmoothingEnabled = false;
      viewrCtx!.save();
      viewrCtx!.clearRect(0, 0, side, side);
      viewrCtx!.scale(side / BLOCK_SIZE, side / BLOCK_SIZE);
      viewrCtx!.drawImage(transformCtx!.canvas, 0, 0);
      viewrCtx!.restore();
      //

      // no need to pixelate manually as, we are pixelating by scaling the canvases
      // pixelate(drawCtx!, transformCtx!, BLOCK_SIZE);

      makePrediction();
    },
    [isDrawing, drawCtx, viewrCtx, transformCtx]
  );

  // this method uses 3 canvases to first take the image, then downscale it, then upscale again to get the pixelated image
  useEffect(() => {
    const drawCanvas = drawCanvasRef.current;
    const transformedCanvas = transformedCanvasRef.current;
    const viewer = viewerRef.current;

    if (!drawCanvas || !transformedCanvas || !viewer) return;

    setDrawCtx(getContext(drawCanvas));
    init(drawCanvas, side);

    setTransformCtx(getContext(transformedCanvas));
    init(transformedCanvas, BLOCK_SIZE);

    setViewerCtx(getContext(viewer));
    init(viewer, side);
  }, []);

  return (
    <div className='container h-screen flex m-auto items-center flex-col p-4'>
      <h1 className='my-8'>
        <p className='text-4xl text-white font-bold'>MNIST Classification</p>
      </h1>
      <div className='mb-8 w-full'>
        <p className='text-2xl text-slate-400 text-center'>
          Draw a number on the left canvas. The second canvas is the input that
          goes into the model.
        </p>
      </div>
      <div className='container flex flex-wrap flex-col'>
        <div className='gap-16 flex flex-wrap justify-center justify-items-center mb-10'>
          <canvas
            ref={drawCanvasRef}
            onMouseDown={startDraw}
            onMouseUp={stopDraw}
            onMouseMove={draw}
          />
          <canvas ref={transformedCanvasRef} className='hidden' />
          <canvas ref={viewerRef} />
        </div>
        <div className='flex flex-col justify-start w-full justify-items-center px-4'>
          <div>
            <h1 className='text-3xl font-bold dark:text-white text-center'>
              Prediction
            </h1>
            <h1 className='mt-8 text-3xl font-bold dark:text-slate-400 text-center'>
              {prediction}
            </h1>
          </div>

          <div className='mt-8 flex justify-center justify-items-center'>
            <button
              onClick={() => {
                clearCanvas(drawCtx!);
                clearCanvas(viewrCtx!);
                setPrediction(-1);
                // pixelate(drawCtx!, transformCtx!, BLOCK_SIZE);
              }}
              title='Clear Canvas'
              className='transition-all px-16 py-2 font-semibold text-sm bg-sky-500 text-white hover:drop-shadow-xl rounded-full border-4 border-sky-500 hover:border-white'
            >
              Clear Canvas
            </button>
          </div>
        </div>
      </div>
      <div className='mt-8 w-full min-h-full'>
        <h1 className='text-3xl text-white font-bold mb-4'>Jupyter Notebook</h1>
        <iframe src={NOTEBOOK_HTML_URL} className='w-full h-full' />
      </div>
    </div>
  );
};

export default App;
