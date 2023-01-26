import { InferenceSession } from 'onnxjs';
import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

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

type TouchEventType = React.TouchEvent<HTMLCanvasElement>;
type MouseEventType = React.MouseEvent<HTMLCanvasElement, MouseEvent>;

// type predicts
const isTouchEvent = (
  ev: TouchEventType | MouseEventType
): ev is TouchEventType => {
  return (ev as TouchEventType).touches !== undefined;
};

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
  const [prediction, setPrediction] = useState<number | null>(null);

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

  const preventDefault = useCallback(
    (ev: TouchEvent) => {
      if (drawCtx && drawCtx.canvas === ev.target) {
        ev.preventDefault();
      }
    },
    [drawCtx]
  );

  // this method uses 3 canvases to first take the image, then downscale it, then upscale again to get the pixelated image
  useLayoutEffect(() => {
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

    drawCanvas.addEventListener('touchstart', preventDefault, {
      passive: false,
    });
    drawCanvas.addEventListener('touchend', preventDefault, { passive: false });
    drawCanvas.addEventListener('touchmove', preventDefault, {
      passive: false,
    });

    return () => {
      drawCanvas.removeEventListener('touchstart', preventDefault);
      drawCanvas.removeEventListener('touchend', preventDefault);
      drawCanvas.removeEventListener('touchmove', preventDefault);
    };
  }, []);

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
    (ev: MouseEventType | TouchEventType) => {
      beginPath(drawCtx!);

      let offsetX = 0,
        offsetY = 0;
      if (isTouchEvent(ev)) {
        (offsetX = ev.touches[0].clientX - drawCtx!.canvas.offsetLeft),
          (offsetY = ev.touches[0].clientY - drawCtx!.canvas.offsetTop);
      } else {
        (offsetX = ev.nativeEvent.offsetX), (offsetY = ev.nativeEvent.offsetY);
      }
      moveCursor(drawCtx!, {
        offsetX,
        offsetY,
      });
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
    (ev: MouseEventType | TouchEventType) => {
      if (!isDrawing) return;

      let offsetX = 0,
        offsetY = 0;
      if (isTouchEvent(ev)) {
        (offsetX = ev.touches[0].clientX - drawCtx!.canvas.offsetLeft),
          (offsetY = ev.touches[0].clientY - drawCtx!.canvas.offsetTop);
      } else {
        (offsetX = ev.nativeEvent.offsetX), (offsetY = ev.nativeEvent.offsetY);
      }

      drawLine(drawCtx!, { offsetX, offsetY });

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
    },
    [isDrawing, drawCtx, viewrCtx, transformCtx]
  );

  return (
    <div className='container h-screen flex m-auto items-center flex-col p-4'>
      <h1 className='my-8 text-4xl text-white font-bold'>
        MNIST Classification
      </h1>
      <div className='mb-8 w-full'>
        <p className='text-2xl text-slate-400 text-center'>
          Draw a number on the first canvas. The second canvas is the input that
          goes into the model.
        </p>
      </div>
      <div className='container flex flex-wrap flex-col'>
        <div className='gap-16 flex flex-wrap justify-center items-center mb-10'>
          <canvas
            ref={drawCanvasRef}
            onMouseDown={startDraw}
            onMouseUp={stopDraw}
            onMouseMove={draw}
            onTouchStart={startDraw}
            onTouchEnd={stopDraw}
            onTouchMove={draw}
          />
          <canvas ref={transformedCanvasRef} className='hidden' />
          <canvas ref={viewerRef} />
        </div>
        <div className='flex flex-col justify-start w-full items-center px-4'>
          <div>
            <h1 className='text-3xl font-bold dark:text-white text-center'>
              Prediction
            </h1>
            <h1 className='mt-8 text-3xl font-bold dark:text-slate-400 text-center'>
              {prediction !== null ? (
                prediction
              ) : (
                <>Draw a number for prediction.</>
              )}
            </h1>
          </div>

          <div className='mt-8 w-full flex flex-col justify-center items-center'>
            <button
              onClick={() => {
                clearCanvas(drawCtx!);
                clearCanvas(viewrCtx!);
                setPrediction(null);
                // pixelate(drawCtx!, transformCtx!, BLOCK_SIZE);
              }}
              title='Clear Canvas'
              className='transition-all w-fit px-16 py-2 font-semibold text-sm bg-sky-500 text-white hover:drop-shadow-xl rounded-full border-4 border-sky-500 hover:border-white'
            >
              Clear Canvas
            </button>
            <p className='text-1xl text-slate-500 w-fit mt-4'>
              <a
                href='https://github.com/DYSLEVIUM/mnist-cnn-react'
                target='_blank'
                title='Github'
                className='underline cursor-pointer'
              >
                Github
              </a>
            </p>
          </div>
        </div>
      </div>
      <div className='mt-8 w-full min-h-full'>
        <h1 className='text-3xl text-white font-bold mb-4'>Jupyter Notebook</h1>
        <iframe src={NOTEBOOK_HTML_URL} className='w-full h-full overflow-scroll' />
      </div>
    </div>
  );
};

export default App;
