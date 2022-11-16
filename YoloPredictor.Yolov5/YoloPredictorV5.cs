using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using DevKen.YoloPredictor;
using Microsoft.ML.OnnxRuntime;
using DevKen.BitmapExtentionsForOnnx;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

namespace DevKen.YoloPredictor.Yolov5
{
    public class YoloPredictorV5 : IYoloPredictor
    {
        private InferenceSession onnxSession;
        private Mutex sessionMutex = new Mutex();

        public YoloPredictorV5(string modelfile_onnx, bool directMl = true)
        {
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;

            if (directMl)
            {
                options.AppendExecutionProvider_DML(0);
            }
            else
            {
                options.IntraOpNumThreads = 2;
                options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                options.InterOpNumThreads = 6;
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                options.OptimizedModelFilePath = ".\\Models\\optimized\\opt_yolov5m.onnx";
                options.AppendExecutionProvider_CPU(0);
            }

            // create inference session
            onnxSession = new InferenceSession(modelfile_onnx, options);
        }

        public List<YoloPrediction> Predict(Bitmap img)
        {
            var resized_image = img.Resize(640, 640);
            var input_tensor = resized_image.FastToOnnxTensor_13hw();
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor("images", input_tensor));
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = onnxSession.Run(container);
            var resultsArray = results.ToArray();
            Tensor<float> tensors = resultsArray[0].AsTensor<float>();
            var array = tensors.ToArray();
            return ParseResults(array);
        }

        private List<YoloPrediction> ParseResults(float[] results)
        {
            //output  1 25200 85
            //box format  0,1,2,3 ->box,4->confidence，5-85 -> coco classes confidence
            int dimensions = 85;
            int rows = results.Length / dimensions;
            int confidenceIndex = 4;
            int labelStartIndex = 5;

            float inputlWidth = 640;
            float inputHeight = 640;

            List<YoloPrediction> detections = new List<YoloPrediction>();

            for (int i = 0; i < rows; ++i)
            {
                var index = i * dimensions;

                if (results[index + confidenceIndex] <= 0.4f) continue;

                for (int j = labelStartIndex; j < dimensions; ++j)
                {
                    results[index + j] = results[index + j] * results[index + confidenceIndex];
                }

                for (int k = labelStartIndex; k < dimensions; ++k)
                {
                    if (results[index + k] <= 0.5f) continue;

                    var value_0 = results[index];
                    var value_1 = results[index + 1];
                    var value_2 = results[index + 2];
                    var value_3 = results[index + 3];

                    var bbox = new BBox((value_0 - value_2 / 2) / inputlWidth,
                                (value_1 - value_3 / 2) / inputlWidth,
                                (value_0 + value_2 / 2) / inputHeight,
                                (value_1 + value_3 / 2) / inputHeight);

                    var l_index = k - labelStartIndex;
                    detections.Add(new YoloPrediction()
                    {
                        Box = bbox,
                        Confidence = results[index + k],
                        LabelIndex = l_index,
                        LabelName = null
                    });
                }


            }
            return detections;
        }

        public Task<List<YoloPrediction>> PredictAsync(Bitmap img)
        {
            throw new NotImplementedException();
        }
    }
}
