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
using System.IO;
using Newtonsoft.Json.Linq;

namespace DevKen.YoloPredictor.Yolov5
{
    public class YoloPredictorV5 : IYoloPredictor
    {
        private InferenceSession onnxSession;
        private int PredictClasses;
        public string InputTensorName { get; private set; }
        public Dictionary<int, string?> Names;
        public int Width, Height;

        public enum Backend
        {
            CUDA, DirectML, CPU
        }

        public YoloPredictorV5(string modelfile_onnx, int classes = -1, Backend backend = Backend.CPU, string? input_tensor_name = null, int input_width = -1, int input_height = -1, string optimized_dir = "./OptimizedModles")
        {
            Names = new Dictionary<int, string?>();
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;

            switch (backend)
            {
                case Backend.CUDA:
                    options.AppendExecutionProvider_CUDA();
                    break;
                case Backend.CPU:
                    {
                        options.IntraOpNumThreads = 2;
                        options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                        options.InterOpNumThreads = 6;
                        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                        options.OptimizedModelFilePath = Path.Combine(optimized_dir, "v5.onnx");
                        options.AppendExecutionProvider_CPU(0);
                    }
                    break;
                case Backend.DirectML:
                    options.AppendExecutionProvider_CPU();
                    break;
            }
            Directory.CreateDirectory(optimized_dir);
            // create inference session
            onnxSession = new InferenceSession(modelfile_onnx, options);
            try
            {
                //Auto detecting classes
                if (classes < 1)
                {
                    var metadata = onnxSession.ModelMetadata.CustomMetadataMap;
                    if (metadata.TryGetValue("names", out string data))
                    {
                        JObject jb = JObject.Parse(data);
                        foreach (var obj in jb)
                        {
                            Names.Add(int.Parse(obj.Key), obj.Value?.ToString());
                        }

                        PredictClasses = Names.Count;
                    }
                    else
                    {
                        throw new ArgumentNullException("Can not autodetect classes from onnx. Set a positive number on argument 'classes'.");
                    }
                }
                else
                {
                    PredictClasses = classes;
                }

                //Auto detecting input tensor name
                if (input_tensor_name is null)
                {
                    var metadata = onnxSession.InputMetadata;
                    if (metadata.Count == 0)
                    {
                        throw new ArgumentNullException("No entrance found in onnx. Check your module file or set 'input_tensor_name'.");
                    }
                    if (metadata.Count != 1)
                    {
                        throw new ArgumentNullException("Muiltiple entrances found in onnx. Set 'input_tensor_name' manualy is required.");
                    }
                    InputTensorName = metadata.Keys.First();
                }
                else
                {
                    InputTensorName = input_tensor_name;
                }

                //Auto detecting input size
                if (input_width < 1 || input_height < 1)
                {
                    var data = onnxSession.InputMetadata.Values.First().Dimensions;
                    if (data.Length != 4)
                    {
                        throw new ArgumentNullException("input_height & input_width", "Can not resolve InputMetadata for input size detection. " +
                            "Manualy set 'input_height' and 'input_width' is required.");
                    }
                    Height = input_height > 0 ? input_height : data[2];
                    Width = input_width > 0 ? input_width : data[3];
                    if (Height <= 0)
                    {
                        throw new ArgumentNullException("input_height", "Auto detected 'input_height' is invalid." +
                            "Manualy set 'input_height' is required.");
                    }
                    if (Width <= 0)
                    {
                        throw new ArgumentNullException("input_width", "Auto detected 'input_width' is invalid." +
                            "Manualy set 'input_width' is required.");
                    }
                }
                else
                {
                    Height = input_height;
                    Width = input_width;
                }
            }
            catch (ArgumentNullException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new ArgumentNullException("Something wrong when trying to autodetect ungiven parameters.", ex);
            }
        }

        public List<YoloPrediction> Predict(Bitmap img)
        {
            var resized_image = img.Resize(Width, Height);
            var input_tensor = resized_image.FastToOnnxTensor_13hw();
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor(InputTensorName, input_tensor));
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = onnxSession.Run(container);
            var resultsArray = results.ToArray();
            Tensor<float> tensors = resultsArray[0].AsTensor<float>();
            var array = tensors.ToArray();
            return ParseResults(array);
        }

        private List<YoloPrediction> ParseResults(float[] results)
        {
            //output  1 25200 {dimensions}
            //box format  0,1,2,3 ->box,4->confidence，5-{dimensions} -> classes confidence
            int dimensions = PredictClasses + 5;
            int rows = results.Length / dimensions;
            int confidenceIndex = 4;
            int labelStartIndex = 5;

            float inputlWidth = Width;
            float inputHeight = Height;

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
                        LabelName = Names.ContainsKey(l_index) ? Names[l_index] : null
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
