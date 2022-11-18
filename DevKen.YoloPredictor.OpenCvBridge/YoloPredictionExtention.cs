using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;

namespace DevKen.YoloPredictor.OpenCvBridge
{
    public static class YoloPredictionExtention
    {
        /// <summary>
        /// Predict on OpenCvSharp.Mat. This requres OpenCvSharp runtimes to exist on your system.
        /// </summary>
        /// <param name="self"></param>
        /// <param name="img"></param>
        /// <returns></returns>
        public static List<YoloPrediction> Predict(this YoloPredictor self,Mat img)
        {
            return self.Predict(OpenCvSharp.Extensions.BitmapConverter.ToBitmap(img));
        }

        /// <summary>
        /// Predict async on OpenCvSharp.Mat. This requres OpenCvSharp runtimes to exist on your system.
        /// </summary>
        /// <param name="self"></param>
        /// <param name="img"></param>
        /// <returns></returns>
        public static async Task<List<YoloPrediction>> PredictAsync(this YoloPredictor self, Mat img)
        {
            return await self.PredictAsync(OpenCvSharp.Extensions.BitmapConverter.ToBitmap(img));
        }
    }
}
