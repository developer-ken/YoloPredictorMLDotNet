using DevKen.YoloPredictor;
using DevKen.YoloPredictor.Yolov5;
using DevKen.YoloPredictor.OpenCvBridge;
using System.Drawing;
using OpenCvSharp;

namespace Yolo5Test
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var modulepath = Console.ReadLine();
            YoloPredictor predictor = new YoloPredictorV5(modulepath, backend:YoloPredictorV5.Backend.CUDA, input_width: 640, input_height: 640);

            VideoCapture vc = new VideoCapture(0);
            Mat image = new Mat();
            DateTime lastgrab = DateTime.Now;
            int cnt = 0;
            while (true)
            {
                if (vc.Read(image))
                {
                    var detresult = predictor.Predict(image).NMSFilter().ConfidenceFilter();
                    Cv2.ImShow("Camera", image);
                    Cv2.WaitKey(1);
                    if (cnt > 30)
                    {
                        double fps = cnt / (DateTime.Now - lastgrab).TotalSeconds;
                        cnt = 0;
                        Console.Title = "fps:" + fps;
                        lastgrab = DateTime.Now;
                    }
                    cnt++;
                }
            }
        }
    }
}