using DevKen.YoloPredictor;
using DevKen.YoloPredictor.Yolov5;
using System.Drawing;

namespace Yolo5Test
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var modulepath = Console.ReadLine();
            IYoloPredictor predictor = new YoloPredictorV5(modulepath, backend:YoloPredictorV5.Backend.CUDA, input_width: 640, input_height: 640);
            var picture = Console.ReadLine();
            var detresult = predictor.Predict((Bitmap)Bitmap.FromFile(picture)).NMSFilter().ConfidenceFilter();
            foreach(var det in detresult)
            {

                Console.WriteLine(det.LabelName + " LeftUp=(" + det.Box.MinX + "," + det.Box.MinY + ") Confidence=" + det.Confidence);
            }
        }
    }
}