# YoloPredictorMLDotNet  
  
This project provide following packages:  
DevKen.YoloPredictor: [![NuGet version (DevKen.YoloPredictor)](https://img.shields.io/nuget/v/DevKen.YoloPredictor.svg?style=flat)](https://www.nuget.org/packages/DevKen.YoloPredictor/)  
DevKen.YoloPredictor.Yolov5: [![NuGet version (DevKen.YoloPredictor.Yolov5)](https://img.shields.io/nuget/v/DevKen.YoloPredictor.Yolov5.svg?style=flat)](https://www.nuget.org/packages/DevKen.YoloPredictor.Yolov5/)  
  
------  
## What is this?  
This project is designed to make YOLO intergration with .NET fast, easy and convenient. Programmers don't have to understand details about YOLO or ML, just feed the Predictor with trained moudle and images, then receive the results.  
## Use in critical projects?  
DO NOT do that. This project is still under developing and comes with NO guarantee.  
Post issues if any problems are found.
## Usage example
```  
//Create a predictor by providing modulepath and a backend.
//Install corresponding OnnxRuntime nuget package.
//For example, you need Microsoft.ML.OnnxRuntime.Gpu for CUDA.
IYoloPredictor predictor = new YoloPredictorV5(modulepath, backend:YoloPredictorV5.Backend.CUDA);

//Predict on a Bitmap, then apply NMS and Confidence filter.
var detresult = predictor.Predict((Bitmap)Bitmap.FromFile(picture)).NMSFilter().ConfidenceFilter();
```
