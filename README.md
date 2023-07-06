# YoloPredictorMLDotNet  
  
This project provide following packages:  
|Package|Link|
|-------|----|
|DevKen.YoloPredictor|[![NuGet version (DevKen.YoloPredictor)](https://img.shields.io/nuget/v/DevKen.YoloPredictor.svg?style=flat)](https://www.nuget.org/packages/DevKen.YoloPredictor/)|  
|DevKen.YoloPredictor.Yolov5|[![NuGet version (DevKen.YoloPredictor.Yolov5)](https://img.shields.io/nuget/v/DevKen.YoloPredictor.Yolov5.svg?style=flat)](https://www.nuget.org/packages/DevKen.YoloPredictor.Yolov5/)|  
|DevKen.YoloPredictor.OpenCvBridge|[![NuGet version (DevKen.YoloPredictor.OpenCvBridge)](https://img.shields.io/nuget/v/DevKen.YoloPredictor.OpenCvBridge.svg?style=flat)](https://www.nuget.org/packages/DevKen.YoloPredictor.OpenCvBridge/)|  
  
------  
## What is this?  
This project is designed to make YOLO intergration with .NET fast, easy and convenient. Programmers don't have to understand details about YOLO or ML, just feed the Predictor with trained moudle and images, then receive the results.  

## Reproduce under DirectML  
In this section, I will explain some details on how to use this repo.  
### What is DirectML?
DirectML is a new feature in DirectX 12. It allows most kind of GPUs to be used to accelerate machine learning, even some of those embeded into  CPUs.  
### Envirounment
When editing this, I just verified that the 3 packages in the table above is all I need to run YOLOv5 via DirectML on `Intel(R) Iris(R) Xe Graphics` in my laptop. It's running `Windows11 22H2 version 22621.1848`, with `Windows Feature Experience Pack 1000.22642.1000.0`.  
  
Packages: 
|Package|Version|
|-------|-------|
|DevKen.YoloPredictor|22.11.322.18|  
|DevKen.YoloPredictor.Yolov5|22.11.322.18|  
|DevKen.YoloPredictor.OpenCvBridge|22.11.322.4|  
|OpenCvSharp4|4.7.0.20230115|  
|OpenCvSharp4.runtime.win|4.7.0.20230115|  
|OpenCvSharp4.Extentions|4.6.0.20220608|  
### Export onnx file from YOLOv5  
Following command is verified to export an onnx file works well with this repo.  
Remember to replace `PATH_TO_TRAINED_WEIGHT_PT` with your weight file (`latest.pt` or `best.pt`),  
replace `PATH_TO_DATASET_CONFIG_FILE_YML` with your `dataset.yml` file.  
Do **NOT** use `--dynamic` if you want to use auto-configure function of the repo, as that prevent some metadata from being written into the onnx file. You have to fill all parameters when creating the YoloPredictor if you use  `--dynamic`.  
`--opset 15` is required because the onnx execution engine we use will complain about opset>15 and throw an exception.
```
python export.py --weights PATH_TO_TRAINED_WEIGHT_PT --data PATH_TO_DATASET_CONFIG_FILE_YML --include onnx --opset 15
```  
Then use the code examples above, change `backend:YoloPredictorV5.Backend.CUDA` to `backend:YoloPredictorV5.Backend.DirectML`.
## Usage example  
### Predict on Bitmap  
```  
//Create a predictor by providing modulepath and a backend.
//Install corresponding OnnxRuntime nuget package.
//For example, you need Microsoft.ML.OnnxRuntime.Gpu for CUDA.
YoloPredictor predictor = new YoloPredictorV5(modulepath, backend:YoloPredictorV5.Backend.CUDA);

//Predict on a Bitmap, then apply NMS and Confidence filter.
var detresult = predictor.Predict((Bitmap)Bitmap.FromFile(picture)).NMSFilter().ConfidenceFilter();
```  
### Predict on Mat  
Opencv read camera and run prediction.
```  
//Create a predictor by providing modulepath and a backend.
YoloPredictor predictor = new YoloPredictorV5(modulepath, backend:YoloPredictorV5.Backend.CUDA);

//Open video device
VideoCapture vc = new VideoCapture(0);

while (true)
{
  //If frame presents
  if (vc.Read(image))
  {
    //Run detector on that frame, then apply NMSFilter and ConfidenceFilter.
    var detresult = predictor.Predict(image).NMSFilter().ConfidenceFilter();
    //** Do something with detresult here **
  }
}
```
