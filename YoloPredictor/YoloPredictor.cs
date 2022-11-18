using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;

namespace DevKen.YoloPredictor
{
    public abstract class YoloPredictor
    {
        public abstract List<YoloPrediction> Predict(Bitmap img);
        public abstract Task<List<YoloPrediction>> PredictAsync(Bitmap img);
    }
}