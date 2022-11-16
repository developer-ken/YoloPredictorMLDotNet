using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;

namespace DevKen.YoloPredictor
{
    public interface IYoloPredictor
    {
        List<YoloPrediction> Predict(Bitmap img);

        Task<List<YoloPrediction>> PredictAsync(Bitmap img);
    }
}
