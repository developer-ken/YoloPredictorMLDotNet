using System;
using System.Collections.Generic;
using System.Text;

namespace DevKen.YoloPredictor
{
    public static class PredictionList
    {
        /// <summary>
        /// Run NMS on current Prediction list.
        /// </summary>
        /// <param name="predictions"></param>
        /// <param name="IOU_threshold"></param>
        /// <returns></returns>
        public static List<YoloPrediction> NMSFilter(this List<YoloPrediction> predictions, float IOU_threshold = 0.45f)
        {
            List<YoloPrediction> final_predications = new List<YoloPrediction>();

            for (int i = 0; i < predictions.Count; i++)
            {
                int j = 0;
                for (j = 0; j < final_predications.Count; j++)
                {
                    if (final_predications[j] % predictions[i] > IOU_threshold)
                    {
                        break;
                    }
                }
                if (j == final_predications.Count)
                {
                    final_predications.Add(predictions[i]);
                }
            }
            return final_predications;
        }

        /// <summary>
        /// Filter out detections that have confidence below threshold.
        /// </summary>
        /// <param name="predictions"></param>
        /// <param name="confidence_threshold"></param>
        /// <returns></returns>
        public static List<YoloPrediction> ConfidenceFilter(this List<YoloPrediction> predictions, float confidence_threshold = 0.3f)
        {
            List<YoloPrediction> final_predications = new List<YoloPrediction>();

            foreach(var p in predictions)
            {
                if(p.Confidence>=confidence_threshold)
                final_predications.Add(p);
            }
            return final_predications;
        }
    }
}
