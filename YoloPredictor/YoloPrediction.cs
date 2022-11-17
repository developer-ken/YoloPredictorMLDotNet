using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace DevKen.YoloPredictor
{
    public class YoloPrediction
    {
        public BBox Box { get; set; } //[required]

        public int LabelIndex { get; set; }

        public string? LabelName { get; set; }

        public float Confidence { get; set; }


        public static List<YoloPrediction> NMS(List<YoloPrediction> predictions, float IOU_threshold = 0.45f, float score_threshold = 0.3f)
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
        /// Get the IOU of two predictions
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>IOU value between 0 and 1</returns>
        public static float operator %(YoloPrediction a, YoloPrediction b)
        {
            return ComputeIOU(a, b);
        }

        private static float ComputeIOU(YoloPrediction DRa, YoloPrediction DRb)
        {
            float ay1 = DRa.Box.MinY;
            float ax1 = DRa.Box.MinX;
            float ay2 = DRa.Box.MaxY;
            float ax2 = DRa.Box.MaxX;
            float by1 = DRb.Box.MinY;
            float bx1 = DRb.Box.MinX;
            float by2 = DRb.Box.MaxY;
            float bx2 = DRb.Box.MaxX;


            float x_left = Math.Max(ax1, bx1);
            float y_top = Math.Max(ay1, by1);
            float x_right = Math.Min(ax2, bx2);
            float y_bottom = Math.Min(ay2, by2);

            if (x_right < x_left || y_bottom < y_top)
                return 0;
            float intersection_area = (x_right - x_left) * (y_bottom - y_top);
            float bb1_area = (ax2 - ax1) * (ay2 - ay1);
            float bb2_area = (bx2 - bx1) * (by2 - by1);
            float iou = intersection_area / (bb1_area + bb2_area - intersection_area);

            //Debug.Assert(iou >= 0 && iou <= 1);
            return iou;
        }
    }


    public class BBox
    {
        public float MinX, MinY, MaxX, MaxY;
        public BBox(float minX, float minY, float maxX, float maxY)
        {
            MinX = minX;
            MinY = minY;
            MaxX = maxX;
            MaxY = maxY;
        }
    }

}
