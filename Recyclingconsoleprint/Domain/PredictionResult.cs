using Microsoft.ML.Data;

namespace Domain;

public class PredictionResult
{
    public string ImagePath { get; set; } = "";
    public string Label { get; set; } = "";
    public float Probability { get; set; }
}
