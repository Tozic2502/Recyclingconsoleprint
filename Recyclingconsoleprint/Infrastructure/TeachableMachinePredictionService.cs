using Domain;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Recyclingconsoleprint.Service;

public class TeachableMachinePredictionService : IImagePredictionService
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;
    private readonly string[] _labels;

    public TeachableMachinePredictionService(string modelPath, string labelsPath)
    {
        _mlContext = new MLContext();

        // Læs labels fra fil
        _labels = File.ReadAllLines(labelsPath);

        // Lav ML.NET pipeline
        var pipeline = _mlContext.Transforms.LoadImages(
                outputColumnName: "input",
                imageFolder: "",
                inputColumnName: nameof(ImageData.ImagePath))
            .Append(_mlContext.Transforms.ResizeImages(
                outputColumnName: "input",
                imageWidth: 224,
                imageHeight: 224,
                inputColumnName: "input"))
            .Append(_mlContext.Transforms.ExtractPixels(
                outputColumnName: "input"))
            .Append(_mlContext.Model.LoadTensorFlowModel(modelPath)
                .ScoreTensorFlowModel(
                    outputColumnNames: new[] { "Identity" },  // typisk output layer fra TM
                    inputColumnNames: new[] { "input" },
                    addBatchDimensionInput: true));

        _model = pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<ImageData>()));
    }

    public PredictionResult Predict(ImageData image)
    {
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<ImageData, TensorFlowPrediction>(_model);
        var result = predictionEngine.Predict(image);

        var maxIndex = result.PredictedLabels.AsSpan().IndexOf(result.PredictedLabels.Max());

        return new PredictionResult
        {
            ImagePath = image.ImagePath,
            Label = _labels[maxIndex],
            Probability = result.PredictedLabels[maxIndex]
        };
    }

    private class TensorFlowPrediction
    {
        [ColumnName("Identity")]   // output layer fra TM-model
        public float[] PredictedLabels { get; set; }
    }
}
