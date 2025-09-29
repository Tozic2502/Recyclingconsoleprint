using Domain;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;

namespace Infrastructure;

public class ModelTrainer
{
    private readonly MLContext _ml;

    public ModelTrainer()
    {
        _ml = new MLContext(seed: 1);
    }

    public (ITransformer model, MulticlassClassificationMetrics metrics) LoadTensorFlowModel(
        string baseDir, string[] classes, string modelPath,
        string inputName, string outputName)
    {
        var imageList = new List<ImageData>();

        foreach (var cls in classes)
        {
            var testDir = Path.Combine(baseDir, cls, "model_test"); // brug dine valid/test-billeder
            if (Directory.Exists(testDir))
            {
                foreach (var f in Directory.EnumerateFiles(testDir, "*.*", SearchOption.AllDirectories))
                {
                    imageList.Add(new ImageData { ImagePath = f, Label = cls });
                }
            }
        }

        if (imageList.Count == 0)
            throw new Exception("Ingen billeder fundet i model_test-mapperne!");

        var data = _ml.Data.LoadFromEnumerable(imageList);

        // === PIPELINE ===
        var pipeline = _ml.Transforms.LoadImages(
                            outputColumnName: inputName,
                            imageFolder: "",
                            inputColumnName: nameof(ImageData.ImagePath))
            .Append(_ml.Transforms.ResizeImages(
                            outputColumnName: inputName,
                            imageWidth: 224,
                            imageHeight: 224,
                            inputColumnName: inputName))
            .Append(_ml.Transforms.ExtractPixels(
                            outputColumnName: inputName,
                            inputColumnName: inputName))

            // Kør TensorFlow-model
            .Append(_ml.Model.LoadTensorFlowModel(modelPath)
                    .ScoreTensorFlowModel(
                        outputColumnNames: new[] { outputName },
                        inputColumnNames: new[] { inputName },
                        addBatchDimensionInput: true))

            // Klassifikation ovenpå TensorFlow-output
            .Append(_ml.Transforms.Conversion.MapValueToKey("LabelAsKey", nameof(ImageData.Label)))
            .Append(_ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                        labelColumnName: "LabelAsKey",
                        featureColumnName: outputName))
            .Append(_ml.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

        Console.WriteLine("Bygger pipeline ovenpå TensorFlow...");
        var model = pipeline.Fit(data);

        // Evaluer på valid-data
        var predictions = model.Transform(data);
        var metrics = _ml.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelAsKey");

        return (model, metrics);
    }

    public ImagePrediction Predict(ITransformer model, string imagePath)
    {
        var engine = _ml.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
        return engine.Predict(new ImageData { ImagePath = imagePath });
    }
}
