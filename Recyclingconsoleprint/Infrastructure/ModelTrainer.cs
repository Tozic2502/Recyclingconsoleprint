using Domain;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
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

    public (ITransformer model, MulticlassClassificationMetrics metrics) TrainAndEvaluate(
        string baseDir, string[] classes)
    {
        var trainList = new List<ImageData>();
        var validList = new List<ImageData>();

        foreach (var cls in classes)
        {
            var trainDir = Path.Combine(baseDir, cls, "train");
            var validDir = Path.Combine(baseDir, cls, "model_test");

            if (Directory.Exists(trainDir))
                foreach (var f in Directory.EnumerateFiles(trainDir, "*.*", SearchOption.AllDirectories))
                    trainList.Add(new ImageData { ImagePath = f, Label = cls });

            if (Directory.Exists(validDir))
                foreach (var f in Directory.EnumerateFiles(validDir, "*.*", SearchOption.AllDirectories))
                    validList.Add(new ImageData { ImagePath = f, Label = cls });
        }

        if (trainList.Count == 0 || validList.Count == 0)
            throw new Exception("Ingen billeder fundet i train/model_test!");

        var trainData = _ml.Data.LoadFromEnumerable(trainList);
        var validData = _ml.Data.LoadFromEnumerable(validList);

        var pipeline =
            _ml.Transforms.Conversion.MapValueToKey("LabelAsKey", nameof(ImageData.Label))
              .Append(_ml.Transforms.LoadImages("Image", "", nameof(ImageData.ImagePath)))
              .Append(_ml.Transforms.ResizeImages("Image", 224, 224))
              .Append(_ml.MulticlassClassification.Trainers.ImageClassification(
                        featureColumnName: "Image",
                        labelColumnName: "LabelAsKey",
                        validationSet: validData))
              .Append(_ml.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

        Console.WriteLine("Træner modellen...");
        var model = pipeline.Fit(trainData);

        var predictions = model.Transform(validData);
        var metrics = _ml.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelAsKey");

        return (model, metrics);
    }

    public ImagePrediction Predict(ITransformer model, string imagePath)
    {
        var engine = _ml.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
        return engine.Predict(new ImageData { ImagePath = imagePath });
    }
}

