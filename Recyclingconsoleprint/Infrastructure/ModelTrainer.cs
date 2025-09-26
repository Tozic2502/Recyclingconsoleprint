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
        // ... existing code ...

        var ml = new MLContext(seed: 1);

        // Initial data
       

        // 1. MapValueToKey
        var keyEstimator = ml.Transforms.Conversion.MapValueToKey("LabelAsKey", nameof(ImageData.Label));
        var keySchema = keyEstimator.GetOutputSchema(GetSchemaShape(trainData.Schema));
        Console.WriteLine("After MapValueToKey:");
        foreach (var col in keySchema)
            Console.WriteLine($"{col.Name}: {col.ItemType}");

        // 2. LoadImages
        var imageEstimator = keyEstimator.Append(ml.Transforms.LoadImages("Image", "", nameof(ImageData.ImagePath)));
        var imageSchema = imageEstimator.GetOutputSchema(GetSchemaShape(trainData.Schema));
        Console.WriteLine("After LoadImages:");
        foreach (var col in imageSchema)
            Console.WriteLine($"{col.Name}: {col.ItemType}");

        // 3. ResizeImages
        var resizeEstimator = imageEstimator.Append(ml.Transforms.ResizeImages("Image", 224, 224));
        var resizeSchema = resizeEstimator.GetOutputSchema(GetSchemaShape(trainData.Schema));
        Console.WriteLine("After ResizeImages:");
        foreach (var col in resizeSchema)
            Console.WriteLine($"{col.Name}: {col.ItemType}");

        // 4. ExtractPixels (if added)
        var pixelEstimator = resizeEstimator.Append(ml.Transforms.ExtractPixels("Image"));
        var pixelSchema = pixelEstimator.GetOutputSchema(GetSchemaShape(trainData.Schema));
        Console.WriteLine("After ExtractPixels:");
        foreach (var col in pixelSchema)
            Console.WriteLine($"{col.Name}: {col.ItemType}");

        // Continue building pipeline as needed...

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

    // Helper method to convert DataViewSchema to SchemaShape
    private static SchemaShape GetSchemaShape(DataViewSchema schema)
    {
        var columns = new List<SchemaShape.Column>();
        foreach (var col in schema)
        {
            columns.Add(new SchemaShape.Column(
                col.Name,
                col.Type is VectorDataViewType vType
                    ? (vType.IsVariableSize ? SchemaShape.Column.VectorKind.VariableVector : SchemaShape.Column.VectorKind.Vector)
                    : SchemaShape.Column.VectorKind.Scalar,
                col.Type,
                col.Annotations.GetValue<bool>("IsKey", ref Unsafe.AsRef(false)),
                SchemaShape.Annotations.Empty));
        }
        return new SchemaShape(columns);
    }
}

