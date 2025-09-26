using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

public class ImageData
{
    public string ImagePath { get; set; } = "";
    public string Label { get; set; } = "";
}

public class ImagePrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = "";
    public float[] Score { get; set; } = Array.Empty<float>();
}

class Program
{
    static void Main()
    {
        var baseDir = @"C:\DATA";   // <- her skal I placere jeres mapper: \papir, \pap, \plast, \metal
        var testImage = @"C:\DATA\unknown\test1.jpg";  // et vilkårligt billede til test

        var ml = new MLContext(seed: 1);

        // Jeres fire klasser
        var classes = new[] { "papir", "pap", "plast", "metal" };

        var trainList = new List<ImageData>();
        var validList = new List<ImageData>();

        Console.WriteLine("=== Scanner mapper ===");
        foreach (var cls in classes)
        {
            var trainDir = Path.Combine(baseDir, cls, "train");
            var validDir = Path.Combine(baseDir, cls, "model_test");

            if (Directory.Exists(trainDir))
            {
                foreach (var f in Directory.EnumerateFiles(trainDir, "*.*", SearchOption.AllDirectories))
                    trainList.Add(new ImageData { ImagePath = f, Label = cls });
            }

            if (Directory.Exists(validDir))
            {
                foreach (var f in Directory.EnumerateFiles(validDir, "*.*", SearchOption.AllDirectories))
                    validList.Add(new ImageData { ImagePath = f, Label = cls });
            }
        }

        Console.WriteLine($"Train={trainList.Count}, Test={validList.Count}");

        if (trainList.Count == 0 || validList.Count == 0)
        {
            Console.WriteLine("❌ Mangler billeder i train/model_test!");
            return;
        }

        // ML pipeline
        var trainData = ml.Data.LoadFromEnumerable(trainList);
        var validData = ml.Data.LoadFromEnumerable(validList);

        var pipeline =
            ml.Transforms.Conversion.MapValueToKey("LabelAsKey", nameof(ImageData.Label))
              .Append(ml.Transforms.LoadImages("Image", "", nameof(ImageData.ImagePath)))
              .Append(ml.Transforms.ResizeImages("Image", 224, 224))
              .Append(ml.Transforms.ExtractPixels("Image"))
              .Append(ml.MulticlassClassification.Trainers.ImageClassification(
                        featureColumnName: "Image",
                        labelColumnName: "LabelAsKey",
                        validationSet: validData))
              .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

        Console.WriteLine("\nTræner modellen...");
        var model = pipeline.Fit(trainData);

        var engine = ml.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
        var pred = engine.Predict(new ImageData { ImagePath = testImage });

        Console.WriteLine("\n=== Klassifikation ===");
        Console.WriteLine($"Billede: {testImage}");
        Console.WriteLine($"Forudsagt label: {pred.PredictedLabel}");

        if (pred.Score != null && pred.Score.Length > 0)
        {
            Console.WriteLine("Sandsynligheder:");
            for (int i = 0; i < classes.Length; i++)
                Console.WriteLine($" {classes[i]}: {pred.Score[i]:P2}");
        }
    }
}
