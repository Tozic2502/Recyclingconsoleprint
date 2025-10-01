using Domain;
using Infrastructure;
using System;
using System.IO;

class Program
{
    static void Main()
    {
        var baseDir = @"C:\DATA";
        var testImage = @"C:\DATA\unknown\";
        var classes = new[] { "papir", "pap", "plast", "metal", "glass" };

        var trainer = new ModelTrainer();
        var (model, metrics) = trainer.TrainAndEvaluate(baseDir, classes);

        // Print metrics
        Console.WriteLine("\n=== Model Metrics ===");
        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");
        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

        // Save CSV
        var csvPath = Path.Combine(baseDir, "confusion_matrix.csv");
        CsvExporter.SaveConfusionMatrix(csvPath, classes, metrics.ConfusionMatrix);
        Console.WriteLine($"\nConfusion matrix gemt til: {csvPath}");

        // Test prediction
        var prediction = trainer.Predict(model, testImage);
        Console.WriteLine("\n=== Klassifikation ===");
        Console.WriteLine($"Billede: {testImage}");
        Console.WriteLine($"Forudsagt label: {prediction.PredictedLabel}");
    }
}
