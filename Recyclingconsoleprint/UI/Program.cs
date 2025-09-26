using Domain;
using Infrastructure;
using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        var modelPath = @"MLModels/saved_model.pb";
        var labelsPath = @"MLModels/labels.txt";
        var imagesFolder = @"ImagesToPredict";
        var outputCsv = @"predictions.csv";

        var predictionService = new TeachableMachinePredictionService(modelPath, labelsPath);
        var batchPrediction = new BatchPrediction(predictionService);

        batchPrediction.PredictFolder(imagesFolder, outputCsv);

        Console.WriteLine("Færdig! Resultater gemt til " + outputCsv);
    }
}
