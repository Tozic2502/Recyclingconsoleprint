using Service;
using Infrastructure;
using Domain;
using System;

namespace RecyclingConsolePrint
{
    class Program
    {
        static void Main(string[] args)
        {
            var modelPath = @"MLModels/saved_model.pb";
            var labelsPath = @"MLModels/labels.txt";
            var imagesFolder = @"ImagesToPredict"; // Opret denne mappe og put billeder ind
            var outputCsv = @"predictions.csv";

            // ML-service
            IImagePredictionService predictionService = (IImagePredictionService)new TeachableMachinePredictionService(modelPath, labelsPath);

            // Batch prediction
            var batchPrediction = new BatchPrediction(predictionService);
            batchPrediction.PredictFolder(imagesFolder, outputCsv);

            Console.WriteLine("Færdig! Resultater gemt til " + outputCsv);
        }
    }
}