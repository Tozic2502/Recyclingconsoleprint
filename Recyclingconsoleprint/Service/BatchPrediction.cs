using CsvHelper;
using Domain;
using Recyclingconsoleprint.Service;
using System;
using System.Collections.Generic;
using System.Formats.Asn1;
using System.Globalization;
using System.IO;
using System.Linq;

namespace RecyclingConsolePrint.Service
{
    public class BatchPrediction
    {
        private readonly IImagePredictionService _predictionService;

        public BatchPrediction(IImagePredictionService predictionService)
        {
            _predictionService = predictionService;
        }

        /// <summary>
        /// Kører alle billeder i mappen gennem modellen og gemmer resultatet i CSV.
        /// </summary>
        public void PredictFolder(string folderPath, string outputCsvPath)
        {
            // Find alle billeder
            var images = Directory.GetFiles(folderPath)
                                  .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                                              f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                                  .Select(f => new ImageData { ImagePath = f });

            var predictions = new List<PredictionResult>();

            foreach (var img in images)
            {
                var result = _predictionService.Predict(img);
                predictions.Add(result);
                Console.WriteLine($"{result.ImagePath} -> {result.Label} ({result.Probability:P1})");
            }

            // Gem til CSV
            using var writer = new StreamWriter(outputCsvPath);
            using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
            csv.WriteRecords(predictions);
        }
    }
}
