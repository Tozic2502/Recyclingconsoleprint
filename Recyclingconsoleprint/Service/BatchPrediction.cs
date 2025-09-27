using Domain;
using Infrastructure;
using System.Globalization;

namespace Service
{
    public class BatchPrediction
    {
        private readonly IImagePredictionService _predictionService;
        private readonly CsvExporter _csvExporter;

        public BatchPrediction(IImagePredictionService predictionService)
        {
            _predictionService = predictionService;
            _csvExporter = new CsvExporter();
        }

        public void RunPrediction()
        {
            // 💡 Her vælger du mappen med billeder
            string imageFolder = Path.Combine(AppContext.BaseDirectory, "DATA");

            if (!Directory.Exists(imageFolder))
            {
                Console.WriteLine($"❌ Mappen '{imageFolder}' findes ikke!");
                return;
            }

            Console.WriteLine($"🔍 Kører batch prediction på: {imageFolder}");

            var imageFiles = Directory.GetFiles(imageFolder, "*.*", SearchOption.TopDirectoryOnly)
                .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase))
                .ToList();

            if (imageFiles.Count == 0)
            {
                Console.WriteLine("⚠️ Ingen billedfiler fundet i mappen.");
                return;
            }

            var results = new List<PredictionResult>();

            foreach (var file in imageFiles)
            {
                var image = new ImageData { ImagePath = file };
                var prediction = _predictionService.Predict(image);

                Console.WriteLine($"🖼️ {Path.GetFileName(file)} → {prediction.Label} ({prediction.Probability.ToString("P2", CultureInfo.InvariantCulture)})");
                results.Add(prediction);
            }

            // 📦 Eksporter resultater til CSV
            string exportPath = Path.Combine(AppContext.BaseDirectory, "prediction_results.csv");
            _csvExporter.ExportToCsv(results, exportPath);

            Console.WriteLine($"✅ Færdig! Resultater gemt i: {exportPath}");
        }
    }
}
